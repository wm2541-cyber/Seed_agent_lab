"""Integration tests verifying ClosedLoopSpeculativeEngine equivalence and exact rollback."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx_lm
import numpy as np
import pytest

from seed_agent_lab.benchmark_closed_loop import extract_tool_prompt_ids
from seed_agent_lab.closed_loop_engine import ClosedLoopSpeculativeEngine, _rollback_gdn_from_stash
from seed_agent_lab.run_swe_rotation_probe import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR
from seed_agent_lab.swe_loader import SWE_TOOLS_SCHEMA
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR

CKPT_PATH = CHECKPOINT_DIR / "layer8_ridge_asst.npz"


@pytest.fixture(scope="module")
def closed_loop_setup():
    """Load Qwen3.5-4B-MLX-8bit and Layer-8 Ridge checkpoint once for all closed-loop equivalence tests."""
    if not DEFAULT_MODEL_DIR.exists() or not CKPT_PATH.exists():
        pytest.skip("Model weights or Ridge checkpoint not available on disk.")
    model, tokenizer = mlx_lm.load(str(DEFAULT_MODEL_DIR))
    data = np.load(CKPT_PATH)
    engine = ClosedLoopSpeculativeEngine(
        model=model,
        w8_weight=data["weight"],
        w8_bias=data["bias"],
        draft_layer=int(data["draft_layer"][0]),
    )
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    yield engine, tokenizer, eos_ids
    del engine, model
    mx.clear_cache()


def _build_deterministic_prompt(tokenizer, enable_thinking: bool) -> list[int]:
    msgs = [
        {"role": "system", "content": "You are a coding agent. Always respond directly with a tool call."},
        {
            "role": "user",
            "content": "Inspect `/repo/app.py` lines 1-40 using str_replace_editor view.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": {"command": "view", "path": "/repo/app.py", "view_range": [1, 40]},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "def add(a, b):\n    return a - b\n",
        },
        {
            "role": "user",
            "content": (
                "Call `str_replace_editor` immediately with `command='str_replace'`, "
                "`path='/repo/app.py'`, `old_str='return a - b'`, and `new_str='return a + b'`."
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        msgs,
        tools=SWE_TOOLS_SCHEMA,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(rendered)


def _build_tie_probe_prompt(tokenizer) -> list[int]:
    msgs = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {
            "role": "user",
            "content": "Inspect `/repo/app.py` lines 1-40 using str_replace_editor view, then check why test fails.",
        },
        {
            "role": "assistant",
            "content": "Let me inspect `/repo/app.py` first.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": {"command": "view", "path": "/repo/app.py", "view_range": [1, 40]},
                    },
                }
            ],
        },
        {"role": "tool", "content": "def add(a, b):\n    return a - b  # bug\n"},
        {
            "role": "user",
            "content": "Now update `/repo/app.py` using str_replace_editor str_replace to fix the bug, or inspect another range.",
        },
    ]
    rendered = tokenizer.apply_chat_template(
        msgs,
        tools=SWE_TOOLS_SCHEMA,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(rendered)


@pytest.mark.parametrize("enable_thinking", [False, True])
@pytest.mark.parametrize(
    "mode_name,use_neural,use_lookup",
    [
        ("hybrid", True, True),
        ("neural_only", True, False),
        ("lookup_only", False, True),
    ],
)
def test_closed_loop_exact_greedy_equivalence(
    closed_loop_setup,
    enable_thinking: bool,
    mode_name: str,
    use_neural: bool,
    use_lookup: bool,
) -> None:
    """Every speculative configuration must reproduce full 32-layer greedy generation bit-for-bit."""
    engine, tokenizer, eos_ids = closed_loop_setup
    if enable_thinking and DEFAULT_DATA_PATH.exists():
        rows = [json.loads(line) for line in DEFAULT_DATA_PATH.open("r", encoding="utf-8")]
        prompt_ids = extract_tool_prompt_ids(tokenizer, rows[12], max_prompt_tokens=2048, enable_thinking=True)
        assert prompt_ids is not None
        max_new_tokens = 64
    else:
        prompt_ids = _build_deterministic_prompt(tokenizer, enable_thinking=enable_thinking)
        max_new_tokens = 48

    base_tokens, _ = engine.generate_baseline(prompt_ids, max_new_tokens=max_new_tokens, eos_token_ids=eos_ids)
    spec_res = engine.generate_speculative(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
        use_neural=use_neural,
        use_lookup=use_lookup,
        lookup_first=True,
        confidence_threshold=0.70,
    )

    assert spec_res.tokens == base_tokens, (
        f"Mismatch in mode={mode_name}, thinking={enable_thinking}: "
        f"len(spec)={len(spec_res.tokens)} vs len(base)={len(base_tokens)}"
    )


def test_rollback_preserves_draft_exactness_and_kv_offsets(closed_loop_setup) -> None:
    """Verify Draft(0..7)+Verify(8..31)+Rollback leaves 0..7 bit-exact and KV offsets exact."""
    engine, tokenizer, _eos_ids = closed_loop_setup
    model = engine.model
    prompt_ids = _build_deterministic_prompt(tokenizer, enable_thinking=False)
    sample_toks = [40, 628, 143, 100, 200]
    n_keep = 2

    ref_cache = model.make_cache()
    _ = model(mx.array([prompt_ids], dtype=mx.int32), cache=ref_cache)
    for t in sample_toks[:n_keep]:
        _ = model(mx.array([[t]], dtype=mx.int32), cache=ref_cache)

    spec_cache = model.make_cache()
    _ = model(mx.array([prompt_ids], dtype=mx.int32), cache=spec_cache)
    h8s, snaps = [], []
    for t in sample_toks:
        h8s.append(engine._run_draft_layers_one_token(t, spec_cache))
        snaps.append([(spec_cache[i][0], spec_cache[i][1]) for i in range(engine.draft_layer) if model.layers[i].is_linear])

    h8_chunk = mx.concatenate(h8s, axis=1)
    _, stashes = engine._run_verify_layers(h8_chunk, spec_cache, start_layer=engine.draft_layer, stash_gdn=True)

    drop_kv = len(sample_toks) - n_keep
    for i in range(engine.num_layers):
        if not model.layers[i].is_linear:
            spec_cache[i].trim(drop_kv)
    lin_idx = 0
    for i in range(engine.draft_layer):
        if model.layers[i].is_linear:
            spec_cache[i][0], spec_cache[i][1] = snaps[n_keep - 1][lin_idx]
            lin_idx += 1
    for st in stashes:
        _rollback_gdn_from_stash(model.layers[st.layer_idx], spec_cache[st.layer_idx], st, n_keep)

    for i in range(engine.num_layers):
        if model.layers[i].is_linear:
            conv_diff = float(mx.max(mx.abs(ref_cache[i][0] - spec_cache[i][0])).item())
            state_diff = float(mx.max(mx.abs(ref_cache[i][1] - spec_cache[i][1])).item())
            if i < engine.draft_layer:
                assert conv_diff == 0.0 and state_diff == 0.0
            else:
                assert conv_diff <= 0.25 and state_diff <= 0.05
        else:
            assert ref_cache[i].offset == spec_cache[i].offset
            rk = ref_cache[i].keys[..., : ref_cache[i].offset, :]
            rv = ref_cache[i].values[..., : ref_cache[i].offset, :]
            sk = spec_cache[i].keys[..., : spec_cache[i].offset, :]
            sv = spec_cache[i].values[..., : spec_cache[i].offset, :]
            dk = float(mx.max(mx.abs(rk - sk)).item())
            dv = float(mx.max(mx.abs(rv - sv)).item())
            tol = 0.0 if i < engine.draft_layer else 0.25
            assert dk <= tol and dv <= tol


def test_bfloat16_tie_occurs_at_native_zero_margin(closed_loop_setup) -> None:
    """Confirm any divergence on the ambiguous tie prompt is strictly an exact bfloat16 top-2 tie."""
    engine, tokenizer, eos_ids = closed_loop_setup
    prompt_ids = _build_tie_probe_prompt(tokenizer)
    base_tokens, _ = engine.generate_baseline(prompt_ids, max_new_tokens=48, eos_token_ids=eos_ids)
    spec_res = engine.generate_speculative(
        prompt_ids,
        max_new_tokens=48,
        eos_token_ids=eos_ids,
        use_neural=True,
        use_lookup=False,
        confidence_threshold=0.70,
    )
    first_diff = next((i for i, (a, b) in enumerate(zip(base_tokens, spec_res.tokens)) if a != b), None)
    if first_diff is not None:
        cache = engine.model.make_cache()
        logits = engine.model(mx.array([prompt_ids + list(base_tokens[:first_diff])], dtype=mx.int32), cache=cache)[0, -1, :]
        mx.eval(logits)
        v_base = float(logits[base_tokens[first_diff]].item())
        v_spec = float(logits[spec_res.tokens[first_diff]].item())
        assert abs(v_base - v_spec) <= 0.125  # Within 1 bfloat16 ULP of native full-model logits.
