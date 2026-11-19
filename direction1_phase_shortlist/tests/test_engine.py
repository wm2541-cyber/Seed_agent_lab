"""Integration tests for PhaseShortlistSpeculativeEngine verifying greedy exactness and phase gating on Qwen3.5-4B."""

from __future__ import annotations

import mlx.core as mx
import mlx_lm
import numpy as np
import pytest

from direction1_phase_shortlist import ensure_metal_gpu_arch
from direction1_phase_shortlist.engine import PhaseShortlistSpeculativeEngine, _truncate_at_first_eos
from direction1_phase_shortlist.logz_calibrator import LogZCalibrator
from direction1_phase_shortlist.phase_router import (
    extract_phase_tag_ids,
    lookup_only_routing,
    phase_adaptive_routing,
    phase_gated_routing,
    uniform_hybrid_routing,
)
from direction1_phase_shortlist.shortlist_vocab import build_static_shortlist
from seed_agent_lab.run_swe_rotation_probe import DEFAULT_MODEL_DIR
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR


def test_truncate_at_first_eos_pure() -> None:
    src = [10, 20, 99, 30, 40]
    out = _truncate_at_first_eos(src, frozenset({99}))
    assert out == [10, 20, 99]
    assert src == [10, 20, 99, 30, 40]
    assert _truncate_at_first_eos([1, 2, 3], frozenset({99})) == [1, 2, 3]


@pytest.fixture(scope="module")
def loaded_engine_and_tokenizer():
    ensure_metal_gpu_arch()
    w8_path = CHECKPOINT_DIR / "layer8_ridge_asst.npz"
    if not DEFAULT_MODEL_DIR.exists() or not w8_path.exists():
        pytest.skip("Qwen3.5-4B-MLX-8bit or layer8_ridge_asst.npz not available")
    model, tokenizer = mlx_lm.load(str(DEFAULT_MODEL_DIR))
    w8 = np.load(w8_path)
    vocab_size = int(model.language_model.model.embed_tokens.weight.shape[0])
    hidden_dim = int(model.language_model.args.hidden_size)
    tags = extract_phase_tag_ids(tokenizer)
    pinned = [t for t in (tags.think_start, tags.think_end, tags.tool_call_start, tags.tool_call_end, tags.im_end) if t is not None]
    static_sl = build_static_shortlist(
        vocab_size=vocab_size,
        target_size=4096,
        assistant_counts={},
        all_token_counts={},
        pinned_token_ids=pinned,
    )
    # Lightweight valid calibrator for deterministic integration testing
    dummy_cal = LogZCalibrator(
        weight=np.zeros((hidden_dim,), dtype=np.float32),
        norm_weight=0.0,
        bias=12.0,
        ridge_lambda=1.0,
        n_samples=1,
    )
    engine = PhaseShortlistSpeculativeEngine(
        model=model,
        w8_weight=w8["weight"],
        w8_bias=w8["bias"],
        phase_tags=tags,
        static_shortlist=static_sl,
        logz_calibrator=dummy_cal,
    )
    return engine, tokenizer


def test_closed_loop_greedy_exactness_across_phase_and_shortlist_modes(loaded_engine_and_tokenizer) -> None:
    engine, tokenizer = loaded_engine_and_tokenizer
    eos_ids = frozenset(int(t) for t in getattr(tokenizer, "eos_token_ids", set()))

    # Prompt 1: starts INSIDE <think>\n (verifies PhaseGate skips L8 inside thinking while matching greedy baseline)
    messages = [{"role": "user", "content": "Write a 1-line bash command to list Python files under src/."}]
    think_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    think_prompt = tokenizer.encode(think_text)

    base_think, _ = engine.generate_baseline(think_prompt, max_new_tokens=28, eos_token_ids=eos_ids)

    res_pg_full = engine.generate_speculative(
        think_prompt,
        max_new_tokens=28,
        eos_token_ids=eos_ids,
        routing=phase_gated_routing(),
        use_shortlist=False,
    )
    assert res_pg_full.tokens == base_think
    assert res_pg_full.thinking_neural_steps == 0
    assert res_pg_full.thinking_tokens > 0

    res_pg_sl = engine.generate_speculative(
        think_prompt,
        max_new_tokens=28,
        eos_token_ids=eos_ids,
        routing=phase_gated_routing(),
        use_shortlist=True,
        use_logz_calibration=True,
    )
    assert res_pg_sl.tokens == base_think
    assert res_pg_sl.thinking_neural_steps == 0
    assert res_pg_sl.effective_shortlist_size >= 4096

    res_hyb_sl = engine.generate_speculative(
        think_prompt,
        max_new_tokens=28,
        eos_token_ids=eos_ids,
        routing=uniform_hybrid_routing(confidence_threshold=0.70, max_neural_len=4),
        use_shortlist=True,
        use_logz_calibration=True,
    )
    assert res_hyb_sl.tokens == base_think

    res_adapt_sl = engine.generate_speculative(
        think_prompt,
        max_new_tokens=28,
        eos_token_ids=eos_ids,
        routing=phase_adaptive_routing(think_confidence_threshold=0.85, think_max_neural_len=3),
        use_shortlist=True,
        use_logz_calibration=True,
    )
    assert res_adapt_sl.tokens == base_think

    # Prompt 2: starts OUTSIDE <think> (enable_thinking=False -> RESPONSE phase, exercises neural drafting + rollback)
    resp_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    resp_prompt = tokenizer.encode(resp_text)
    base_resp, _ = engine.generate_baseline(resp_prompt, max_new_tokens=24, eos_token_ids=eos_ids)

    res_resp_sl = engine.generate_speculative(
        resp_prompt,
        max_new_tokens=24,
        eos_token_ids=eos_ids,
        routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=5),
        use_shortlist=True,
        use_logz_calibration=True,
    )
    assert res_resp_sl.tokens == base_resp
    assert res_resp_sl.non_thinking_neural_steps > 0

    res_lookup = engine.generate_speculative(
        resp_prompt,
        max_new_tokens=24,
        eos_token_ids=eos_ids,
        routing=lookup_only_routing(),
        use_shortlist=False,
    )
    assert res_lookup.tokens == base_resp
    assert res_lookup.total_neural_steps == 0
    mx.clear_cache()
