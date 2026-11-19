"""Unit & integration tests for the Unified End-to-End Pipeline (Prefill D2/D3 + Decode D1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import mlx.core as mx
import mlx_lm
import numpy as np
import pytest

from direction1_phase_shortlist.build_artifacts import ARTIFACTS_DIR as D1_ARTIFACTS
from direction1_phase_shortlist.phase_router import extract_phase_tag_ids
from direction1_phase_shortlist.shortlist_vocab import load_static_shortlist
from direction2_hybrid_prefill_align.aligned_prefill import compile_hybrid_aligner
from direction2_hybrid_prefill_align.kv_gdn_aligner import load_hybrid_aligner
from direction2_hybrid_prefill_align.train_aligners import ARTIFACTS_DIR as D2_ARTIFACTS
from seed_agent_lab.run_swe_rotation_probe import DEFAULT_MODEL_DIR
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR
from unified_e2e_pipeline import (
    UnifiedSpeculativePipeline,
    baseline_e2e_config,
    decode_only_d1_config,
    ensure_metal_gpu_arch,
    prefill_only_d2_config,
    unified_d23_d1_config,
    unified_d2_d1_config,
)


def test_pipeline_config_presets_are_immutable() -> None:
    cfg = unified_d2_d1_config()
    assert cfg.prefill_policy == "Uniform_k16"
    assert cfg.prefill_align.align_attn_kv
    assert cfg.use_speculative_decode
    assert cfg.use_shortlist
    with pytest.raises(FrozenInstanceError):
        cfg.name = "mutated"  # type: ignore[misc]


def test_unified_pipeline_end_to_end_on_qwen35() -> None:
    ensure_metal_gpu_arch()
    w8_path = CHECKPOINT_DIR / "layer8_ridge_asst.npz"
    sl_path = D1_ARTIFACTS / "shortlist_8192.npz"
    al_path = D2_ARTIFACTS / "hybrid_aligner_L16_r16.npz"
    if not (DEFAULT_MODEL_DIR.exists() and w8_path.exists() and sl_path.exists() and al_path.exists()):
        pytest.skip("Required model or saved D1/D2 artifacts missing")

    model, tokenizer = mlx_lm.load(str(DEFAULT_MODEL_DIR))
    tags = extract_phase_tag_ids(tokenizer)
    eos = frozenset(int(t) for t in getattr(tokenizer, "eos_token_ids", set()))
    w8 = np.load(w8_path)
    sl8 = load_static_shortlist(sl_path)
    compiled = compile_hybrid_aligner(load_hybrid_aligner(al_path), attn_rank=None)

    pipe = UnifiedSpeculativePipeline(
        model=model,
        tokenizer=tokenizer,
        w8_weight=np.asarray(w8["weight"], dtype=np.float32),
        w8_bias=np.asarray(w8["bias"], dtype=np.float32),
        phase_tags=tags,
        static_shortlist=sl8,
        compiled_prefill_aligner=compiled,
    )

    msgs = [
        {"role": "user", "content": "Show a 1-line bash command to count Python files in src/."},
    ]
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tokenizer.encode(rendered)

    base = pipe.generate(pids, max_new_tokens=20, eos_token_ids=eos, config=baseline_e2e_config())
    d1 = pipe.generate(pids, max_new_tokens=20, eos_token_ids=eos, config=decode_only_d1_config())
    d2 = pipe.generate(pids, max_new_tokens=20, eos_token_ids=eos, config=prefill_only_d2_config())
    u21 = pipe.generate(pids, max_new_tokens=20, eos_token_ids=eos, config=unified_d2_d1_config())
    u231 = pipe.generate(pids, max_new_tokens=20, eos_token_ids=eos, config=unified_d23_d1_config())

    # DecodeOnly_D1 must match Baseline_E2E token-for-token
    assert d1.tokens == base.tokens
    # Unified D2+D1 must match PrefillOnly_D2 token-for-token (same Prefill, lossless D1 Decode!)
    assert u21.tokens == d2.tokens
    assert len(u231.tokens) > 0
    assert u21.total_seconds > 0.0
    mx.clear_cache()
