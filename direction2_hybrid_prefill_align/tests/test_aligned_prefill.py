"""Integration tests for MLX aligned shallow prefill on Qwen3.5-4B (Attn KV + GDN FreezeB/Ax)."""

from __future__ import annotations

import mlx.core as mx
import mlx_lm
import numpy as np
import pytest

from direction2_hybrid_prefill_align import ensure_metal_gpu_arch
from direction2_hybrid_prefill_align.aligned_prefill import (
    AlignedPrefillConfig,
    compile_hybrid_aligner,
    rms_normalize_hidden,
    run_aligned_shallow_prefill,
)
from direction2_hybrid_prefill_align.eval_and_benchmark import _slice_compiled_gdn_rank
from direction2_hybrid_prefill_align.kv_gdn_aligner import (
    AttnKVMap,
    HybridPrefillAligner,
    build_gdn_per_head_low_rank_map,
)
from direction2_hybrid_prefill_align.streaming_ridge import AffineLinearMap
from seed_agent_lab.run_seed_coding_eval import _score_continuation
from seed_agent_lab.run_swe_rotation_probe import DEFAULT_MODEL_DIR


def test_rms_normalize_hidden_is_scale_invariant() -> None:
    ensure_metal_gpu_arch()
    x = mx.array([[[1.0, -2.0, 3.0, -4.0]]], dtype=mx.bfloat16)
    y1 = rms_normalize_hidden(x)
    y2 = rms_normalize_hidden(x * 7.5)
    mx.eval(y1, y2)
    np.testing.assert_allclose(np.array(y1.astype(mx.float32)), np.array(y2.astype(mx.float32)), atol=1e-2)


def test_aligned_shallow_prefill_runs_cleanly_on_qwen35() -> None:
    ensure_metal_gpu_arch()
    if not DEFAULT_MODEL_DIR.exists():
        pytest.skip("Qwen3.5-4B-MLX-8bit not available")

    model, tokenizer = mlx_lm.load(str(DEFAULT_MODEL_DIR))
    d_model = int(model.language_model.args.hidden_size)
    rng = np.random.default_rng(11)

    # Build lightweight synthetic aligner covering 1 Attn layer (19) and 1 GDN layer (17)
    attn_map = AttnKVMap(
        layer_idx=19,
        k_map=AffineLinearMap(rng.normal(scale=0.01, size=(d_model, 1024)).astype(np.float32), np.zeros((1024,), np.float32), 100.0, 10),
        v_map=AffineLinearMap(rng.normal(scale=0.01, size=(d_model, 1024)).astype(np.float32), np.zeros((1024,), np.float32), 100.0, 10),
        k_target_is_post_norm=False,
    )
    mv = rng.normal(size=(32, 128)).astype(np.float32)
    ma = rng.normal(size=(32, 128)).astype(np.float32)
    gdn_aff = AffineLinearMap(rng.normal(scale=0.01, size=(d_model, 4096)).astype(np.float32), np.zeros((4096,), np.float32), 100.0, 10)
    gdn_ax = build_gdn_per_head_low_rank_map(17, gdn_aff, mv, rank=16, freeze_b=True)
    gdn_fv = build_gdn_per_head_low_rank_map(17, gdn_aff, mv, rank=16, freeze_b=False)
    gdn_k = AffineLinearMap(rng.normal(scale=0.01, size=(d_model, 2048)).astype(np.float32), np.zeros((2048,), np.float32), 100.0, 10)

    aligner = HybridPrefillAligner(
        anchor_layer=16,
        attn_kv_maps=(attn_map,),
        gdn_ax_maps=(gdn_ax,),
        gdn_fullv_maps=(gdn_fv,),
        gdn_k_maps=((17, gdn_k),),
        mean_v_by_layer=((17, mv),),
        mean_a_by_layer=((17, ma),),
    )
    comp_full = compile_hybrid_aligner(aligner, attn_rank=None)
    comp_lr = compile_hybrid_aligner(aligner, attn_rank=64)
    comp_r8 = _slice_compiled_gdn_rank(comp_full, rank=8)
    assert comp_r8.gdn_ax_by_layer[17].v_left.shape == (32, d_model, 8)
    assert comp_r8.gdn_ax_by_layer[17].v_right.shape == (32, 8, 128)

    prompt = tokenizer.encode("def foo(x):\n    return x + 1\n")
    cont = tokenizer.encode("print(foo(5))")

    for comp, cfg in (
        (comp_full, AlignedPrefillConfig(align_attn_kv=True, gdn_mode="freeze_b_fit_ax", align_gdn_k=True, anchor_mean_sink_b=True)),
        (comp_lr, AlignedPrefillConfig(align_attn_kv=True, gdn_mode="fit_full_v", align_gdn_k=False)),
        (comp_r8, AlignedPrefillConfig(align_attn_kv=True, gdn_mode="freeze_b_fit_ax", align_gdn_k=False)),
    ):
        cache = run_aligned_shallow_prefill(model, [(prompt, 16)], compiled=comp, config=cfg)
        assert cache[19].offset == len(prompt)
        logits = _score_continuation(model, cache, cont)
        assert logits.shape[0] == len(cont)
        assert np.all(np.isfinite(logits))
        del cache, logits
        mx.clear_cache()
