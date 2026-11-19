"""Unit tests verifying FreezeB-FitAx orthogonality advantage and HybridPrefillAligner serialization."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from direction2_hybrid_prefill_align.kv_gdn_aligner import (
    AttnKVMap,
    HybridPrefillAligner,
    build_gdn_per_head_low_rank_map,
    load_hybrid_aligner,
    save_hybrid_aligner,
)
from direction2_hybrid_prefill_align.streaming_ridge import (
    AffineLinearMap,
    project_out_head_sinks,
)


def test_freeze_b_fit_ax_beats_fit_full_v_on_orthogonal_content_at_fixed_rank() -> None:
    """When rank-r is tight and a huge Rank-1 sink b exists, freezing b devotes all r ranks to dynamic ax."""
    rng = np.random.default_rng(101)
    hv, dv, d_in = 2, 16, 32
    mean_v = rng.normal(size=(hv, dv))
    v_hat = mean_v / np.linalg.norm(mean_v, axis=-1, keepdims=True)

    # Construct full Ridge weight where each head has: huge Rank-1 sink (norm ~50) + exact Rank-4 content (norm ~1)
    w_blocks = []
    b_blocks = []
    for h in range(hv):
        sink_dir = v_hat[h]  # (dv,)
        sink_w = np.outer(rng.normal(size=(d_in,)) * 10.0, sink_dir)  # Rank 1, huge energy along sink!
        raw_r4 = rng.normal(size=(d_in, 4)) @ rng.normal(size=(4, dv))
        # Project raw_r4 orthogonal to sink_dir
        content_w = raw_r4 - np.outer(raw_r4 @ sink_dir, sink_dir)
        w_blocks.append((sink_w + content_w).astype(np.float32))
        b_blocks.append((sink_dir * 5.0 + rng.normal(size=(dv,)) * 0.2).astype(np.float32))

    full_aff = AffineLinearMap(
        weight=np.concatenate(w_blocks, axis=1),
        bias=np.concatenate(b_blocks, axis=0),
        ridge_lambda=1.0,
        n_samples=100,
    )

    # At Rank 4: FreezeB_FitAx captures 100% of orthogonal content; FitFullV wastes 1 rank on the sink!
    ax_map = build_gdn_per_head_low_rank_map(17, full_aff, mean_v, rank=4, freeze_b=True)
    fullv_map = build_gdn_per_head_low_rank_map(17, full_aff, mean_v, rank=4, freeze_b=False)

    assert ax_map.mean_retained_energy == pytest.approx(1.0, abs=1e-5)

    x_test = rng.normal(size=(40, d_in)).astype(np.float32)
    true_perp = project_out_head_sinks((x_test @ full_aff.weight) + full_aff.bias, mean_v).reshape(40, hv, dv)

    def _eval_gdn_map(gm) -> np.ndarray:
        outs = []
        for h in range(hv):
            outs.append(((x_test @ gm.left[h]) @ gm.right[h]) + gm.bias[h])
        return np.stack(outs, axis=1)  # (40, hv, dv)

    pred_ax = _eval_gdn_map(ax_map)
    pred_fullv = _eval_gdn_map(fullv_map)
    pred_fullv_perp = project_out_head_sinks(pred_fullv.reshape(40, hv * dv), mean_v).reshape(40, hv, dv)

    # FreezeB_FitAx output is strictly orthogonal to sink and recovers exact orthogonal content
    sink_leak = np.max(np.abs(np.sum(pred_ax * v_hat[None, :, :], axis=-1)))
    assert float(sink_leak) < 1e-4

    err_ax = float(np.linalg.norm(pred_ax - true_perp))
    err_fullv = float(np.linalg.norm(pred_fullv_perp - true_perp))
    assert err_ax < 1e-3
    assert err_ax < err_fullv * 0.01  # FreezeB is >100x more accurate on dynamic content ax!


def test_hybrid_aligner_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    kw = rng.normal(size=(16, 8)).astype(np.float32)
    kb = rng.normal(size=(8,)).astype(np.float32)
    vw = rng.normal(size=(16, 8)).astype(np.float32)
    vb = rng.normal(size=(8,)).astype(np.float32)
    attn_map = AttnKVMap(
        layer_idx=19,
        k_map=AffineLinearMap(kw, kb, 100.0, 50),
        v_map=AffineLinearMap(vw, vb, 100.0, 50),
        k_target_is_post_norm=False,
    )
    mv = rng.normal(size=(2, 8)).astype(np.float32)
    ma = rng.normal(size=(2, 8)).astype(np.float32)
    gdn_aff = AffineLinearMap(rng.normal(size=(16, 16)).astype(np.float32), rng.normal(size=(16,)).astype(np.float32), 100.0, 50)
    gdn_ax = build_gdn_per_head_low_rank_map(17, gdn_aff, mv, rank=4, freeze_b=True)
    gdn_fv = build_gdn_per_head_low_rank_map(17, gdn_aff, mv, rank=4, freeze_b=False)
    gdn_k = AffineLinearMap(rng.normal(size=(16, 8)).astype(np.float32), rng.normal(size=(8,)).astype(np.float32), 100.0, 50)

    aligner = HybridPrefillAligner(
        anchor_layer=16,
        attn_kv_maps=(attn_map,),
        gdn_ax_maps=(gdn_ax,),
        gdn_fullv_maps=(gdn_fv,),
        gdn_k_maps=((17, gdn_k),),
        mean_v_by_layer=((17, mv),),
        mean_a_by_layer=((17, ma),),
    )
    saved = save_hybrid_aligner(tmp_path / "aligner.npz", aligner)
    loaded = load_hybrid_aligner(saved)

    assert loaded.anchor_layer == 16
    assert len(loaded.attn_kv_maps) == 1
    np.testing.assert_allclose(loaded.attn_kv_maps[0].k_map.weight, kw)
    np.testing.assert_allclose(loaded.gdn_ax_maps[0].left, gdn_ax.left)
    np.testing.assert_allclose(loaded.gdn_fullv_maps[0].right, gdn_fv.right)
    np.testing.assert_allclose(dict(loaded.mean_v_by_layer)[17], mv)
    assert not loaded.attn_kv_maps[0].k_map.weight.flags.writeable
