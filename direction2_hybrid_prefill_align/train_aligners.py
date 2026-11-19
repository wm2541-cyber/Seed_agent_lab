"""Train Upgraded Direction 2 aligners strictly on the 48 SWE training traces via shared-Gram closed-form Ridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
import numpy as np

from direction2_hybrid_prefill_align import ensure_metal_gpu_arch
from direction2_hybrid_prefill_align.aligned_prefill import rms_normalize_hidden
from direction2_hybrid_prefill_align.kv_gdn_aligner import (
    AttnKVMap,
    HybridPrefillAligner,
    build_gdn_per_head_low_rank_map,
    save_hybrid_aligner,
)
from direction2_hybrid_prefill_align.streaming_ridge import (
    init_shared_gram,
    solve_shared_gram_ridge,
    update_shared_gram,
)
from seed_agent_lab.run_seed_coding_eval import estimate_mean_values_and_sinks
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    _prepare_trajectory,
)
from seed_agent_lab.train_and_eval_adapter import EXTRA_TRAIN_PATH

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _extract_trace_anchor_and_deep_targets(
    model,
    token_ids: Sequence[int],
    anchor_layer: int,
    attn_layers: Sequence[int],
    gdn_layers: Sequence[int],
    k_target_is_post_norm: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run one exact 32-layer forward pass collecting `rms_norm(h_anchor)` and deep Attn K/V + GDN K/V targets."""
    text_model = model.language_model.model
    attn_set = set(attn_layers)
    gdn_set = set(gdn_layers)

    hidden = text_model.embed_tokens(mx.array([list(token_ids)], dtype=mx.int32))
    fa_mask = create_attention_mask(hidden, None)
    ssm_mask = create_ssm_mask(hidden, None)

    anchor_x: np.ndarray | None = None
    targets: dict[str, np.ndarray] = {}

    for idx, layer in enumerate(model.layers):
        if idx == anchor_layer:
            anc = rms_normalize_hidden(hidden)[0].astype(mx.float32)
            mx.eval(anc)
            anchor_x = np.array(anc, dtype=np.float32, copy=True)

        if idx in attn_set and not layer.is_linear:
            attn = layer.self_attn
            x_norm = layer.input_layernorm(hidden)
            B, S, _ = x_norm.shape
            k_arr = attn.k_proj(x_norm)
            if k_target_is_post_norm:
                k_arr = attn.k_norm(k_arr.reshape(B, S, attn.num_key_value_heads, -1)).reshape(B, S, -1)
            v_arr = attn.v_proj(x_norm)
            k_f32 = k_arr[0].astype(mx.float32)
            v_f32 = v_arr[0].astype(mx.float32)
            mx.eval(k_f32, v_f32)
            targets[f"attn_{idx}_k"] = np.array(k_f32, dtype=np.float32, copy=True)
            targets[f"attn_{idx}_v"] = np.array(v_f32, dtype=np.float32, copy=True)
            del x_norm, k_arr, v_arr, k_f32, v_f32

        if idx in gdn_set and layer.is_linear:
            gdn = layer.linear_attn
            x_norm = layer.input_layernorm(hidden)
            B, S, _ = x_norm.shape
            qkv = gdn.in_proj_qkv(x_norm)
            conv_pad = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x_norm.dtype)
            conv_out = nn.silu(gdn.conv1d(mx.concatenate([conv_pad, qkv], axis=1)))
            k_f32 = conv_out[0, :, gdn.key_dim : 2 * gdn.key_dim].astype(mx.float32)
            v_f32 = conv_out[0, :, 2 * gdn.key_dim :].astype(mx.float32)
            mx.eval(k_f32, v_f32)
            targets[f"gdn_{idx}_k"] = np.array(k_f32, dtype=np.float32, copy=True)
            targets[f"gdn_{idx}_v"] = np.array(v_f32, dtype=np.float32, copy=True)
            del x_norm, qkv, conv_pad, conv_out, k_f32, v_f32

        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=None)
        if idx % 8 == 7:
            mx.eval(hidden)

    del hidden, fa_mask, ssm_mask
    mx.clear_cache()
    assert anchor_x is not None
    return anchor_x, targets


def train_and_save_aligner(
    model_dir: Path = DEFAULT_MODEL_DIR,
    base_data: Path = DEFAULT_DATA_PATH,
    extra_train: Path = EXTRA_TRAIN_PATH,
    out_dir: Path = ARTIFACTS_DIR,
    anchor_layer: int = 16,
    max_tokens: int = 2560,
    ridge_lambda: float = 200.0,
    gdn_rank: int = 16,
    n_sink_traces: int = 12,
    k_target_is_post_norm: bool = False,
) -> Path:
    """Fit 4-layer Attn KV + 11-layer GDN (FreezeB-Ax & FullV) maps strictly on the 48 SWE training traces."""
    ensure_metal_gpu_arch()
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = [json.loads(line) for line in base_data.open("r", encoding="utf-8") if line.strip()]
    extra_rows = (
        [json.loads(line) for line in extra_train.open("r", encoding="utf-8") if line.strip()]
        if extra_train.exists()
        else []
    )
    train_rows = base_rows[:12] + extra_rows

    model, tokenizer = mlx_lm.load(str(model_dir))
    hidden_dim = int(model.language_model.args.hidden_size)
    num_layers = len(model.layers)

    attn_layers = tuple(i for i in range(anchor_layer, num_layers) if not model.layers[i].is_linear)
    # Layer `anchor_layer` itself already receives exact h_anchor (0.0 error), so align GDN layers > anchor_layer.
    gdn_layers = tuple(i for i in range(anchor_layer + 1, num_layers) if model.layers[i].is_linear)

    sample_attn = model.layers[attn_layers[0]].self_attn
    attn_kv_dim = int(sample_attn.num_key_value_heads * sample_attn.head_dim)
    sample_gdn = model.layers[gdn_layers[0]].linear_attn
    gdn_k_dim = int(sample_gdn.key_dim)
    gdn_v_dim = int(sample_gdn.value_dim)

    print(
        f"Training Direction 2 Aligner (anchor=L{anchor_layer}, attn={attn_layers}, gdn={gdn_layers}, "
        f"train_rows={len(train_rows)}, max_tokens={max_tokens}, lambda={ridge_lambda}, gdn_rank={gdn_rank})..."
    )

    # Step 1: Estimate training-only Rank-1 Sink anchors (mean_v, mean_a) on first `n_sink_traces` train traces
    sink_seqs = []
    for row in train_rows[:n_sink_traces]:
        item = _prepare_trajectory(tokenizer, row, max_tokens=min(max_tokens, 2048))
        if item is not None:
            sink_seqs.append(item[0])
    mean_v, mean_a = estimate_mean_values_and_sinks(model, sink_seqs)
    print(f"Estimated training mean_v & mean_a across {len(sink_seqs)} traces ({time.perf_counter() - t0:.1f}s).")

    # Step 2: Stream shared-Gram Ridge accumulation across all 48 training traces
    target_dims: dict[str, int] = {}
    for l_idx in attn_layers:
        target_dims[f"attn_{l_idx}_k"] = attn_kv_dim
        target_dims[f"attn_{l_idx}_v"] = attn_kv_dim
    for l_idx in gdn_layers:
        target_dims[f"gdn_{l_idx}_k"] = gdn_k_dim
        target_dims[f"gdn_{l_idx}_v"] = gdn_v_dim

    acc = init_shared_gram(hidden_dim, target_dims)
    n_used = 0
    for row in train_rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is None:
            continue
        token_ids, _tags, _mask = item
        anc_x, targs = _extract_trace_anchor_and_deep_targets(
            model=model,
            token_ids=token_ids,
            anchor_layer=anchor_layer,
            attn_layers=attn_layers,
            gdn_layers=gdn_layers,
            k_target_is_post_norm=k_target_is_post_norm,
        )
        acc = update_shared_gram(acc, anc_x, targs)
        n_used += 1
        del anc_x, targs

    print(
        f"Accumulated shared Gram over {n_used} train traces ({acc.n_samples} tokens) in {time.perf_counter() - t0:.1f}s. Solving Ridge..."
    )
    t_solve = time.perf_counter()
    solved = solve_shared_gram_ridge(acc, ridge_lambda=ridge_lambda)
    print(f"Solved all {len(solved)} target maps via single Cholesky in {time.perf_counter() - t_solve:.2f}s.")

    attn_kv_maps = tuple(
        AttnKVMap(
            layer_idx=l_idx,
            k_map=solved[f"attn_{l_idx}_k"],
            v_map=solved[f"attn_{l_idx}_v"],
            k_target_is_post_norm=k_target_is_post_norm,
        )
        for l_idx in attn_layers
    )
    gdn_ax_maps = tuple(
        build_gdn_per_head_low_rank_map(
            layer_idx=l_idx,
            full_v_affine=solved[f"gdn_{l_idx}_v"],
            mean_v_heads=mean_v[l_idx],
            rank=gdn_rank,
            freeze_b=True,
        )
        for l_idx in gdn_layers
    )
    gdn_fullv_maps = tuple(
        build_gdn_per_head_low_rank_map(
            layer_idx=l_idx,
            full_v_affine=solved[f"gdn_{l_idx}_v"],
            mean_v_heads=mean_v[l_idx],
            rank=gdn_rank,
            freeze_b=False,
        )
        for l_idx in gdn_layers
    )
    gdn_k_maps = tuple((l_idx, solved[f"gdn_{l_idx}_k"]) for l_idx in gdn_layers)

    print(
        f"Per-head Rank-{gdn_rank} retained energy: "
        f"FreezeB_FitAx={np.mean([m.mean_retained_energy for m in gdn_ax_maps])*100:.2f}% | "
        f"FitFullV={np.mean([m.mean_retained_energy for m in gdn_fullv_maps])*100:.2f}%"
    )

    aligner = HybridPrefillAligner(
        anchor_layer=anchor_layer,
        attn_kv_maps=attn_kv_maps,
        gdn_ax_maps=gdn_ax_maps,
        gdn_fullv_maps=gdn_fullv_maps,
        gdn_k_maps=gdn_k_maps,
        mean_v_by_layer=tuple((l_idx, mean_v[l_idx]) for l_idx in sorted(mean_v.keys())),
        mean_a_by_layer=tuple((l_idx, mean_a[l_idx]) for l_idx in sorted(mean_a.keys())),
    )
    out_path = save_hybrid_aligner(out_dir / f"hybrid_aligner_L{anchor_layer}_r{gdn_rank}.npz", aligner)
    print(f"Saved HybridPrefillAligner ({out_path.stat().st_size / 1e6:.2f} MB) -> {out_path} in {time.perf_counter() - t0:.1f}s")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Upgraded Direction 2 aligners on 48 SWE training traces.")
    parser.add_argument("--anchor-layer", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2560)
    parser.add_argument("--ridge-lambda", type=float, default=200.0)
    parser.add_argument("--gdn-rank", type=int, default=16)
    parser.add_argument("--n-sink-traces", type=int, default=12)
    args = parser.parse_args()
    train_and_save_aligner(
        anchor_layer=args.anchor_layer,
        max_tokens=args.max_tokens,
        ridge_lambda=args.ridge_lambda,
        gdn_rank=args.gdn_rank,
        n_sink_traces=args.n_sink_traces,
    )


if __name__ == "__main__":
    main()
