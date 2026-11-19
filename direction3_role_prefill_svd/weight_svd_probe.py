"""Pretrained weight SVD and Rank-1 Sink subspace alignment probe for Qwen3.5-4B GDN and Full Attention layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mlx.core as mx
import numpy as np

from direction3_role_prefill_svd.svd_analysis import (
    SingularSpectrum,
    WeightSinkAlignment,
    analyze_head_weight_sink_alignment,
    compute_singular_spectrum,
)


@dataclass(frozen=True)
class LayerWeightSVDRecord:
    """Immutable SVD summary of one Qwen3.5 layer's core projection weights."""

    layer_idx: int
    layer_type: str  # "gdn" or "full_attn"
    proj_spectra: tuple[tuple[str, SingularSpectrum], ...]
    # GDN-only sink alignment aggregates across the 32 value heads (None for full_attn)
    mean_sink_top1_cos: float | None = None
    max_sink_top1_cos: float | None = None
    mean_sink_top4_energy: float | None = None
    mean_sink_wv_energy_frac: float | None = None
    mean_wout_sink_gain: float | None = None
    mean_wv_head_eff_rank: float | None = None
    mean_wv_perp_eff_rank: float | None = None
    mean_wv_head_r90: float | None = None
    mean_wv_perp_r90: float | None = None


def dequantize_linear_weight(module) -> np.ndarray:
    """Purely extract a `(out_features, in_features)` float32 weight matrix from QuantizedLinear or Linear."""
    if hasattr(module, "scales") and hasattr(module, "biases"):
        w = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            group_size=module.group_size,
            bits=module.bits,
        ).astype(mx.float32)
    else:
        w = module.weight.astype(mx.float32)
    mx.eval(w)
    out = np.array(w, dtype=np.float32, copy=True)
    del w
    mx.clear_cache()
    return out


def analyze_model_weights_svd(
    model,
    mean_v_by_layer: Mapping[int, np.ndarray] | None = None,
) -> tuple[LayerWeightSVDRecord, ...]:
    """Stream across all 32 layers of Qwen3.5-4B to compute weight SVD and GDN sink-subspace alignment."""
    records: list[LayerWeightSVDRecord] = []

    for idx, layer in enumerate(model.layers):
        if layer.is_linear:
            gdn = layer.linear_attn
            key_dim = int(gdn.num_k_heads * gdn.head_k_dim)
            num_v_heads = int(gdn.num_v_heads)
            head_v_dim = int(gdn.head_v_dim)

            w_qkv = dequantize_linear_weight(gdn.in_proj_qkv)  # (8192, 2560)
            w_q = w_qkv[:key_dim, :]
            w_k = w_qkv[key_dim : 2 * key_dim, :]
            w_v = w_qkv[2 * key_dim :, :]  # (4096, 2560)
            w_z = dequantize_linear_weight(gdn.in_proj_z)  # (4096, 2560)
            w_out = dequantize_linear_weight(gdn.out_proj)  # (2560, 4096)

            proj_spectra = (
                ("W_Q", compute_singular_spectrum(w_q, store_top_k=32)),
                ("W_K", compute_singular_spectrum(w_k, store_top_k=32)),
                ("W_V", compute_singular_spectrum(w_v, store_top_k=32)),
                ("W_Z", compute_singular_spectrum(w_z, store_top_k=32)),
                ("W_out", compute_singular_spectrum(w_out, store_top_k=32)),
            )

            mean_cos = max_cos = mean_top4 = mean_wv_frac = mean_gain = None
            mean_eff = mean_perp_eff = mean_r90 = mean_perp_r90 = None

            if mean_v_by_layer is not None and idx in mean_v_by_layer:
                mv = mean_v_by_layer[idx]  # (Hv, Dv)
                alignments: list[WeightSinkAlignment] = [
                    analyze_head_weight_sink_alignment(
                        head_idx=h,
                        w_v_head=w_v[h * head_v_dim : (h + 1) * head_v_dim, :],
                        w_out_head=w_out[:, h * head_v_dim : (h + 1) * head_v_dim],
                        mean_v_head=mv[h],
                    )
                    for h in range(num_v_heads)
                ]
                mean_cos = float(np.mean([a.sink_top1_u_cos for a in alignments]))
                max_cos = float(np.max([a.sink_top1_u_cos for a in alignments]))
                mean_top4 = float(np.mean([a.sink_top4_subspace_energy for a in alignments]))
                mean_wv_frac = float(np.mean([a.sink_wv_energy_fraction for a in alignments]))
                mean_gain = float(np.mean([a.wout_sink_gain_ratio for a in alignments]))
                mean_eff = float(np.mean([a.wv_spectrum.effective_rank for a in alignments]))
                mean_perp_eff = float(np.mean([a.wv_perp_spectrum.effective_rank for a in alignments]))
                mean_r90 = float(np.mean([a.wv_spectrum.r90 for a in alignments]))
                mean_perp_r90 = float(np.mean([a.wv_perp_spectrum.r90 for a in alignments]))

            records.append(
                LayerWeightSVDRecord(
                    layer_idx=idx,
                    layer_type="gdn",
                    proj_spectra=proj_spectra,
                    mean_sink_top1_cos=mean_cos,
                    max_sink_top1_cos=max_cos,
                    mean_sink_top4_energy=mean_top4,
                    mean_sink_wv_energy_frac=mean_wv_frac,
                    mean_wout_sink_gain=mean_gain,
                    mean_wv_head_eff_rank=mean_eff,
                    mean_wv_perp_eff_rank=mean_perp_eff,
                    mean_wv_head_r90=mean_r90,
                    mean_wv_perp_r90=mean_perp_r90,
                )
            )
            del w_qkv, w_q, w_k, w_v, w_z, w_out
        else:
            attn = layer.self_attn
            w_k = dequantize_linear_weight(attn.k_proj)
            w_v = dequantize_linear_weight(attn.v_proj)
            w_o = dequantize_linear_weight(attn.o_proj)
            proj_spectra = (
                ("W_K", compute_singular_spectrum(w_k, store_top_k=32)),
                ("W_V", compute_singular_spectrum(w_v, store_top_k=32)),
                ("W_O", compute_singular_spectrum(w_o, store_top_k=32)),
            )
            records.append(
                LayerWeightSVDRecord(
                    layer_idx=idx,
                    layer_type="full_attn",
                    proj_spectra=proj_spectra,
                )
            )
            del w_k, w_v, w_o
        mx.clear_cache()

    return tuple(records)


def summarize_weight_svd_by_depth(
    records: tuple[LayerWeightSVDRecord, ...],
    split_layer: int = 16,
) -> dict[str, dict[str, float]]:
    """Purely aggregate weight SVD statistics comparing shallow (0..split-1) vs deep (split..31) layers."""
    summary: dict[str, dict[str, float]] = {}

    for band_name, pred in (
        ("gdn_shallow_0_15", lambda r: r.layer_type == "gdn" and r.layer_idx < split_layer),
        ("gdn_deep_16_30", lambda r: r.layer_type == "gdn" and r.layer_idx >= split_layer),
        ("attn_shallow_3_15", lambda r: r.layer_type == "full_attn" and r.layer_idx < split_layer),
        ("attn_deep_19_31", lambda r: r.layer_type == "full_attn" and r.layer_idx >= split_layer),
    ):
        subset = [r for r in records if pred(r)]
        if not subset:
            continue
        band_stats: dict[str, float] = {"n_layers": float(len(subset))}
        proj_names = [name for name, _ in subset[0].proj_spectra]
        for pname in proj_names:
            specs = [dict(r.proj_spectra)[pname] for r in subset]
            band_stats[f"{pname}_eff_rank"] = float(np.mean([s.effective_rank for s in specs]))
            band_stats[f"{pname}_r90"] = float(np.mean([s.r90 for s in specs]))
            band_stats[f"{pname}_r95"] = float(np.mean([s.r95 for s in specs]))
            band_stats[f"{pname}_top16_energy"] = float(np.mean([s.top16_energy for s in specs]))
        if subset[0].mean_sink_top1_cos is not None:
            band_stats["sink_top1_cos"] = float(np.mean([r.mean_sink_top1_cos for r in subset if r.mean_sink_top1_cos is not None]))
            band_stats["sink_top4_energy"] = float(np.mean([r.mean_sink_top4_energy for r in subset if r.mean_sink_top4_energy is not None]))
            band_stats["sink_wv_energy_frac"] = float(np.mean([r.mean_sink_wv_energy_frac for r in subset if r.mean_sink_wv_energy_frac is not None]))
            band_stats["wout_sink_gain"] = float(np.mean([r.mean_wout_sink_gain for r in subset if r.mean_wout_sink_gain is not None]))
            band_stats["wv_head_eff_rank"] = float(np.mean([r.mean_wv_head_eff_rank for r in subset if r.mean_wv_head_eff_rank is not None]))
            band_stats["wv_perp_eff_rank"] = float(np.mean([r.mean_wv_perp_eff_rank for r in subset if r.mean_wv_perp_eff_rank is not None]))
            band_stats["wv_head_r90"] = float(np.mean([r.mean_wv_head_r90 for r in subset if r.mean_wv_head_r90 is not None]))
            band_stats["wv_perp_r90"] = float(np.mean([r.mean_wv_perp_r90 for r in subset if r.mean_wv_perp_r90 is not None]))
        summary[band_name] = band_stats
    return summary
