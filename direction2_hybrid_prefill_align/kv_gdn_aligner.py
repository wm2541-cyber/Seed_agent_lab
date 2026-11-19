"""Immutable container & serialization for Upgraded Direction 2 (4-Layer Attn KV + 11-Layer GDN FreezeB/Ax) aligners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from direction2_hybrid_prefill_align.streaming_ridge import (
    AffineLinearMap,
    LowRankAffineMap,
    factorize_affine_svd,
    project_out_head_sinks,
)


@dataclass(frozen=True)
class AttnKVMap:
    """Immutable alignment maps predicting pre-RoPE K and V for one deep Full-Attention layer from norm(h_k)."""

    layer_idx: int
    k_map: AffineLinearMap  # (D -> kv_dim)
    v_map: AffineLinearMap  # (D -> kv_dim)
    k_target_is_post_norm: bool


@dataclass(frozen=True)
class GDNHeadValueMap:
    """Immutable per-head low-rank map for one deep GDN layer (supports FreezeB-Ax and FullV ablations).

    Stores per-head SVD factors:
      left:  `(Hv, D, rank)`
      right: `(Hv, rank, Dv)`
      bias:  `(Hv, Dv)`
    """

    layer_idx: int
    mode: str  # "freeze_b_fit_ax" or "fit_full_v"
    rank: int
    left: np.ndarray  # (Hv, D, rank) float32
    right: np.ndarray  # (Hv, rank, Dv) float32
    bias: np.ndarray  # (Hv, Dv) float32
    mean_retained_energy: float
    unit_v_bar: np.ndarray  # (Hv, Dv) float32


@dataclass(frozen=True)
class HybridPrefillAligner:
    """Immutable full checkpoint for Upgraded Direction 2 shallow prefill alignment."""

    anchor_layer: int
    attn_kv_maps: tuple[AttnKVMap, ...]
    gdn_ax_maps: tuple[GDNHeadValueMap, ...]  # FreezeB -> Fit orthogonal ax
    gdn_fullv_maps: tuple[GDNHeadValueMap, ...]  # Ablation -> Fit full v without separating b
    gdn_k_maps: tuple[tuple[int, AffineLinearMap], ...]  # Optional GDN key alignment (D -> 2048)
    mean_v_by_layer: tuple[tuple[int, np.ndarray], ...]
    mean_a_by_layer: tuple[tuple[int, np.ndarray], ...]


def build_gdn_per_head_low_rank_map(
    layer_idx: int,
    full_v_affine: AffineLinearMap,
    mean_v_heads: np.ndarray,
    rank: int,
    freeze_b: bool,
) -> GDNHeadValueMap:
    """Purely build per-head Rank-`r` GDN value maps with either frozen Rank-1 Sink `b` (ax-only) or full `v`.

    When `freeze_b=True`, projects out `v_hat_h` from every head's Ridge weight & bias BEFORE per-head SVD
    truncation so 100% of the Rank-`r` budget models dynamic content `ax` orthogonal to `b`.
    When `freeze_b=False`, truncates full `v` directly at the exact same Rank-`r` (ablation control).
    """
    mv = np.asarray(mean_v_heads, dtype=np.float64)
    hv, dv = mv.shape
    in_dim = int(full_v_affine.weight.shape[0])
    norms = np.linalg.norm(mv, axis=-1, keepdims=True)
    v_hat = np.where(norms > 1e-12, mv / np.maximum(norms, 1e-12), 0.0)

    w_src = full_v_affine.weight.astype(np.float64)
    b_src = full_v_affine.bias.astype(np.float64)
    if freeze_b:
        w_src = project_out_head_sinks(w_src, mv)
        b_src = project_out_head_sinks(b_src, mv)

    r_eff = min(int(rank), dv, in_dim)
    lefts = np.zeros((hv, in_dim, r_eff), dtype=np.float32)
    rights = np.zeros((hv, r_eff, dv), dtype=np.float32)
    biases = np.zeros((hv, dv), dtype=np.float32)
    energies: list[float] = []

    for h in range(hv):
        s_col = h * dv
        e_col = (h + 1) * dv
        head_aff = AffineLinearMap(
            weight=w_src[:, s_col:e_col].astype(np.float32),
            bias=b_src[s_col:e_col].astype(np.float32),
            ridge_lambda=full_v_affine.ridge_lambda,
            n_samples=full_v_affine.n_samples,
        )
        lr = factorize_affine_svd(head_aff, rank=r_eff)
        lefts[h] = lr.left
        rights[h] = lr.right
        biases[h] = lr.bias
        energies.append(lr.retained_energy)

    unit_v = v_hat.astype(np.float32).copy()
    for arr in (lefts, rights, biases, unit_v):
        arr.setflags(write=False)

    return GDNHeadValueMap(
        layer_idx=int(layer_idx),
        mode="freeze_b_fit_ax" if freeze_b else "fit_full_v",
        rank=r_eff,
        left=lefts,
        right=rights,
        bias=biases,
        mean_retained_energy=float(np.mean(energies)),
        unit_v_bar=unit_v,
    )


def truncate_attn_kv_map(kv_map: AttnKVMap, rank: int) -> tuple[LowRankAffineMap, LowRankAffineMap]:
    """Purely factorize a layer's Attn K and V AffineLinearMaps to Rank-`r` via SVD."""
    return factorize_affine_svd(kv_map.k_map, rank=rank), factorize_affine_svd(kv_map.v_map, rank=rank)


def save_hybrid_aligner(path: Path, aligner: HybridPrefillAligner) -> Path:
    """Persist a HybridPrefillAligner cleanly as compressed `.npz`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "anchor_layer": np.asarray([aligner.anchor_layer], dtype=np.int32),
        "attn_layers": np.asarray([m.layer_idx for m in aligner.attn_kv_maps], dtype=np.int32),
        "gdn_ax_layers": np.asarray([m.layer_idx for m in aligner.gdn_ax_maps], dtype=np.int32),
        "gdn_fullv_layers": np.asarray([m.layer_idx for m in aligner.gdn_fullv_maps], dtype=np.int32),
        "gdn_k_layers": np.asarray([idx for idx, _ in aligner.gdn_k_maps], dtype=np.int32),
        "mean_layers": np.asarray([idx for idx, _ in aligner.mean_v_by_layer], dtype=np.int32),
    }

    for m in aligner.attn_kv_maps:
        p = f"attn_{m.layer_idx}"
        payload[f"{p}_kw"] = m.k_map.weight
        payload[f"{p}_kb"] = m.k_map.bias
        payload[f"{p}_vw"] = m.v_map.weight
        payload[f"{p}_vb"] = m.v_map.bias
        payload[f"{p}_meta"] = np.asarray(
            [m.k_map.ridge_lambda, float(m.k_map.n_samples), 1.0 if m.k_target_is_post_norm else 0.0],
            dtype=np.float64,
        )

    for tag, group in (("gdn_ax", aligner.gdn_ax_maps), ("gdn_fullv", aligner.gdn_fullv_maps)):
        for gm in group:
            p = f"{tag}_{gm.layer_idx}"
            payload[f"{p}_left"] = gm.left
            payload[f"{p}_right"] = gm.right
            payload[f"{p}_bias"] = gm.bias
            payload[f"{p}_unit_v"] = gm.unit_v_bar
            payload[f"{p}_meta"] = np.asarray([float(gm.rank), gm.mean_retained_energy], dtype=np.float64)

    for idx, kmap in aligner.gdn_k_maps:
        p = f"gdn_k_{idx}"
        payload[f"{p}_w"] = kmap.weight
        payload[f"{p}_b"] = kmap.bias
        payload[f"{p}_meta"] = np.asarray([kmap.ridge_lambda, float(kmap.n_samples)], dtype=np.float64)

    mv_dict = dict(aligner.mean_v_by_layer)
    ma_dict = dict(aligner.mean_a_by_layer)
    for idx in mv_dict:
        payload[f"mean_v_{idx}"] = np.asarray(mv_dict[idx], dtype=np.float32)
        payload[f"mean_a_{idx}"] = np.asarray(ma_dict[idx], dtype=np.float32)

    np.savez_compressed(path, **payload)
    return path


def load_hybrid_aligner(path: Path) -> HybridPrefillAligner:
    """Load an immutable HybridPrefillAligner from `.npz`."""
    data = np.load(path)
    anchor_layer = int(data["anchor_layer"][0])

    def _ro(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32).copy()
        out.setflags(write=False)
        return out

    attn_maps: list[AttnKVMap] = []
    for l_idx in data["attn_layers"].tolist():
        p = f"attn_{l_idx}"
        meta = data[f"{p}_meta"]
        lam, n_samp, post_norm = float(meta[0]), int(meta[1]), bool(meta[2] > 0.5)
        attn_maps.append(
            AttnKVMap(
                layer_idx=int(l_idx),
                k_map=AffineLinearMap(_ro(data[f"{p}_kw"]), _ro(data[f"{p}_kb"]), lam, n_samp),
                v_map=AffineLinearMap(_ro(data[f"{p}_vw"]), _ro(data[f"{p}_vb"]), lam, n_samp),
                k_target_is_post_norm=post_norm,
            )
        )

    def _load_gdn_group(tag: str, mode_name: str, layers_key: str) -> tuple[GDNHeadValueMap, ...]:
        out: list[GDNHeadValueMap] = []
        for l_idx in data[layers_key].tolist():
            p = f"{tag}_{l_idx}"
            meta = data[f"{p}_meta"]
            out.append(
                GDNHeadValueMap(
                    layer_idx=int(l_idx),
                    mode=mode_name,
                    rank=int(meta[0]),
                    left=_ro(data[f"{p}_left"]),
                    right=_ro(data[f"{p}_right"]),
                    bias=_ro(data[f"{p}_bias"]),
                    mean_retained_energy=float(meta[1]),
                    unit_v_bar=_ro(data[f"{p}_unit_v"]),
                )
            )
        return tuple(out)

    gdn_ax = _load_gdn_group("gdn_ax", "freeze_b_fit_ax", "gdn_ax_layers")
    gdn_fullv = _load_gdn_group("gdn_fullv", "fit_full_v", "gdn_fullv_layers")

    gdn_k_list: list[tuple[int, AffineLinearMap]] = []
    for l_idx in data["gdn_k_layers"].tolist():
        p = f"gdn_k_{l_idx}"
        meta = data[f"{p}_meta"]
        gdn_k_list.append(
            (int(l_idx), AffineLinearMap(_ro(data[f"{p}_w"]), _ro(data[f"{p}_b"]), float(meta[0]), int(meta[1])))
        )

    mean_v_items: list[tuple[int, np.ndarray]] = []
    mean_a_items: list[tuple[int, np.ndarray]] = []
    for l_idx in data["mean_layers"].tolist():
        mean_v_items.append((int(l_idx), _ro(data[f"mean_v_{l_idx}"])))
        mean_a_items.append((int(l_idx), _ro(data[f"mean_a_{l_idx}"])))

    return HybridPrefillAligner(
        anchor_layer=anchor_layer,
        attn_kv_maps=tuple(attn_maps),
        gdn_ax_maps=gdn_ax,
        gdn_fullv_maps=gdn_fullv,
        gdn_k_maps=tuple(gdn_k_list),
        mean_v_by_layer=tuple(mean_v_items),
        mean_a_by_layer=tuple(mean_a_items),
    )
