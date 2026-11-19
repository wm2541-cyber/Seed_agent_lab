"""Pure functional SVD spectral diagnostics for GDN states (b + ax), state errors, and pretrained weights."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from seed_agent_lab.seed_ops import decompose_state_sink


@dataclass(frozen=True)
class SingularSpectrum:
    """Immutable singular-value spectrum summary for a 2D matrix."""

    shape: tuple[int, int]
    frobenius_norm: float
    effective_rank: float
    r50: int
    r80: int
    r90: int
    r95: int
    r99: int
    top1_energy: float
    top8_energy: float
    top16_energy: float
    condition_ratio_1_16: float
    top_singular_values: tuple[float, ...]


@dataclass(frozen=True)
class StateErrorDecomposition:
    """Immutable orthogonal breakdown of (S_true - S_shallow) into Rank-1 sink error E_b and content error E_ax."""

    total_err_norm_sq: float
    sink_err_norm_sq: float
    content_err_norm_sq: float
    sink_err_share: float
    content_err_share: float
    rel_sink_err: float
    rel_content_err: float
    content_err_spectrum: SingularSpectrum


@dataclass(frozen=True)
class WeightSinkAlignment:
    """Immutable alignment between a GDN head's empirical Rank-1 sink direction and its pretrained W_V / W_out."""

    head_idx: int
    wv_spectrum: SingularSpectrum
    wv_perp_spectrum: SingularSpectrum
    sink_top1_u_cos: float
    sink_top4_subspace_energy: float
    sink_wv_energy_fraction: float
    wout_sink_gain_ratio: float


def _rank_for_energy(cum_energy: np.ndarray, threshold: float) -> int:
    """Return smallest 1-based rank reaching `threshold` of cumulative squared Frobenius energy."""
    if cum_energy.size == 0:
        return 0
    idx = int(np.searchsorted(cum_energy, threshold, side="left"))
    return min(idx + 1, int(cum_energy.size))


def compute_singular_spectrum(
    matrix: np.ndarray,
    store_top_k: int = 32,
) -> SingularSpectrum:
    """Purely compute singular value spectrum and effective rank metrics of a 2D matrix."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
    svals = np.linalg.svd(arr, full_matrices=False, compute_uv=False)
    s2 = svals * svals
    total_energy = float(np.sum(s2))
    frob = math.sqrt(total_energy)

    if total_energy <= 1e-18 or svals.size == 0:
        return SingularSpectrum(
            shape=(int(arr.shape[0]), int(arr.shape[1])),
            frobenius_norm=0.0,
            effective_rank=0.0,
            r50=0,
            r80=0,
            r90=0,
            r95=0,
            r99=0,
            top1_energy=0.0,
            top8_energy=0.0,
            top16_energy=0.0,
            condition_ratio_1_16=1.0,
            top_singular_values=(),
        )

    probs = s2 / total_energy
    nonzero = probs[probs > 1e-18]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    eff_rank = float(math.exp(entropy))

    cum = np.cumsum(probs)
    top1 = float(cum[0])
    top8 = float(cum[min(7, cum.size - 1)])
    top16 = float(cum[min(15, cum.size - 1)])
    s16 = float(svals[min(15, svals.size - 1)])
    cond_1_16 = float(svals[0] / max(s16, 1e-12))

    return SingularSpectrum(
        shape=(int(arr.shape[0]), int(arr.shape[1])),
        frobenius_norm=frob,
        effective_rank=eff_rank,
        r50=_rank_for_energy(cum, 0.50),
        r80=_rank_for_energy(cum, 0.80),
        r90=_rank_for_energy(cum, 0.90),
        r95=_rank_for_energy(cum, 0.95),
        r99=_rank_for_energy(cum, 0.99),
        top1_energy=top1,
        top8_energy=top8,
        top16_energy=top16,
        condition_ratio_1_16=cond_1_16,
        top_singular_values=tuple(float(x) for x in svals[:store_top_k]),
    )


def decompose_state_error(
    true_state_dv_dk: np.ndarray,
    shallow_state_dv_dk: np.ndarray,
    mean_v: np.ndarray,
) -> StateErrorDecomposition:
    """Purely decompose (S_true - S_shallow) into orthogonal sink error E_b and dynamic content error E_ax."""
    true_dec = decompose_state_sink(true_state_dv_dk, mean_v)
    sh_dec = decompose_state_sink(shallow_state_dv_dk, mean_v)

    b_true = np.outer(true_dec.unit_v_bar, true_dec.sink_coeff)
    b_sh = np.outer(sh_dec.unit_v_bar, sh_dec.sink_coeff)
    e_b = b_true - b_sh
    e_ax = true_dec.content - sh_dec.content
    e_tot = np.asarray(true_state_dv_dk, dtype=np.float64) - np.asarray(shallow_state_dv_dk, dtype=np.float64)

    tot_sq = float(np.sum(e_tot * e_tot))
    b_sq = float(np.sum(e_b * e_b))
    ax_sq = float(np.sum(e_ax * e_ax))

    b_true_norm = float(np.linalg.norm(b_true))
    ax_true_norm = float(np.linalg.norm(true_dec.content))

    return StateErrorDecomposition(
        total_err_norm_sq=tot_sq,
        sink_err_norm_sq=b_sq,
        content_err_norm_sq=ax_sq,
        sink_err_share=b_sq / tot_sq if tot_sq > 1e-18 else 0.0,
        content_err_share=ax_sq / tot_sq if tot_sq > 1e-18 else 0.0,
        rel_sink_err=math.sqrt(b_sq) / max(b_true_norm, 1e-12),
        rel_content_err=math.sqrt(ax_sq) / max(ax_true_norm, 1e-12),
        content_err_spectrum=compute_singular_spectrum(e_ax, store_top_k=24),
    )


def low_rank_replace_content_error(
    true_state_dv_dk: np.ndarray,
    shallow_state_dv_dk: np.ndarray,
    mean_v: np.ndarray,
    rank: int,
    mean_sink_coeff: np.ndarray | None = None,
) -> np.ndarray:
    """Pure oracle helper for Direction 2 ceiling: anchor sink `b` and add Rank-`r` SVD correction to `ax`.

    If `mean_sink_coeff` is provided, sets `b = outer(v_hat, mean_sink_coeff)` (freezing `b` to training mean);
    otherwise keeps the shallow state's `b`. Then adds the best Rank-`r` approximation of `E_ax = ax_true - ax_shallow`.
    """
    true_dec = decompose_state_sink(true_state_dv_dk, mean_v)
    sh_dec = decompose_state_sink(shallow_state_dv_dk, mean_v)

    sink_coeff = np.asarray(mean_sink_coeff, dtype=np.float64) if mean_sink_coeff is not None else sh_dec.sink_coeff
    anchored_b = np.outer(sh_dec.unit_v_bar, sink_coeff)

    if rank <= 0:
        return (anchored_b + sh_dec.content).astype(np.float32)

    e_ax = true_dec.content - sh_dec.content
    u, s, vt = np.linalg.svd(e_ax, full_matrices=False)
    r = min(int(rank), s.shape[0])
    e_ax_r = (u[:, :r] * s[:r]) @ vt[:r, :]
    return (anchored_b + sh_dec.content + e_ax_r).astype(np.float32)


def analyze_head_weight_sink_alignment(
    head_idx: int,
    w_v_head: np.ndarray,
    w_out_head: np.ndarray,
    mean_v_head: np.ndarray,
) -> WeightSinkAlignment:
    """Purely quantify how a GDN head's Rank-1 state sink aligns with pretrained W_V and W_out weights.

    Args:
      w_v_head: `(Dv, D_model)` value projection slice for one head (e.g. `128 x 2560`).
      w_out_head: `(D_model, Dv)` output readout slice for one head (e.g. `2560 x 128`).
      mean_v_head: `(Dv,)` empirical mean value vector defining the Rank-1 sink direction `v_hat`.
    """
    wv = np.asarray(w_v_head, dtype=np.float64)
    wo = np.asarray(w_out_head, dtype=np.float64)
    v = np.asarray(mean_v_head, dtype=np.float64).reshape(-1)
    v_norm = float(np.linalg.norm(v))
    v_hat = v / v_norm if v_norm > 1e-12 else np.zeros_like(v)

    u_wv, s_wv, _ = np.linalg.svd(wv, full_matrices=False)
    wv_spec = compute_singular_spectrum(wv, store_top_k=16)

    # Orthogonal complement of sink direction in value space: W_V^perp = (I - v_hat v_hat^T) W_V
    sink_row = v_hat @ wv  # (D_model,)
    wv_perp = wv - np.outer(v_hat, sink_row)
    wv_perp_spec = compute_singular_spectrum(wv_perp, store_top_k=16)

    wv_energy = float(np.sum(wv * wv))
    sink_wv_energy = float(np.sum(sink_row * sink_row))
    sink_wv_frac = sink_wv_energy / wv_energy if wv_energy > 1e-18 else 0.0

    top1_cos = abs(float(np.dot(v_hat, u_wv[:, 0]))) if u_wv.shape[1] > 0 else 0.0
    k4 = min(4, u_wv.shape[1])
    top4_proj = u_wv[:, :k4].T @ v_hat
    top4_energy = float(np.sum(top4_proj * top4_proj))

    # How strongly does out_proj amplify/pass the Rank-1 sink direction vs its top singular value?
    s_wo = np.linalg.svd(wo, full_matrices=False, compute_uv=False)
    wo_top1 = float(s_wo[0]) if s_wo.size > 0 else 1.0
    wo_sink_norm = float(np.linalg.norm(wo @ v_hat))
    gain_ratio = wo_sink_norm / max(wo_top1, 1e-12)

    return WeightSinkAlignment(
        head_idx=int(head_idx),
        wv_spectrum=wv_spec,
        wv_perp_spectrum=wv_perp_spec,
        sink_top1_u_cos=top1_cos,
        sink_top4_subspace_energy=top4_energy,
        sink_wv_energy_fraction=sink_wv_frac,
        wout_sink_gain_ratio=gain_ratio,
    )
