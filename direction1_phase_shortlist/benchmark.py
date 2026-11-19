"""End-to-end M5 verification & ablation suite for Direction 1 (Phase-Aware Routing + Shortlist Vocab + logZ)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx_lm
import numpy as np

from direction1_phase_shortlist import ensure_metal_gpu_arch
from direction1_phase_shortlist.build_artifacts import (
    ARTIFACTS_DIR,
    DEFAULT_SHORTLIST_SIZES,
    _chunked_full_logz,
    _forward_to_layer_normed,
)
from direction1_phase_shortlist.engine import PhaseShortlistSpeculativeEngine, PhaseSpecResult
from direction1_phase_shortlist.logz_calibrator import (
    LogZCalibrator,
    calibrated_max_prob_numpy,
    evaluate_logz_calibrator,
    load_logz_calibrator,
    predict_logz_numpy,
)
from direction1_phase_shortlist.phase_router import (
    RoutingConfig,
    extract_phase_tag_ids,
    lookup_only_routing,
    phase_adaptive_routing,
    phase_gated_routing,
    uniform_hybrid_routing,
)
from direction1_phase_shortlist.quantized_head import project_shortlist_logits, slice_quantized_embedding
from direction1_phase_shortlist.shortlist_vocab import (
    StaticShortlist,
    build_dynamic_shortlist,
    evaluate_shortlist_coverage,
    load_static_shortlist,
)
from seed_agent_lab.benchmark_closed_loop import extract_tool_prompt_ids
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    _prepare_trajectory,
)
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR


@dataclass(frozen=True)
class BenchmarkMode:
    """Immutable configuration for one closed-loop ablation mode."""

    name: str
    routing: RoutingConfig
    shortlist_size: int | None  # None => full 248K vocabulary
    use_logz_calibration: bool = True
    logz_deadband_margin: float = 0.0


@dataclass(frozen=True)
class OfflineShortlistDiagnostic:
    """Immutable held-out coverage & OOV false-confidence diagnostic for one shortlist size K."""

    static_k: int
    mean_dynamic_size: float
    total_eval_tokens: int
    static_coverage: float
    dynamic_coverage: float
    oov_rate: float
    full_argmax_in_shortlist_rate: float
    oov_draft_count: int
    naive_oov_false_high_conf_rate: float
    strict_oov_false_high_conf_rate: float
    guard_oov_false_high_conf_rate: float
    strict_in_vocab_high_conf_recall: float
    guard_in_vocab_high_conf_recall: float
    head_latency_ms: float
    full_head_latency_ms: float


@dataclass(frozen=True)
class ClosedLoopPromptRecord:
    """Immutable measured wall-clock result for one held-out prompt and mode."""

    row_idx: int
    prompt_len: int
    mode_name: str
    gen_len: int
    exact_match: bool
    elapsed_seconds: float
    sub_head_build_seconds: float
    tokens_per_sec: float
    verify_calls: int
    mean_accepted_tokens: float
    total_neural_steps: int
    total_lookup_steps: int
    rollback_count: int
    draft_seconds: float
    verify_seconds: float
    rollback_seconds: float
    effective_shortlist_size: int
    thinking_tokens: int
    response_tokens: int
    tool_call_tokens: int
    thinking_neural_steps: int
    non_thinking_neural_steps: int


def build_default_modes() -> tuple[BenchmarkMode, ...]:
    """Pure factory returning the full Direction 1 ablation matrix."""
    return (
        BenchmarkMode(
            name="LookupOnly",
            routing=lookup_only_routing(),
            shortlist_size=None,
            use_logz_calibration=False,
        ),
        BenchmarkMode(
            name="Hybrid_FullVocab",
            routing=uniform_hybrid_routing(confidence_threshold=0.70, max_neural_len=6),
            shortlist_size=None,
            use_logz_calibration=False,
        ),
        BenchmarkMode(
            name="PhaseGate_FullVocab",
            routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
            shortlist_size=None,
            use_logz_calibration=False,
        ),
        BenchmarkMode(
            name="Hybrid_SL8K_Guard",
            routing=uniform_hybrid_routing(confidence_threshold=0.70, max_neural_len=6),
            shortlist_size=8192,
            use_logz_calibration=True,
            logz_deadband_margin=2.0,
        ),
        BenchmarkMode(
            name="PhaseGate_SL8K_NoCal",
            routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
            shortlist_size=8192,
            use_logz_calibration=False,
        ),
        BenchmarkMode(
            name="PhaseGate_SL8K_Guard",
            routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
            shortlist_size=8192,
            use_logz_calibration=True,
            logz_deadband_margin=2.0,
        ),
        BenchmarkMode(
            name="PhaseGate_SL8K_Strict",
            routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
            shortlist_size=8192,
            use_logz_calibration=True,
            logz_deadband_margin=0.0,
        ),
        BenchmarkMode(
            name="PhaseGate_SL16K_Guard",
            routing=phase_gated_routing(confidence_threshold=0.70, max_neural_len=6),
            shortlist_size=16384,
            use_logz_calibration=True,
            logz_deadband_margin=2.0,
        ),
        BenchmarkMode(
            name="PhaseAdapt_SL8K_Guard",
            routing=phase_adaptive_routing(
                think_confidence_threshold=0.85,
                think_max_neural_len=3,
                active_confidence_threshold=0.70,
                active_max_neural_len=6,
            ),
            shortlist_size=8192,
            use_logz_calibration=True,
            logz_deadband_margin=2.0,
        ),
    )


def _bench_head_ms(fn, warmup: int = 8, iters: int = 40) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000.0


def run_offline_diagnostics(
    model,
    tokenizer,
    held_out_rows: Sequence[dict],
    w8_weight_bf16: mx.array,
    w8_bias_bf16: mx.array,
    shortlists: dict[int, StaticShortlist],
    calibrator: LogZCalibrator,
    draft_layer: int = 8,
    conf_threshold: float = 0.70,
) -> tuple[dict[str, float], tuple[OfflineShortlistDiagnostic, ...]]:
    """Measure held-out coverage, logZ regression accuracy, OOV false-confidence suppression, and head latency."""
    embed_tokens = model.language_model.model.embed_tokens

    # Microbenchmark full vs shortlist LM head
    probe_x = mx.random.normal((1, 1, int(w8_weight_bf16.shape[1]))).astype(mx.bfloat16)
    mx.eval(probe_x)
    full_head_ms = _bench_head_ms(lambda: embed_tokens.as_linear(probe_x))

    # Collect held-out assistant frozen normed_h8 + rotated hidden states + ground-truth full logits stats
    trace_records: list[tuple[list[int], list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    all_h8_list: list[np.ndarray] = []
    all_logz_list: list[np.ndarray] = []

    for row in held_out_rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=3584)
        if item is None:
            continue
        token_ids, _tags, asst_mask = item
        asst_pos = [i for i, flag in enumerate(asst_mask[1:]) if flag]
        if not asst_pos:
            continue
        first_asst = asst_pos[0]
        prompt_prefix = token_ids[: first_asst + 1]
        target_asst_tokens = [int(token_ids[pos + 1]) for pos in asst_pos]

        normed_h8 = _forward_to_layer_normed(model, token_ids[:-1], stop_layer=draft_layer)
        h8_asst = normed_h8[mx.array(asst_pos, dtype=mx.int32)]
        rotated_bf16 = (h8_asst @ w8_weight_bf16) + w8_bias_bf16
        mx.eval(h8_asst, rotated_bf16)

        # Chunked exact full-vocab argmax, max_prob, and logZ
        full_argmax_chunks: list[np.ndarray] = []
        full_maxprob_chunks: list[np.ndarray] = []
        full_logz_chunks: list[np.ndarray] = []
        for start in range(0, int(rotated_bf16.shape[0]), 128):
            sub = rotated_bf16[start : start + 128]
            logits_f32 = embed_tokens.as_linear(sub).astype(mx.float32)
            am = mx.argmax(logits_f32, axis=-1)
            ml = mx.max(logits_f32, axis=-1)
            lz = mx.logsumexp(logits_f32, axis=-1)
            mp = mx.exp(ml - lz)
            mx.eval(am, mp, lz)
            full_argmax_chunks.append(np.asarray(am, dtype=np.int32))
            full_maxprob_chunks.append(np.asarray(mp, dtype=np.float64))
            full_logz_chunks.append(np.asarray(lz, dtype=np.float64))

        h8_np = np.asarray(h8_asst.astype(mx.float32), dtype=np.float32)
        rot_np = np.asarray(rotated_bf16.astype(mx.float32), dtype=np.float32)
        full_am = np.concatenate(full_argmax_chunks)
        full_mp = np.concatenate(full_maxprob_chunks)
        full_lz = np.concatenate(full_logz_chunks)

        trace_records.append((prompt_prefix, target_asst_tokens, h8_np, rot_np, full_am, full_mp, full_lz))
        all_h8_list.append(h8_np)
        all_logz_list.append(full_lz)
        del normed_h8, h8_asst, rotated_bf16
        mx.clear_cache()

    all_h8 = np.concatenate(all_h8_list, axis=0)
    all_lz = np.concatenate(all_logz_list, axis=0)
    logz_eval = evaluate_logz_calibrator(calibrator, all_h8, all_lz)
    logz_summary = {
        "n_samples": float(logz_eval.n_samples),
        "mae": logz_eval.mae,
        "rmse": logz_eval.rmse,
        "r2": logz_eval.r2,
        "mean_bias": logz_eval.mean_bias,
    }

    diagnostics: list[OfflineShortlistDiagnostic] = []
    for k in sorted(shortlists.keys()):
        sl = shortlists[k]
        sub_head = slice_quantized_embedding(embed_tokens, sl.token_ids)
        head_ms = _bench_head_ms(lambda: project_shortlist_logits(probe_x, sub_head))

        total_toks = 0
        cov_static = 0
        cov_dynamic = 0
        dyn_sizes: list[int] = []
        full_am_in_sl = 0
        oov_draft_n = 0
        naive_false_high = 0
        strict_false_high = 0
        guard_false_high = 0
        true_in_vocab_high_n = 0
        strict_in_vocab_high_hits = 0
        guard_in_vocab_high_hits = 0

        for prompt_prefix, target_asst_tokens, h8_np, rot_np, full_am, full_mp, _full_lz in trace_records:
            cov = evaluate_shortlist_coverage(sl, prompt_prefix, target_asst_tokens)
            total_toks += cov.total_tokens
            cov_static += cov.covered_by_static
            cov_dynamic += cov.covered_by_dynamic
            dyn = build_dynamic_shortlist(sl, prompt_prefix)
            dyn_sizes.append(dyn.total_size)

            dyn_set = frozenset(dyn.all_ids)
            dyn_head = slice_quantized_embedding(embed_tokens, dyn.all_ids)
            rot_bf16 = mx.array(rot_np, dtype=mx.bfloat16)
            sl_logits = np.asarray(project_shortlist_logits(rot_bf16, dyn_head).astype(mx.float32), dtype=np.float64)
            mx.clear_cache()

            pred_lz = predict_logz_numpy(calibrator, h8_np)
            _, naive_prob = calibrated_max_prob_numpy(sl_logits, np.full_like(pred_lz, -1e9))
            _, strict_prob = calibrated_max_prob_numpy(sl_logits, pred_lz, deadband_margin=0.0)
            _, guard_prob = calibrated_max_prob_numpy(sl_logits, pred_lz, deadband_margin=2.0)

            for i in range(len(full_am)):
                in_sl = int(full_am[i]) in dyn_set
                if in_sl:
                    full_am_in_sl += 1
                    if full_mp[i] >= conf_threshold:
                        true_in_vocab_high_n += 1
                        if strict_prob[i] >= conf_threshold:
                            strict_in_vocab_high_hits += 1
                        if guard_prob[i] >= conf_threshold:
                            guard_in_vocab_high_hits += 1
                else:
                    oov_draft_n += 1
                    if naive_prob[i] >= conf_threshold:
                        naive_false_high += 1
                    if strict_prob[i] >= conf_threshold:
                        strict_false_high += 1
                    if guard_prob[i] >= conf_threshold:
                        guard_false_high += 1

        diagnostics.append(
            OfflineShortlistDiagnostic(
                static_k=k,
                mean_dynamic_size=float(np.mean(dyn_sizes)) if dyn_sizes else float(k),
                total_eval_tokens=total_toks,
                static_coverage=cov_static / total_toks if total_toks > 0 else 0.0,
                dynamic_coverage=cov_dynamic / total_toks if total_toks > 0 else 0.0,
                oov_rate=(total_toks - cov_dynamic) / total_toks if total_toks > 0 else 0.0,
                full_argmax_in_shortlist_rate=full_am_in_sl / total_toks if total_toks > 0 else 0.0,
                oov_draft_count=oov_draft_n,
                naive_oov_false_high_conf_rate=naive_false_high / oov_draft_n if oov_draft_n > 0 else 0.0,
                strict_oov_false_high_conf_rate=strict_false_high / oov_draft_n if oov_draft_n > 0 else 0.0,
                guard_oov_false_high_conf_rate=guard_false_high / oov_draft_n if oov_draft_n > 0 else 0.0,
                strict_in_vocab_high_conf_recall=strict_in_vocab_high_hits / true_in_vocab_high_n if true_in_vocab_high_n > 0 else 0.0,
                guard_in_vocab_high_conf_recall=guard_in_vocab_high_hits / true_in_vocab_high_n if true_in_vocab_high_n > 0 else 0.0,
                head_latency_ms=head_ms,
                full_head_latency_ms=full_head_ms,
            )
        )

    return logz_summary, tuple(diagnostics)


def aggregate_mode_records(
    records: Sequence[ClosedLoopPromptRecord],
    mode_order: Sequence[str],
) -> list[dict[str, float | int | str]]:
    """Pure aggregation over prompt records producing structured summary rows."""
    by_mode: dict[str, list[ClosedLoopPromptRecord]] = {name: [] for name in mode_order}
    for rec in records:
        if rec.mode_name in by_mode:
            by_mode[rec.mode_name].append(rec)

    base_rows = by_mode.get("Baseline_32L", [])
    base_time = sum(r.elapsed_seconds for r in base_rows)
    base_tokens = sum(r.gen_len for r in base_rows)
    base_tps = base_tokens / base_time if base_time > 0 else 1.0

    summary: list[dict[str, float | int | str]] = []
    for name in mode_order:
        rows = by_mode[name]
        if not rows:
            continue
        tot_tok = sum(r.gen_len for r in rows)
        tot_t = sum(r.elapsed_seconds for r in rows)
        tot_sub = sum(r.sub_head_build_seconds for r in rows)
        tot_vcalls = sum(r.verify_calls for r in rows)
        tot_draft_t = sum(r.draft_seconds for r in rows)
        tot_verify_t = sum(r.verify_seconds for r in rows)
        tot_roll_t = sum(r.rollback_seconds for r in rows)
        tps = tot_tok / tot_t if tot_t > 0 else 0.0
        tps_inc_sub = tot_tok / (tot_t + tot_sub) if (tot_t + tot_sub) > 0 else 0.0
        exact_n = sum(1 for r in rows if r.exact_match)
        summary.append(
            {
                "mode": name,
                "exact_matches": exact_n,
                "total_prompts": len(rows),
                "total_tokens": tot_tok,
                "elapsed_s": tot_t,
                "sub_head_ms": tot_sub * 1000.0,
                "tokens_per_sec": tps,
                "tokens_per_sec_with_subhead": tps_inc_sub,
                "speedup": tps / base_tps if base_tps > 0 else 0.0,
                "tau": tot_tok / tot_vcalls if tot_vcalls > 0 else 1.0,
                "draft_pct": (tot_draft_t / tot_t * 100.0) if tot_t > 0 else 0.0,
                "verify_pct": (tot_verify_t / tot_t * 100.0) if tot_t > 0 else 0.0,
                "rollback_pct": (tot_roll_t / tot_t * 100.0) if tot_t > 0 else 0.0,
                "neural_steps": sum(r.total_neural_steps for r in rows),
                "thinking_neural_steps": sum(r.thinking_neural_steps for r in rows),
                "non_thinking_neural_steps": sum(r.non_thinking_neural_steps for r in rows),
                "lookup_steps": sum(r.total_lookup_steps for r in rows),
                "rollbacks": sum(r.rollback_count for r in rows),
                "mean_shortlist_size": float(np.mean([r.effective_shortlist_size for r in rows])),
            }
        )
    return summary


def plot_direction1_summary(
    summary_rows: Sequence[dict[str, float | int | str]],
    records: Sequence[ClosedLoopPromptRecord],
    diagnostics: Sequence[OfflineShortlistDiagnostic],
    out_path: Path,
) -> Path:
    """Render a clean 4-panel comparison figure saved inside direction1_phase_shortlist/artifacts/."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=160)
    fig.suptitle(
        "Direction 1: Phase-Aware Routing + Shortlist Vocab + logZ Calibration (Qwen3.5-4B-MLX-8bit on M5)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: End-to-end tokens/sec & speedup across all modes
    ax = axes[0, 0]
    modes = [str(r["mode"]) for r in summary_rows]
    tps_vals = [float(r["tokens_per_sec"]) for r in summary_rows]
    sp_vals = [float(r["speedup"]) for r in summary_rows]
    colors = ["#64748b" if m == "Baseline_32L" else "#2563eb" if "_SL" in m else "#0ea5e9" if "PhaseGate" in m else "#94a3b8" for m in modes]
    y_pos = np.arange(len(modes))
    bars = ax.barh(y_pos, tps_vals, color=colors, height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modes, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_title("A. Closed-Loop M5 Throughput (rows[12:18])")
    ax.set_xlim(0, max(tps_vals) * 1.28)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, r, tps, sp in zip(bars, summary_rows, tps_vals, sp_vals):
        eq_tag = "" if int(r["exact_matches"]) == int(r["total_prompts"]) else f" [{r['exact_matches']}/{r['total_prompts']}]"
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, f"{tps:.1f} ({sp:.2f}x){eq_tag}", va="center", fontsize=8.2)

    # Panel B: Stacked wall-clock breakdown (Draft / Verify / Rollback)
    ax = axes[0, 1]
    draft_s = [float(r["elapsed_s"]) * float(r["draft_pct"]) / 100.0 for r in summary_rows]
    verify_s = [float(r["elapsed_s"]) * float(r["verify_pct"]) / 100.0 for r in summary_rows]
    roll_s = [float(r["elapsed_s"]) * float(r["rollback_pct"]) / 100.0 for r in summary_rows]
    ax.barh(y_pos, draft_s, label="Draft", color="#f59e0b", height=0.6)
    ax.barh(y_pos, verify_s, left=draft_s, label="Verify", color="#10b981", height=0.6)
    left_roll = [d + v for d, v in zip(draft_s, verify_s)]
    ax.barh(y_pos, roll_s, left=left_roll, label="Rollback", color="#ef4444", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modes, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Total Wall-Clock Time Across 6 Held-Out Prompts (s)")
    ax.set_title("B. Wall-Clock Time Decomposition (Draft vs Verify vs Rollback)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Panel C: Per-prompt speedup comparing Baseline, Hybrid_FullVocab, PhaseGate_FullVocab, best Shortlist mode
    ax = axes[1, 0]
    best_sl_row = max(
        (r for r in summary_rows if "_SL" in str(r["mode"])),
        key=lambda r: float(r["tokens_per_sec"]),
        default=summary_rows[-1],
    )
    best_sl_name = str(best_sl_row["mode"])
    compare_modes = ["LookupOnly", "Hybrid_FullVocab", "PhaseGate_FullVocab", best_sl_name]
    row_ids = sorted({r.row_idx for r in records})
    base_map = {r.row_idx: r.tokens_per_sec for r in records if r.mode_name == "Baseline_32L"}
    x = np.arange(len(row_ids))
    width = 0.2
    palette = ["#94a3b8", "#f59e0b", "#0ea5e9", "#1d4ed8"]
    for idx_m, m_name in enumerate(compare_modes):
        rec_map = {r.row_idx: r.tokens_per_sec for r in records if r.mode_name == m_name}
        vals = [rec_map.get(rid, 0.0) / base_map[rid] for rid in row_ids]
        ax.bar(x + (idx_m - 1.5) * width, vals, width=width, label=m_name, color=palette[idx_m])
    ax.axhline(1.0, color="#334155", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Row {rid}" for rid in row_ids], fontsize=9)
    ax.set_ylabel("Speedup vs Baseline_32L")
    ax.set_ylim(0, 2.75)
    ax.set_title("C. Per-Prompt Speedup (Fixing Long-<think> Row 16)")
    ax.legend(fontsize=7.8, loc="upper right", ncol=2, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel D: Offline Shortlist Coverage & OOV False-Confidence Suppression (Naive vs Guard vs Strict)
    ax = axes[1, 1]
    ks = [f"{d.static_k // 1024}K" for d in diagnostics]
    xk = np.arange(len(ks))
    naive_fc = [d.naive_oov_false_high_conf_rate * 100.0 for d in diagnostics]
    guard_fc = [d.guard_oov_false_high_conf_rate * 100.0 for d in diagnostics]
    guard_rec = [d.guard_in_vocab_high_conf_recall * 100.0 for d in diagnostics]
    dyn_cov = [d.dynamic_coverage * 100.0 for d in diagnostics]
    w2 = 0.20
    ax.bar(xk - 1.5 * w2, naive_fc, width=w2, label="Naive OOV False-Conf (%)", color="#ef4444")
    ax.bar(xk - 0.5 * w2, guard_fc, width=w2, label="Guard(d=2) OOV False-Conf (%)", color="#10b981")
    ax.bar(xk + 0.5 * w2, guard_rec, width=w2, label="Guard(d=2) In-Vocab Recall (%)", color="#6366f1")
    ax.bar(xk + 1.5 * w2, dyn_cov, width=w2, label="Dynamic Coverage (%)", color="#3b82f6")
    ax.set_xticks(xk)
    ax.set_xticklabels(ks, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 128)
    ax.set_title("D. Shortlist Coverage & normed_h8 logZ Calibration")
    ax.legend(fontsize=7.6, loc="upper center", ncol=2, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_benchmark(
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
    w8_checkpoint: Path = CHECKPOINT_DIR / "layer8_ridge_asst.npz",
    artifacts_dir: Path = ARTIFACTS_DIR,
    max_new_tokens: int = 160,
    max_prompt_tokens: int = 2560,
    enable_thinking: bool = True,
) -> dict:
    """Run full Direction 1 offline diagnostics + closed-loop M5 benchmark on held-out rows[12:18]."""
    ensure_metal_gpu_arch()
    all_rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8")]
    held_out_rows = all_rows[12:18]

    w8_npz = np.load(w8_checkpoint)
    w8_weight_np = np.asarray(w8_npz["weight"], dtype=np.float32)
    w8_bias_np = np.asarray(w8_npz["bias"], dtype=np.float32)

    shortlists: dict[int, StaticShortlist] = {
        k: load_static_shortlist(artifacts_dir / f"shortlist_{k}.npz")
        for k in DEFAULT_SHORTLIST_SIZES
    }
    calibrator = load_logz_calibrator(artifacts_dir / "layer8_logz_calibrator.npz")

    model, tokenizer = mlx_lm.load(str(model_dir))
    phase_tags = extract_phase_tag_ids(tokenizer)
    eos_token_ids = frozenset(int(t) for t in getattr(tokenizer, "eos_token_ids", set()))

    w8_weight_bf16 = mx.array(w8_weight_np, dtype=mx.bfloat16)
    w8_bias_bf16 = mx.array(w8_bias_np, dtype=mx.bfloat16)
    mx.eval(w8_weight_bf16, w8_bias_bf16)

    print("=== PART A: Held-Out Offline Diagnostics (rows[12:18]) ===")
    logz_summary, diagnostics = run_offline_diagnostics(
        model=model,
        tokenizer=tokenizer,
        held_out_rows=held_out_rows,
        w8_weight_bf16=w8_weight_bf16,
        w8_bias_bf16=w8_bias_bf16,
        shortlists=shortlists,
        calibrator=calibrator,
    )
    print(
        f"LogZCalibrator held-out (n={int(logz_summary['n_samples'])}): "
        f"MAE={logz_summary['mae']:.4f} | RMSE={logz_summary['rmse']:.4f} | "
        f"R2={logz_summary['r2']:.4f} | bias={logz_summary['mean_bias']:+.4f}"
    )
    for d in diagnostics:
        print(
            f"  K={d.static_k:5d} (dyn={d.mean_dynamic_size:7.1f}) | "
            f"StaticCov={d.static_coverage*100:5.2f}% | DynCov={d.dynamic_coverage*100:5.2f}% | "
            f"OOV={d.oov_rate*100:4.2f}% | FullArgmaxInSL={d.full_argmax_in_shortlist_rate*100:5.2f}% | "
            f"OOVFalseConf(naive/guard/strict)={d.naive_oov_false_high_conf_rate*100:4.1f}%/{d.guard_oov_false_high_conf_rate*100:4.1f}%/{d.strict_oov_false_high_conf_rate*100:4.1f}% "
            f"(n_oov={d.oov_draft_count}) | InVocabRec(guard/strict)={d.guard_in_vocab_high_conf_recall*100:4.1f}%/{d.strict_in_vocab_high_conf_recall*100:4.1f}% | "
            f"Head={d.head_latency_ms:.3f}ms vs {d.full_head_latency_ms:.3f}ms ({d.full_head_latency_ms/d.head_latency_ms:.1f}x)"
        )

    print("\n=== PART B: Closed-Loop M5 Wall-Clock Benchmark (rows[12:18], max_new_tokens=%d) ===" % max_new_tokens)
    modes = build_default_modes()
    engines_by_k: dict[int | None, PhaseShortlistSpeculativeEngine] = {
        None: PhaseShortlistSpeculativeEngine(
            model=model,
            w8_weight=w8_weight_np,
            w8_bias=w8_bias_np,
            phase_tags=phase_tags,
            static_shortlist=None,
            logz_calibrator=None,
        ),
        8192: PhaseShortlistSpeculativeEngine(
            model=model,
            w8_weight=w8_weight_np,
            w8_bias=w8_bias_np,
            phase_tags=phase_tags,
            static_shortlist=shortlists[8192],
            logz_calibrator=calibrator,
        ),
        16384: PhaseShortlistSpeculativeEngine(
            model=model,
            w8_weight=w8_weight_np,
            w8_bias=w8_bias_np,
            phase_tags=phase_tags,
            static_shortlist=shortlists[16384],
            logz_calibrator=calibrator,
        ),
    }

    # Warmup MLX Metal kernels once before timing
    warm_prompt = extract_tool_prompt_ids(tokenizer, held_out_rows[0], max_prompt_tokens=max_prompt_tokens, enable_thinking=enable_thinking)
    assert warm_prompt is not None
    _ = engines_by_k[None].generate_baseline(warm_prompt, max_new_tokens=8, eos_token_ids=eos_token_ids)
    _ = engines_by_k[8192].generate_speculative(
        warm_prompt,
        max_new_tokens=8,
        eos_token_ids=eos_token_ids,
        routing=phase_gated_routing(),
        use_shortlist=True,
        use_logz_calibration=True,
    )

    records: list[ClosedLoopPromptRecord] = []
    for offset, raw_row in enumerate(held_out_rows):
        row_idx = 12 + offset
        prompt_ids = extract_tool_prompt_ids(
            tokenizer,
            raw_row,
            max_prompt_tokens=max_prompt_tokens,
            enable_thinking=enable_thinking,
        )
        assert prompt_ids is not None

        base_tokens, base_elapsed = engines_by_k[None].generate_baseline(
            prompt_ids, max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids
        )
        base_tps = len(base_tokens) / base_elapsed if base_elapsed > 0 else 0.0
        records.append(
            ClosedLoopPromptRecord(
                row_idx=row_idx,
                prompt_len=len(prompt_ids),
                mode_name="Baseline_32L",
                gen_len=len(base_tokens),
                exact_match=True,
                elapsed_seconds=base_elapsed,
                sub_head_build_seconds=0.0,
                tokens_per_sec=base_tps,
                verify_calls=len(base_tokens),
                mean_accepted_tokens=1.0,
                total_neural_steps=0,
                total_lookup_steps=0,
                rollback_count=0,
                draft_seconds=0.0,
                verify_seconds=base_elapsed,
                rollback_seconds=0.0,
                effective_shortlist_size=int(model.language_model.model.embed_tokens.weight.shape[0]),
                thinking_tokens=0,
                response_tokens=0,
                tool_call_tokens=0,
                thinking_neural_steps=0,
                non_thinking_neural_steps=0,
            )
        )

        row_line_parts = [f"[Row {row_idx}] plen={len(prompt_ids)}, gen={len(base_tokens)} | Base {base_tps:.2f}"]
        for mode in modes:
            eng = engines_by_k[mode.shortlist_size]
            res: PhaseSpecResult = eng.generate_speculative(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                routing=mode.routing,
                use_shortlist=(mode.shortlist_size is not None),
                use_logz_calibration=mode.use_logz_calibration,
                logz_deadband_margin=mode.logz_deadband_margin,
            )
            exact = res.tokens == base_tokens
            records.append(
                ClosedLoopPromptRecord(
                    row_idx=row_idx,
                    prompt_len=len(prompt_ids),
                    mode_name=mode.name,
                    gen_len=len(res.tokens),
                    exact_match=exact,
                    elapsed_seconds=res.elapsed_seconds,
                    sub_head_build_seconds=res.sub_head_build_seconds,
                    tokens_per_sec=res.tokens_per_sec,
                    verify_calls=res.verify_calls,
                    mean_accepted_tokens=res.mean_accepted_tokens,
                    total_neural_steps=res.total_neural_steps,
                    total_lookup_steps=res.total_lookup_steps,
                    rollback_count=res.rollback_count,
                    draft_seconds=res.draft_seconds,
                    verify_seconds=res.verify_seconds,
                    rollback_seconds=res.rollback_seconds,
                    effective_shortlist_size=res.effective_shortlist_size,
                    thinking_tokens=res.thinking_tokens,
                    response_tokens=res.response_tokens,
                    tool_call_tokens=res.tool_call_tokens,
                    thinking_neural_steps=res.thinking_neural_steps,
                    non_thinking_neural_steps=res.non_thinking_neural_steps,
                )
            )
            sp = res.tokens_per_sec / base_tps if base_tps > 0 else 0.0
            row_line_parts.append(
                f"{mode.name} {res.tokens_per_sec:.2f}({sp:.2f}x,eq={exact},th={res.thinking_tokens}/tc={res.tool_call_tokens})"
            )
        print(" | ".join(row_line_parts))

    mode_order = ["Baseline_32L"] + [m.name for m in modes]
    summary_rows = aggregate_mode_records(records, mode_order)

    print("\n=== AGGREGATE SUMMARY (6 Held-Out SWE Prompts, rows[12:18]) ===")
    header = (
        f"{'Mode':<28} | {'Exact':<5} | {'Time(s)':>7} | {'tok/s':>6} | {'Speedup':>7} | "
        f"{'Tau':>4} | {'Draft%':>6} | {'Ver%':>5} | {'Roll%':>5} | {'N_thk/oth':>11} | {'SubHead(ms)':>11}"
    )
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        print(
            f"{s['mode']:<28} | {s['exact_matches']}/{s['total_prompts']:<3} | {s['elapsed_s']:7.2f} | "
            f"{s['tokens_per_sec']:6.2f} | {s['speedup']:6.2f}x | {s['tau']:4.2f} | "
            f"{s['draft_pct']:5.1f}% | {s['verify_pct']:4.1f}% | {s['rollback_pct']:4.1f}% | "
            f"{int(s['thinking_neural_steps']):>5}/{int(s['non_thinking_neural_steps']):<5} | {s['sub_head_ms']:11.2f}"
        )

    report = {
        "logz_held_out": logz_summary,
        "offline_shortlist_diagnostics": [asdict(d) for d in diagnostics],
        "summary": summary_rows,
        "per_prompt_records": [asdict(r) for r in records],
    }
    json_path = artifacts_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_path = plot_direction1_summary(
        summary_rows=summary_rows,
        records=records,
        diagnostics=diagnostics,
        out_path=artifacts_dir / "direction1_comparison.png",
    )
    print(f"\nSaved JSON report -> {json_path}")
    print(f"Saved comparison figure -> {plot_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Direction 1 offline diagnostics and M5 closed-loop benchmark.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-prompt-tokens", type=int, default=2560)
    args = parser.parse_args()
    run_benchmark(max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens)


if __name__ == "__main__":
    main()
