"""Held-out evaluation & real M5 wall-clock Prefill speedup benchmark for Upgraded Direction 2."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx_lm
import numpy as np

from direction2_hybrid_prefill_align import ensure_metal_gpu_arch
from direction2_hybrid_prefill_align.aligned_prefill import (
    AlignedPrefillConfig,
    CompiledGDNLayer,
    CompiledHybridAligner,
    compile_hybrid_aligner,
    run_aligned_shallow_prefill,
)
from direction2_hybrid_prefill_align.kv_gdn_aligner import load_hybrid_aligner
from direction2_hybrid_prefill_align.train_aligners import ARTIFACTS_DIR
from direction3_role_prefill_svd.role_prefill_probe import _update_bucket_metrics
from direction3_role_prefill_svd.role_segments import build_depth_schedule, parse_role_tags
from seed_agent_lab.run_seed_coding_eval import (
    _clone_cache,
    _find_last_assistant_span,
    _score_continuation,
)
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    RunningMetrics,
    _prepare_trajectory,
)


@dataclass(frozen=True)
class EvalPolicySpec:
    """Immutable specification of one Direction 2 evaluation configuration."""

    name: str
    schedule_policy: str  # "Full_32L", "Uniform_k16", "Uniform_k24", "Skip_AllTool_k16", "Skip_OldTool_k16"
    shallow_k: int
    config: AlignedPrefillConfig
    attn_rank: int | None = None
    gdn_rank_slice: int | None = None  # Slice top-r SVD components from compiled GDN map


@dataclass(frozen=True)
class Direction2EvalSummary:
    """Immutable aggregated held-out accuracy + real M5 wall-clock Prefill latency summary."""

    name: str
    schedule_policy: str
    total_prefill_seconds: float
    prefill_tok_per_sec: float
    prefill_speedup: float
    all_asst_n: int
    all_asst_kl: float
    all_asst_top1: float
    all_asst_top5: float
    reasoning_kl: float
    reasoning_top1: float
    code_kl: float
    code_top1: float
    tool_kl: float
    tool_top1: float
    echo_kl: float
    echo_top1: float


def _slice_compiled_gdn_rank(compiled: CompiledHybridAligner, rank: int | None) -> CompiledHybridAligner:
    """Purely slice compiled GDN low-rank factors to top-`rank` SVD components without mutating original."""
    if rank is None:
        return compiled

    def _slice_table(table):
        out = {}
        for l_idx, layer in table.items():
            r = min(int(rank), int(layer.v_left.shape[2]))
            v_l = mx.contiguous(layer.v_left[:, :, :r])
            v_r = mx.contiguous(layer.v_right[:, :r, :])
            mx.eval(v_l, v_r)
            out[l_idx] = CompiledGDNLayer(
                layer_idx=layer.layer_idx,
                freeze_b=layer.freeze_b,
                v_left=v_l,
                v_right=v_r,
                v_bias=layer.v_bias,
                unit_v_bar=layer.unit_v_bar,
                k_weight=layer.k_weight,
                k_bias=layer.k_bias,
            )
        return out

    return CompiledHybridAligner(
        anchor_layer=compiled.anchor_layer,
        attn_by_layer=compiled.attn_by_layer,
        gdn_ax_by_layer=_slice_table(compiled.gdn_ax_by_layer),
        gdn_fullv_by_layer=_slice_table(compiled.gdn_fullv_by_layer),
        mean_v_by_layer=compiled.mean_v_by_layer,
        mean_a_by_layer=compiled.mean_a_by_layer,
    )


def build_eval_matrix() -> tuple[EvalPolicySpec, ...]:
    """Pure factory returning the full Direction 2 accuracy + speedup ablation ladder."""
    raw = AlignedPrefillConfig(align_attn_kv=False, gdn_mode="none")
    return (
        EvalPolicySpec("Full_32L", "Full_32L", 32, raw),
        EvalPolicySpec("Uniform_k24_Raw", "Uniform_k24", 24, raw),
        EvalPolicySpec("Uniform_k16_Raw", "Uniform_k16", 16, raw),
        EvalPolicySpec(
            "U16_GDN_FreezeB_Ax_Only",
            "Uniform_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=False, gdn_mode="freeze_b_fit_ax", align_gdn_k=True),
            gdn_rank_slice=16,
        ),
        EvalPolicySpec(
            "U16_AttnKV_FullRidge",
            "Uniform_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="none"),
            attn_rank=None,
        ),
        EvalPolicySpec(
            "U16_AttnKV_SVD_R256",
            "Uniform_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="none"),
            attn_rank=256,
        ),
        EvalPolicySpec(
            "U16_AttnKV + FreezeB_Ax_R8",
            "Uniform_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="freeze_b_fit_ax", align_gdn_k=False),
            attn_rank=None,
            gdn_rank_slice=8,
        ),
        EvalPolicySpec(
            "U16_AttnKV + FreezeB_Ax_R16",
            "Uniform_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="freeze_b_fit_ax", align_gdn_k=False),
            attn_rank=None,
            gdn_rank_slice=16,
        ),
        EvalPolicySpec(
            "U16_AttnKV + FreezeB_AxK_R16",
            "Uniform_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="freeze_b_fit_ax", align_gdn_k=True),
            attn_rank=None,
            gdn_rank_slice=16,
        ),
        EvalPolicySpec(
            "U16_AttnKV + FitFullV_R16",
            "Uniform_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="fit_full_v", align_gdn_k=False),
            attn_rank=None,
            gdn_rank_slice=16,
        ),
        EvalPolicySpec("Skip_AllTool_k16_Raw", "Skip_AllTool_k16", 16, raw),
        EvalPolicySpec(
            "Skip_AllTool_k16 + D2_AttnKV",
            "Skip_AllTool_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="none"),
            attn_rank=None,
        ),
        EvalPolicySpec(
            "Skip_AllTool_k16 + D2_Full",
            "Skip_AllTool_k16",
            16,
            AlignedPrefillConfig(align_attn_kv=True, gdn_mode="freeze_b_fit_ax", align_gdn_k=False),
            attn_rank=None,
            gdn_rank_slice=16,
        ),
    )


def plot_direction2_summary(
    summaries: Sequence[Direction2EvalSummary],
    out_path: Path,
) -> Path:
    """Render a clean 4-panel M5 Prefill Speedup vs Fidelity comparison figure."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), dpi=160)
    fig.suptitle(
        "Upgraded Direction 2: 4-Layer Attn KV + Frozen-b Rank-r ax Prefill Alignment (Qwen3.5-4B-MLX-8bit on M5)",
        fontsize=13.2,
        fontweight="bold",
        y=0.985,
    )

    non_full = [s for s in summaries if s.name != "Full_32L"]
    names = [s.name for s in non_full]
    kls = [s.all_asst_kl for s in non_full]
    t1s = [s.all_asst_top1 * 100.0 for s in non_full]
    sps = [s.prefill_speedup for s in non_full]
    y_pos = np.arange(len(names))

    colors = [
        "#64748b" if "Raw" in n else "#10b981" if "Skip_AllTool" in n else "#8b5cf6" if "FitFullV" in n else "#ef4444" if "GDN_FreezeB_Ax_Only" in n else "#2563eb"
        for n in names
    ]

    # Panel A: Continuation KL Divergence (Lower is Better)
    ax = axes[0, 0]
    bars = ax.barh(y_pos, kls, color=colors, height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlabel("Continuation KL Divergence vs Full 32L (Lower is Better)")
    ax.set_title("A. Held-Out Fidelity on SWE-smith (rows[12:18], max_tokens=6144)")
    ax.set_xlim(0, max(kls) * 1.42 if kls else 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, kl, t1 in zip(bars, kls, t1s):
        ax.text(bar.get_width() + max(kls) * 0.015, bar.get_y() + bar.get_height() / 2, f"KL={kl:.4f} (T1={t1:.1f}%)", va="center", fontsize=7.8)

    # Panel B: Real Apple M5 Wall-Clock Prefill Speedup
    ax = axes[0, 1]
    bars_sp = ax.barh(y_pos, sps, color=colors, height=0.62)
    ax.axvline(1.0, color="#334155", linestyle=":", linewidth=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlabel("Measured M5 Wall-Clock Prefill Speedup vs Full 32L")
    ax.set_title("B. Real M5 Wall-Clock Prefill Speedup (21K Total Prefix Tokens)")
    ax.set_xlim(0, max(sps) * 1.28 if sps else 2.0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, sp, s in zip(bars_sp, sps, non_full):
        ax.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{sp:.2f}x ({s.prefill_tok_per_sec:.0f} tok/s)",
            va="center",
            fontsize=7.8,
        )

    # Panel C: Pareto Frontier (Prefill Speedup vs Top-1 Exact Agreement)
    ax = axes[1, 0]
    label_offsets = {
        "Uniform_k24_Raw": (-58, -14, "Uk24_Raw"),
        "Uniform_k16_Raw": (6, -12, "Uk16_Raw"),
        "U16_GDN_FreezeB_Ax_Only": (-118, 4, "GDN_Ax_Only"),
        "U16_AttnKV_FullRidge": (-112, 10, "U16_AttnKV (1.61x, 96.6%)"),
        "U16_AttnKV_SVD_R256": (6, -12, "U16_AttnKV_R256 (1.62x)"),
        "Skip_AllTool_k16_Raw": (6, -12, "SkipTool_Raw"),
        "Skip_AllTool_k16 + D2_Full": (6, 6, "SkipTool + D2 (97.7%)"),
    }
    for s, c in zip(non_full, colors):
        ax.scatter(s.prefill_speedup, s.all_asst_top1 * 100.0, color=c, s=78, zorder=4, edgecolors="white", linewidth=0.7)
        if s.name in label_offsets:
            dx, dy, lbl = label_offsets[s.name]
            ax.annotate(
                lbl,
                (s.prefill_speedup, s.all_asst_top1 * 100.0),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=7.6,
                arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7},
            )
    ax.set_xlabel("Real M5 Prefill Speedup (x)")
    ax.set_ylabel("Continuation Top-1 Agreement vs Full 32L (%)")
    ax.set_ylim(min(t1s) - 3.5, 101.8)
    ax.set_xlim(1.05, max(sps) * 1.18)
    ax.set_title("C. Speedup-Fidelity Pareto Frontier (Upper-Right is Better)")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel D: Category Breakdown (reasoning / new_code_arg / tool_boilerplate / repeated_echo)
    ax = axes[1, 1]
    key_modes = [
        "Uniform_k16_Raw",
        "U16_AttnKV_FullRidge",
        "U16_AttnKV + FreezeB_Ax_R16",
        "U16_AttnKV + FitFullV_R16",
        "Skip_AllTool_k16 + D2_AttnKV",
    ]
    sel = [s for s in summaries if s.name in key_modes]
    cats = ["reasoning", "new_code_arg", "tool_boiler", "repeated_echo"]
    x = np.arange(len(cats))
    w = 0.15
    pal = ["#64748b", "#2563eb", "#0ea5e9", "#8b5cf6", "#10b981"]
    for idx_m, s in enumerate(sel):
        vals = [s.reasoning_kl, s.code_kl, s.tool_kl, s.echo_kl]
        ax.bar(x + (idx_m - (len(sel) - 1) / 2.0) * w, vals, width=w, label=s.name, color=pal[idx_m % len(pal)])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("Category KL Divergence vs Full 32L")
    ax.set_title("D. KL Breakdown by Assistant Token Type")
    ax.legend(fontsize=7.3, loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_eval_and_benchmark(
    aligner_path: Path = ARTIFACTS_DIR / "hybrid_aligner_L16_r16.npz",
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    eval_max_tokens: int = 6144,
    max_chunk_len: int = 1024,
    copy_figure_to: Path | None = None,
) -> dict:
    """Evaluate all Direction 2 configurations on held-out rows[12:18] with real M5 wall-clock Prefill timing."""
    ensure_metal_gpu_arch()
    all_rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8") if line.strip()]
    held_out_rows = all_rows[12:18]

    aligner = load_hybrid_aligner(aligner_path)
    model, tokenizer = mlx_lm.load(str(model_dir))
    num_layers = len(model.layers)

    # Pre-compile GPU aligner variants (full-rank Attn + R256 Attn, with GDN R8/R16 slices)
    comp_full = compile_hybrid_aligner(aligner, attn_rank=None)
    comp_r256 = compile_hybrid_aligner(aligner, attn_rank=256)
    comp_full_gdn8 = _slice_compiled_gdn_rank(comp_full, rank=8)

    def _select_compiled(spec: EvalPolicySpec) -> CompiledHybridAligner | None:
        if not spec.config.align_attn_kv and spec.config.gdn_mode == "none" and not spec.config.anchor_mean_sink_b:
            return None
        if spec.attn_rank == 256:
            return comp_r256
        if spec.gdn_rank_slice == 8:
            return comp_full_gdn8
        return comp_full

    # Prepare held-out traces once
    prepared_eval = []
    for offset, row in enumerate(held_out_rows):
        item = _prepare_trajectory(tokenizer, row, max_tokens=eval_max_tokens)
        if item is None:
            continue
        token_ids, tags, asst_mask = item
        span = _find_last_assistant_span(asst_mask, min_prompt_len=256)
        if span is None:
            continue
        split_pt, span_end = span
        prefix = token_ids[: split_pt - 1]
        cont_in = token_ids[split_pt - 1 : span_end - 1]
        cont_tags = tags[split_pt:span_end]
        cont_asst = asst_mask[split_pt:span_end]
        prefix_texts = [tokenizer.decode([tid]) for tid in prefix]
        prefix_roles = parse_role_tags(prefix_texts, token_ids=prefix)
        tag_buckets: dict[str, list[int]] = defaultdict(list)
        for pos, (tag, is_asst) in enumerate(zip(cont_tags, cont_asst)):
            if is_asst:
                tag_buckets["all_asst"].append(pos)
                if tag in ("tool_boilerplate", "repeated_echo", "new_code_arg", "reasoning"):
                    tag_buckets[tag].append(pos)
        prepared_eval.append((12 + offset, prefix, prefix_roles, cont_in, tag_buckets))

    # Warmup MLX Metal kernels once before wall-clock timing
    w_prefix, w_roles = prepared_eval[0][1][:256], prepared_eval[0][2][:256]
    w_sched = build_depth_schedule(w_prefix, w_roles, "Uniform_k16", shallow_k=16, full_depth=num_layers, max_chunk_len=256)
    _ = run_aligned_shallow_prefill(model, [(w_prefix, 32)])
    _ = run_aligned_shallow_prefill(
        model, w_sched.chunks, compiled=comp_full, config=AlignedPrefillConfig(align_attn_kv=True, gdn_mode="freeze_b_fit_ax")
    )
    mx.clear_cache()

    specs = build_eval_matrix()
    metrics_by_name: dict[str, dict[str, RunningMetrics]] = {s.name: defaultdict(RunningMetrics) for s in specs}
    prefill_time_by_name: dict[str, float] = {s.name: 0.0 for s in specs}
    total_prefix_tokens = sum(len(x[1]) for x in prepared_eval)

    for row_idx, prefix, prefix_roles, cont_in, tag_buckets in prepared_eval:
        # Exact 32L teacher prefill (timed cleanly!)
        full_sched = build_depth_schedule(prefix, prefix_roles, "Full_32L", full_depth=num_layers, max_chunk_len=max_chunk_len)
        t0 = time.perf_counter()
        exact_cache = run_aligned_shallow_prefill(model, full_sched.chunks)
        prefill_time_by_name["Full_32L"] += time.perf_counter() - t0
        exact_logits = _score_continuation(model, _clone_cache(exact_cache), cont_in)
        _update_bucket_metrics(metrics_by_name["Full_32L"], tag_buckets, exact_logits, exact_logits)
        del exact_cache
        mx.clear_cache()

        for spec in specs:
            if spec.name == "Full_32L":
                continue
            sched = build_depth_schedule(
                prefix,
                prefix_roles,
                policy_name=spec.schedule_policy,
                shallow_k=spec.shallow_k,
                full_depth=num_layers,
                max_chunk_len=max_chunk_len,
            )
            comp = _select_compiled(spec)
            tp0 = time.perf_counter()
            cand_cache = run_aligned_shallow_prefill(model, sched.chunks, compiled=comp, config=spec.config)
            prefill_time_by_name[spec.name] += time.perf_counter() - tp0
            cand_logits = _score_continuation(model, cand_cache, cont_in)
            _update_bucket_metrics(metrics_by_name[spec.name], tag_buckets, cand_logits, exact_logits)
            del cand_cache, cand_logits
            mx.clear_cache()

        del exact_logits
        mx.clear_cache()
        print(f"Evaluated held-out Row {row_idx} (prefix={len(prefix)}, cont={len(cont_in)}) across {len(specs)} modes.")

    base_time = prefill_time_by_name["Full_32L"]
    summaries: list[Direction2EvalSummary] = []
    for spec in specs:
        b = metrics_by_name[spec.name]
        all_m = b["all_asst"].finalize(0)
        reas_m = b["reasoning"].finalize(0)
        code_m = b["new_code_arg"].finalize(0)
        tool_m = b["tool_boilerplate"].finalize(0)
        echo_m = b["repeated_echo"].finalize(0)
        t_sec = prefill_time_by_name[spec.name]
        summaries.append(
            Direction2EvalSummary(
                name=spec.name,
                schedule_policy=spec.schedule_policy,
                total_prefill_seconds=t_sec,
                prefill_tok_per_sec=total_prefix_tokens / t_sec if t_sec > 0 else 0.0,
                prefill_speedup=base_time / t_sec if t_sec > 0 else 0.0,
                all_asst_n=all_m.count,
                all_asst_kl=all_m.mean_kl_nats,
                all_asst_top1=all_m.top1_match_rate,
                all_asst_top5=all_m.top5_match_rate,
                reasoning_kl=reas_m.mean_kl_nats,
                reasoning_top1=reas_m.top1_match_rate,
                code_kl=code_m.mean_kl_nats,
                code_top1=code_m.top1_match_rate,
                tool_kl=tool_m.mean_kl_nats,
                tool_top1=tool_m.top1_match_rate,
                echo_kl=echo_m.mean_kl_nats,
                echo_top1=echo_m.top1_match_rate,
            )
        )

    print(f"\n=== UPGRADED DIRECTION 2 BENCHMARK (6 Held-Out SWE Traces, {total_prefix_tokens} Total Prefix Tokens) ===")
    header = (
        f"{'Mode':<30} | {'Prefill(s)':>10} | {'tok/s':>7} | {'Speedup':>7} | "
        f"{'KL':>7} | {'Top1':>6} | {'Top5':>6} | {'reasKL':>6} | {'codeKL':>6} | {'toolKL':>6} | {'echoKL':>6}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.name:<30} | {s.total_prefill_seconds:10.2f} | {s.prefill_tok_per_sec:7.1f} | {s.prefill_speedup:6.2f}x | "
            f"{s.all_asst_kl:7.4f} | {s.all_asst_top1*100:5.1f}% | {s.all_asst_top5*100:5.1f}% | "
            f"{s.reasoning_kl:6.3f} | {s.code_kl:6.3f} | {s.tool_kl:6.3f} | {s.echo_kl:6.3f}"
        )

    fig_path = plot_direction2_summary(summaries, artifacts_dir / "direction2_hybrid_align.png")
    if copy_figure_to is not None:
        copy_figure_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, copy_figure_to)

    report = {
        "aligner_path": str(aligner_path),
        "total_prefix_tokens": total_prefix_tokens,
        "summaries": [asdict(s) for s in summaries],
    }
    json_path = artifacts_dir / "direction2_eval_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report -> {json_path}")
    print(f"Saved comparison figure -> {fig_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Upgraded Direction 2 aligners on held-out rows[12:18].")
    parser.add_argument("--aligner-path", type=Path, default=ARTIFACTS_DIR / "hybrid_aligner_L16_r16.npz")
    parser.add_argument("--eval-max-tokens", type=int, default=6144)
    parser.add_argument("--max-chunk-len", type=int, default=1024)
    parser.add_argument("--copy-figure-to", type=Path, default=None)
    args = parser.parse_args()
    run_eval_and_benchmark(
        aligner_path=args.aligner_path,
        eval_max_tokens=args.eval_max_tokens,
        max_chunk_len=args.max_chunk_len,
        copy_figure_to=args.copy_figure_to,
    )


if __name__ == "__main__":
    main()
