"""Generate a publication-grade 2x2 comparison figure for Qwen3.5-4B M5 speculative results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


@dataclass(frozen=True)
class OfflineTokenAccuracy:
    """Observed held-out Layer-8 Top-1 match rates against Layer-32 from task-357.log."""

    method: str
    tool_boilerplate: float
    new_code_arg: float
    reasoning: float
    repeated_echo: float


@dataclass(frozen=True)
class ClosedLoopTraceSpeed:
    """Observed per-trace real M5 throughput (tok/s) from task-493.log."""

    label: str
    baseline: float
    lookup_only: float
    l8_only: float
    hybrid: float


@dataclass(frozen=True)
class ModeAggregate:
    """Observed aggregate M5 closed-loop metrics across 6 held-out traces (468 tokens)."""

    name: str
    total_seconds: float
    tokens_per_sec: float
    speedup: float
    tau: float
    draft_pct: float
    verify_pct: float
    rollback_pct: float


OFFLINE_ACCURACIES: tuple[OfflineTokenAccuracy, ...] = (
    OfflineTokenAccuracy("1_Ridge_AllTokens", 0.925, 0.806, 0.561, 0.269),
    OfflineTokenAccuracy("2_Ridge_AssistantOnly", 0.993, 0.883, 0.757, 0.478),
    OfflineTokenAccuracy("4_KL_SwiGLU(r=512)", 0.985, 0.900, 0.740, 0.582),
)

TRACE_SPEEDS: tuple[ClosedLoopTraceSpeed, ...] = (
    ClosedLoopTraceSpeed("Row 12\n(70 tok)", 24.84, 36.72, 35.22, 42.64),
    ClosedLoopTraceSpeed("Row 13\n(67 tok)", 25.15, 34.91, 35.28, 43.32),
    ClosedLoopTraceSpeed("Row 14\n(62 tok)", 22.88, 36.24, 35.61, 43.60),
    ClosedLoopTraceSpeed("Row 15\n(76 tok)", 25.45, 33.81, 33.90, 38.62),
    ClosedLoopTraceSpeed("Row 16\n(134 tok\nLong Think)", 23.85, 33.97, 23.63, 27.40),
    ClosedLoopTraceSpeed("Row 17\n(59 tok)", 25.74, 36.81, 33.40, 44.20),
    ClosedLoopTraceSpeed("Aggregate\n(468 tok)", 24.52, 35.10, 30.57, 36.55),
)

AGGREGATES: tuple[ModeAggregate, ...] = (
    ModeAggregate("Baseline (32L)", 19.09, 24.52, 1.00, 1.00, 0.0, 100.0, 0.0),
    ModeAggregate("LookupOnly", 13.33, 35.10, 1.43, 1.50, 0.2, 99.3, 0.5),
    ModeAggregate("L8_Conf>=0.70", 15.31, 30.57, 1.25, 2.45, 58.3, 41.3, 0.4),
    ModeAggregate("Hybrid (Lookup+L8)", 12.80, 36.55, 1.49, 2.54, 45.7, 53.6, 0.7),
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "Heiti SC", "Hiragino Sans GB", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
        }
    )


def _annotate_bars(ax, rects, fmt: str = "{:.1f}", fontsize: int = 8, y_offset: float = 0.015) -> None:
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * y_offset
    for rect in rects:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + pad,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#1E293B",
            fontweight="medium",
        )


def build_comparison_figure(out_paths: tuple[Path, ...]) -> None:
    """Render 2x2 summary dashboard from observed offline + M5 closed-loop metrics."""
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.6), dpi=220)
    fig.suptitle(
        "Qwen3.5-4B-MLX-8bit (Apple M5): Layer-8 Linear Rotation + Prompt Lookup 实测全景对比",
        fontsize=15.0,
        fontweight="bold",
        color="#0F172A",
        y=0.985,
    )
    fig.text(
        0.5,
        0.950,
        "Training: 48 SWE-smith traces (10,908 Assistant tokens)  |  Held-out Closed-Loop: 6 SWE-smith traces (468 tokens, 100% Exact Greedy Match)",
        ha="center",
        fontsize=10.2,
        color="#475569",
    )

    # Panel A: Held-out Layer-8 Top-1 Accuracy by Agent Token Category
    ax_a = axes[0, 0]
    cats = ("tool_boilerplate\n(n=455)", "new_code_arg\n(n=180)", "reasoning\n(n=711)", "repeated_echo\n(n=67)")
    x = np.arange(len(cats))
    w = 0.25
    colors_a = ("#94A3B8", "#2563EB", "#F59E0B")
    ax_a.set_ylim(0, 1.15)
    for idx, acc in enumerate(OFFLINE_ACCURACIES):
        vals = [acc.tool_boilerplate, acc.new_code_arg, acc.reasoning, acc.repeated_echo]
        rects = ax_a.bar(x + (idx - 1) * w, vals, width=w, label=acc.method, color=colors_a[idx], edgecolor="white", linewidth=0.6)
        _annotate_bars(ax_a, rects, fmt="{:.1%}", fontsize=7.5, y_offset=0.015)

    ax_a.annotate(
        "+19.6pp",
        xy=(2.0, 0.815),
        xytext=(2.0, 0.96),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#1D4ED8",
        arrowprops={"arrowstyle": "->", "color": "#1D4ED8", "lw": 1.3},
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cats, fontsize=9.2)
    ax_a.set_ylabel("Top-1 Match Rate vs Layer 32", fontsize=10, fontweight="bold", color="#1E293B")
    ax_a.set_title("A. 离线单步对齐效果：Assistant-Only Ridge 去噪跃升", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax_a.grid(axis="y", zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5, loc="upper right")

    # Panel B: Real M5 Wall-Clock Throughput per Held-Out Trace
    ax_b = axes[0, 1]
    x_b = np.arange(len(TRACE_SPEEDS))
    wb = 0.20
    colors_b = ("#64748B", "#0EA5E9", "#F59E0B", "#10B981")
    series_b = (
        ("Baseline (32L)", [t.baseline for t in TRACE_SPEEDS]),
        ("LookupOnly", [t.lookup_only for t in TRACE_SPEEDS]),
        ("L8_Conf>=0.70", [t.l8_only for t in TRACE_SPEEDS]),
        ("Hybrid (Lookup+L8)", [t.hybrid for t in TRACE_SPEEDS]),
    )
    ax_b.set_ylim(0, 55)
    for idx, (label, vals) in enumerate(series_b):
        rects = ax_b.bar(x_b + (idx - 1.5) * wb, vals, width=wb, label=label, color=colors_b[idx], edgecolor="white", linewidth=0.5)
        if label in ("Baseline (32L)", "Hybrid (Lookup+L8)"):
            _annotate_bars(ax_b, rects, fmt="{:.2f}", fontsize=6.5, y_offset=0.015)

    ax_b.axvline(5.5, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax_b.text(1.75, 49.5, "Tool-Call 轨迹 (Row 12-15,17): Hybrid 38.62~44.20 tok/s (1.52x~1.91x)", fontsize=8.0, color="#047857", fontweight="bold", ha="center")
    ax_b.text(4.0, 38.8, "Row 16 长思考\nL8 退化至 0.99x", fontsize=7.8, color="#B45309", fontweight="bold", ha="center")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([t.label for t in TRACE_SPEEDS], fontsize=8.5)
    ax_b.set_ylabel("Wall-Clock Speed (tokens / sec)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_b.set_title("B. 真实 M5 闭环逐条吞吐对比 (6/6 Exact Match)", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax_b.grid(axis="y", zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.2, ncol=2, loc="upper right")

    # Panel C: Theoretical Layer-Ratio Roofline vs Real M5 End-to-End Speedup
    ax_c = axes[1, 0]
    policies_c = ("L8_Conf>=0.70", "Hybrid (Lookup+L8)")
    xc = np.arange(len(policies_c))
    wc = 0.30
    offline_speedup = (2.12, 2.47)
    real_speedup = (1.25, 1.49)
    offline_tau = (3.59, 3.82)
    real_tau = (2.45, 2.54)

    ax_c.set_ylim(0, 3.10)
    r_off = ax_c.bar(xc - wc / 2, offline_speedup, width=wc, label="离线理论 Roofline (仅算 8/32 层比例 + Teacher-Forcing)", color="#93C5FD", edgecolor="#2563EB", linewidth=1.0, zorder=2)
    r_real = ax_c.bar(xc + wc / 2, real_speedup, width=wc, label="M5 真实端到端实测 (含 0.64B LM_head + 自回归回滚)", color="#059669", edgecolor="white", linewidth=0.8, zorder=2)

    text_bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2}
    for rect, tau in zip(r_off, offline_tau):
        h = rect.get_height()
        ax_c.text(rect.get_x() + rect.get_width() / 2, h + 0.05, f"{h:.2f}x\n(tau={tau:.2f})", ha="center", va="bottom", fontsize=8.8, color="#1E3A8A", fontweight="bold", bbox=text_bbox, zorder=4)
    for rect, tau in zip(r_real, real_tau):
        h = rect.get_height()
        ax_c.text(rect.get_x() + rect.get_width() / 2, h + 0.06, f"{h:.2f}x\n(tau={tau:.2f})", ha="center", va="bottom", fontsize=8.8, color="#065F46", fontweight="bold", bbox=text_bbox, zorder=4)

    ax_c.axhline(1.0, color="#64748B", linestyle="--", linewidth=1.0, zorder=1)
    ax_c.axhline(1.43, color="#0EA5E9", linestyle="-.", linewidth=1.2, label="LookupOnly 实测基线 (1.43x, tau=1.50)", zorder=1)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(policies_c, fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Speedup vs Baseline 32L (x)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_c.set_title("C. 离线理论加速 vs M5 实机兑现差距归因", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax_c.grid(axis="y", zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.3, loc="upper left")

    # Panel D: Wall-Clock Time Breakdown across 468 Tokens (Seconds)
    ax_d = axes[1, 1]
    xd = np.arange(len(AGGREGATES))
    wd = 0.52
    verify_s = np.array([m.total_seconds * m.verify_pct / 100.0 for m in AGGREGATES])
    draft_s = np.array([m.total_seconds * m.draft_pct / 100.0 for m in AGGREGATES])
    roll_s = np.array([m.total_seconds * m.rollback_pct / 100.0 for m in AGGREGATES])

    ax_d.set_ylim(0, 23.0)
    ax_d.bar(xd, verify_s, width=wd, label="Verify (Layers 8..31 / 0..31)", color="#3B82F6", edgecolor="white")
    ax_d.bar(xd, draft_s, width=wd, bottom=verify_s, label="Draft (Layers 0..7 + W_8 + 0.64B LM_head)", color="#F59E0B", edgecolor="white")
    ax_d.bar(xd, roll_s, width=wd, bottom=verify_s + draft_s, label="Rollback (KV trim + GDN O(1) stash)", color="#EF4444", edgecolor="white")

    for i, m in enumerate(AGGREGATES):
        roll_tag = f"\nRoll {m.rollback_pct:.1f}%" if m.rollback_pct > 0 else ""
        ax_d.text(
            i,
            m.total_seconds + 0.35,
            f"{m.total_seconds:.2f}s ({m.tokens_per_sec:.2f} tok/s){roll_tag}",
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold",
            color="#0F172A",
        )
        if m.draft_pct > 10:
            ax_d.text(i, verify_s[i] + draft_s[i] / 2.0, f"Draft\n{m.draft_pct:.1f}%", ha="center", va="center", fontsize=8.2, color="white", fontweight="bold")
        ax_d.text(i, verify_s[i] / 2.0, f"Verify\n{m.verify_pct:.1f}%", ha="center", va="center", fontsize=8.2, color="white", fontweight="bold")

    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([m.name for m in AGGREGATES], fontsize=9.2, fontweight="bold")
    ax_d.set_ylabel("Total Wall-Clock Time for 468 Tokens (s)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_d.set_title("D. 耗时拆解：回滚开销 <0.7%，Draft 词表投影是下一瓶颈", fontsize=11.5, fontweight="bold", pad=10, color="#0F172A")
    ax_d.grid(axis="y", zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.3, loc="upper right")

    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.925))
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        print(f"Saved comparison figure -> {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_repo_out = Path(__file__).resolve().parent.parent / "figures" / "closed_loop_comparison.png"
    parser.add_argument("--out", type=Path, default=default_repo_out)
    parser.add_argument("--extra-out", type=Path, default=None)
    args = parser.parse_args()
    targets = (args.out, args.extra_out) if args.extra_out is not None else (args.out,)
    build_comparison_figure(targets)


if __name__ == "__main__":
    main()
