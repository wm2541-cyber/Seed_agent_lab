"""End-to-end CLI runner for Direction 3 (Role-Aware Shallow Prefill) + Direction 2 SVD Diagnostic Blueprint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from direction3_role_prefill_svd import ensure_metal_gpu_arch
from direction3_role_prefill_svd.role_prefill_probe import (
    DeepLayerStateErrorSummary,
    PolicyAggregateMetrics,
    run_role_prefill_and_state_svd_probe,
)
from direction3_role_prefill_svd.weight_svd_probe import (
    LayerWeightSVDRecord,
    analyze_model_weights_svd,
    summarize_weight_svd_by_depth,
)
from seed_agent_lab.run_seed_coding_eval import estimate_mean_values_and_sinks
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    _prepare_trajectory,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _format_cat_brief(by_category: Sequence[tuple[str, tuple[int, float, float]]]) -> str:
    cat_map = {name: vals for name, vals in by_category}
    parts = []
    for short, key in (("reas", "reasoning"), ("code", "new_code_arg"), ("tool", "tool_boilerplate"), ("echo", "repeated_echo")):
        if key in cat_map:
            _n, kl, t1 = cat_map[key]
            parts.append(f"{short}:{kl:.3f}/{t1*100:.1f}%")
    return " | ".join(parts)


def plot_direction3_svd_summary(
    prefill_summary: Sequence[PolicyAggregateMetrics],
    oracle_summary: Sequence[PolicyAggregateMetrics],
    state_err_summaries: dict[str, tuple[DeepLayerStateErrorSummary, ...]],
    weight_records: Sequence[LayerWeightSVDRecord],
    out_path: Path,
) -> Path:
    """Render a 6-panel scientific diagnostic figure covering Direction 3 + Direction 2 SVD blueprint."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(19, 11.2), dpi=160)
    fig.suptitle(
        "Direction 3 & Direction 2 Prep: Role-Aware Shallow Prefill + State/Weight SVD Blueprint (Qwen3.5-4B-MLX-8bit)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    # Panel A: Role-Aware Prefill KL & Top-1 vs FLOPs Skipped
    ax = axes[0, 0]
    p_names = [p.policy_name for p in prefill_summary]
    p_kls = [p.all_asst_kl for p in prefill_summary]
    p_t1s = [p.all_asst_top1 * 100.0 for p in prefill_summary]
    p_skip = [p.mean_skipped_layer_frac * 100.0 for p in prefill_summary]
    y_pos = np.arange(len(p_names))
    colors = [
        "#ef4444" if "UserAsst" in n else "#10b981" if "Tool" in n else "#64748b"
        for n in p_names
    ]
    bars = ax.barh(y_pos, p_kls, color=colors, height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(p_names, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel("Continuation KL Divergence vs Full 32L (Lower is Better)")
    ax.set_title("A. Role-Aware Prefill vs Matched Counterfactuals")
    ax.set_xlim(0, max(p_kls) * 1.65 if p_kls else 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, kl, t1, sk in zip(bars, p_kls, p_t1s, p_skip):
        ax.text(
            bar.get_width() + max(p_kls) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"KL={kl:.3f} | T1={t1:.1f}% | -{sk:.0f}%L",
            va="center",
            fontsize=7.6,
        )

    # Panel B: Direction 2 Causal Oracle Ceiling on Uniform_k16
    ax = axes[0, 1]
    u16 = next((p for p in prefill_summary if p.policy_name == "Uniform_k16"), None)
    o_items = ([u16] if u16 is not None else []) + list(oracle_summary)
    o_names = [o.policy_name for o in o_items]
    o_kls = [o.all_asst_kl for o in o_items]
    o_t1s = [o.all_asst_top1 * 100.0 for o in o_items]
    yo = np.arange(len(o_names))
    o_colors = [
        "#64748b" if n == "Uniform_k16" else "#f59e0b" if "AttnKV_Only" in n else "#2563eb" if "GDN" in n else "#0ea5e9" if "Oracle_AttnKV +" in n else "#8b5cf6"
        for n in o_names
    ]
    bars_o = ax.barh(yo, o_kls, color=o_colors, height=0.62)
    ax.set_yticks(yo)
    ax.set_yticklabels(o_names, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlabel("Continuation KL Divergence vs Full 32L")
    ax.set_title("B. Direction 2 Causal Attribution (GDN b+ax vs Attn KV)")
    ax.set_xlim(0, max(o_kls) * 1.42 if o_kls else 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, kl, t1 in zip(bars_o, o_kls, o_t1s):
        ax.text(
            bar.get_width() + max(o_kls) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"KL={kl:.4f} (T1={t1:.1f}%)",
            va="center",
            fontsize=7.6,
        )

    # Panel C: Deep GDN Layer-by-Layer State Error (Uniform_k16 vs Skip_AllTool vs Matched_UserAsst)
    ax = axes[0, 2]
    for pname, color, marker in (
        ("Uniform_k16", "#64748b", "o"),
        ("Skip_AllTool_k16", "#10b981", "s"),
        ("Matched_SkipTool_k16", "#0ea5e9", "^"),
        ("Matched_SkipUserAsst_k16", "#ef4444", "D"),
    ):
        rows = state_err_summaries.get(pname, ())
        if not rows:
            continue
        xs = [r.layer_idx for r in rows]
        ys = [r.mean_rel_content_err for r in rows]
        ax.plot(xs, ys, label=pname, color=color, marker=marker, linewidth=1.8, markersize=5)
    ax.set_xlabel("Deep GDN Layer Index (L16..L30)")
    ax.set_ylabel("Relative Content Error ||E_ax|| / ||ax_true||")
    ax.set_title("C. Deep GDN Dynamic Content Error E_ax by Layer")
    ax.legend(fontsize=7.6, loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel D: E_ax Error Energy Share (Sink E_b vs Content E_ax) & Effective Rank of E_ax on Uniform_k16
    ax = axes[1, 0]
    u16_errs = state_err_summaries.get("Uniform_k16", ())
    if u16_errs:
        xs = [r.layer_idx for r in u16_errs]
        r90 = [r.mean_eax_r90 for r in u16_errs]
        eff = [r.mean_eax_eff_rank for r in u16_errs]
        sink_sh = [r.mean_sink_err_share * 100.0 for r in u16_errs]
        ax.plot(xs, eff, label="E_ax Effective Rank", color="#2563eb", marker="o", linewidth=2.0)
        ax.plot(xs, r90, label="E_ax Rank for 90% Energy (r90)", color="#8b5cf6", marker="s", linewidth=2.0)
        ax.set_xlabel("Deep GDN Layer Index (L16..L30)")
        ax.set_ylabel("Rank (out of 128)")
        ax.set_ylim(0, max(max(r90), max(eff), 20) * 1.35)
        ax2 = ax.twinx()
        ax2.bar(xs, sink_sh, alpha=0.22, color="#f59e0b", width=0.55, label="Sink E_b Error Share (%)")
        ax2.set_ylabel("Sink Error Share ||E_b||^2 / ||E||^2 (%)")
        ax2.set_ylim(0, 105)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper right", frameon=True)
    ax.set_title("D. Uniform_k16 GDN Error SVD (Target Rank for Direction 2)")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel E: Pretrained Weight SVD Effective Rank Across All 32 Layers
    ax = axes[1, 1]
    gdn_recs = [r for r in weight_records if r.layer_type == "gdn"]
    attn_recs = [r for r in weight_records if r.layer_type == "full_attn"]
    gx = [r.layer_idx for r in gdn_recs]
    for pname, color, style in (
        ("W_K", "#0ea5e9", "-"),
        ("W_V", "#2563eb", "-"),
        ("W_Z", "#10b981", "--"),
        ("W_out", "#f59e0b", "-."),
    ):
        ys = [dict(r.proj_spectra)[pname].effective_rank for r in gdn_recs]
        ax.plot(gx, ys, label=f"GDN {pname}", color=color, linestyle=style, marker="o", markersize=3.5, linewidth=1.7)
    if attn_recs:
        ax_x = [r.layer_idx for r in attn_recs]
        ax_v = [dict(r.proj_spectra)["W_V"].effective_rank for r in attn_recs]
        ax_o = [dict(r.proj_spectra)["W_O"].effective_rank for r in attn_recs]
        ax.plot(ax_x, ax_v, label="Attn W_V", color="#ef4444", marker="D", markersize=4.5, linewidth=1.7)
        ax.plot(ax_x, ax_o, label="Attn W_O", color="#991b1b", linestyle=":", marker="x", markersize=5.0, linewidth=1.7)
    ax.axvline(15.5, color="#334155", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Layer Index (0..31, dotted = L16 split)")
    ax.set_ylabel("Weight Matrix Effective Rank")
    ax.set_ylim(350, 2020)
    ax.set_title("E. Pretrained Weight SVD Effective Rank Across Depth")
    ax.legend(fontsize=7.3, loc="upper center", ncol=3, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel F: GDN Rank-1 Sink Alignment with Pretrained W_V & W_out Across Depth
    ax = axes[1, 2]
    gdn_aligned = [r for r in gdn_recs if r.mean_sink_top1_cos is not None]
    if gdn_aligned:
        lx = [r.layer_idx for r in gdn_aligned]
        cos1 = [float(r.mean_sink_top1_cos) for r in gdn_aligned]
        top4 = [float(r.mean_sink_top4_energy) for r in gdn_aligned]
        gain = [float(r.mean_wout_sink_gain) for r in gdn_aligned]
        ax.plot(lx, cos1, label="|cos(v_hat, u1(W_V))|", color="#2563eb", marker="o", linewidth=1.8)
        ax.plot(lx, top4, label="||U_4^T v_hat||^2 in W_V", color="#8b5cf6", marker="s", linewidth=1.8)
        ax.plot(lx, gain, label="||W_out v_hat|| / sigma_1(W_out)", color="#ef4444", marker="^", linewidth=1.8)
        ax.axvline(15.5, color="#334155", linestyle=":", linewidth=1.1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7.6, loc="upper right", frameon=True)
    ax.set_xlabel("GDN Layer Index (0..30)")
    ax.set_ylabel("Alignment / Gain Ratio (0..1)")
    ax.set_title("F. Rank-1 Sink Alignment with W_V & W_out")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_all_probes(
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    n_train: int = 12,
    train_max_tokens: int = 2048,
    eval_max_tokens: int = 6144,
    max_chunk_len: int = 1024,
    copy_figure_to: Path | None = None,
) -> dict:
    """Execute Direction 3 Role-Aware Prefill, GDN State-Error SVD, Causal Oracles, and Full-Model Weight SVD."""
    ensure_metal_gpu_arch()
    t0 = time.perf_counter()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8") if line.strip()]
    train_rows = rows[:n_train]
    held_out_rows = rows[n_train:]

    model, tokenizer = mlx_lm.load(str(model_dir))

    print(f"=== STEP 1: Estimating Training Sink Anchors (mean_v, mean_a) on {len(train_rows)} Train Traces ===")
    train_seqs = []
    for row in train_rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=train_max_tokens)
        if item is not None:
            train_seqs.append(item[0])
    mean_v, mean_a = estimate_mean_values_and_sinks(model, train_seqs)
    print(f"Estimated mean_v & mean_a across {len(train_seqs)} train traces ({len(mean_v)} GDN layers).")

    print(f"\n=== STEP 2: Running Probe 1 (Role-Aware Prefill) & Probe 2 (State SVD + Oracles) on {len(held_out_rows)} Held-Out Traces (max_tokens={eval_max_tokens}) ===")
    prefill_summary, oracle_summary, state_err_summaries, trace_meta = run_role_prefill_and_state_svd_probe(
        model=model,
        tokenizer=tokenizer,
        held_out_rows=held_out_rows,
        mean_v=mean_v,
        mean_a=mean_a,
        max_tokens=eval_max_tokens,
        max_chunk_len=max_chunk_len,
    )

    print("\n--- Held-Out Trace Role Composition ---")
    for tm in trace_meta:
        rc = tm["role_counts"]
        print(
            f"  Row {tm['row_idx']}: prefix={tm['prefix_len']:4d}, cont={tm['cont_len']:2d} | "
            f"sys={rc['system']:4d}, user={rc['user']:4d}, asst={rc['assistant']:4d}, "
            f"tool={rc['tool']:4d} ({tm['tool_fraction']*100:4.1f}%)"
        )

    print("\n--- PROBE 1: Role-Aware Shallow Prefill vs Matched-Budget Counterfactuals ---")
    for p in prefill_summary:
        print(
            f"  {p.policy_name:<30} | skipL={p.mean_skipped_layer_frac*100:4.1f}% (tok={p.mean_skipped_tokens:6.1f}, pos={p.mean_skipped_rel_pos:.2f}) | "
            f"ALL(n={p.all_asst_n:3d}): KL={p.all_asst_kl:.4f}, Top1={p.all_asst_top1*100:5.1f}%, Top5={p.all_asst_top5*100:5.1f}% | "
            f"{_format_cat_brief(p.by_category)}"
        )

    print("\n--- PROBE 2B: Direction-2 Causal Oracle Attribution on Uniform_k16 ---")
    for o in oracle_summary:
        print(
            f"  {o.policy_name:<36} | ALL(n={o.all_asst_n:3d}): KL={o.all_asst_kl:.4f}, Top1={o.all_asst_top1*100:5.1f}%, Top5={o.all_asst_top5*100:5.1f}% | "
            f"{_format_cat_brief(o.by_category)}"
        )

    print("\n--- PROBE 2A: Deep GDN State Error (E_b vs E_ax) & E_ax SVD Spectrum (Mean Across L16..L30) ---")
    for pname, layer_rows in state_err_summaries.items():
        if not layer_rows:
            continue
        print(
            f"  {pname:<26} | SinkShare={np.mean([r.mean_sink_err_share for r in layer_rows])*100:5.1f}% "
            f"| ContentShare={np.mean([r.mean_content_err_share for r in layer_rows])*100:5.1f}% "
            f"| relE_b={np.mean([r.mean_rel_sink_err for r in layer_rows]):.3f} "
            f"| relE_ax={np.mean([r.mean_rel_content_err for r in layer_rows]):.3f} "
            f"| E_ax effRank={np.mean([r.mean_eax_eff_rank for r in layer_rows]):4.1f} "
            f"| r50={np.mean([r.mean_eax_r50 for r in layer_rows]):4.1f}, r80={np.mean([r.mean_eax_r80 for r in layer_rows]):4.1f}, "
            f"r90={np.mean([r.mean_eax_r90 for r in layer_rows]):4.1f}, r95={np.mean([r.mean_eax_r95 for r in layer_rows]):4.1f} "
            f"| top8={np.mean([r.mean_eax_top8_energy for r in layer_rows])*100:4.1f}%, top16={np.mean([r.mean_eax_top16_energy for r in layer_rows])*100:4.1f}%"
        )

    print("\n=== STEP 3: Running Probe 3 (Full 32-Layer Pretrained Weight SVD & Sink Alignment) ===")
    weight_records = analyze_model_weights_svd(model, mean_v_by_layer=mean_v)
    weight_depth_summary = summarize_weight_svd_by_depth(weight_records, split_layer=16)
    for band, stats in weight_depth_summary.items():
        if band.startswith("gdn"):
            print(
                f"  {band:<18} (n={int(stats['n_layers']):2d}) | "
                f"effRank: W_Q={stats['W_Q_eff_rank']:6.1f}, W_K={stats['W_K_eff_rank']:6.1f}, "
                f"W_V={stats['W_V_eff_rank']:6.1f}, W_Z={stats['W_Z_eff_rank']:6.1f}, W_out={stats['W_out_eff_rank']:6.1f} | "
                f"r90(W_V/W_out)={stats['W_V_r90']:.0f}/{stats['W_out_r90']:.0f} | "
                f"SinkAlign: cos1={stats['sink_top1_cos']:.3f}, top4={stats['sink_top4_energy']*100:4.1f}%, "
                f"W_V_frac={stats['sink_wv_energy_frac']*100:4.2f}%, W_out_gain={stats['wout_sink_gain']:.3f} | "
                f"head_W_V effRank(full->perp)={stats['wv_head_eff_rank']:.1f}->{stats['wv_perp_eff_rank']:.1f}"
            )
        else:
            print(
                f"  {band:<18} (n={int(stats['n_layers']):2d}) | "
                f"effRank: W_K={stats['W_K_eff_rank']:6.1f}, W_V={stats['W_V_eff_rank']:6.1f}, W_O={stats['W_O_eff_rank']:6.1f} | "
                f"r90(W_K/W_V/W_O)={stats['W_K_r90']:.0f}/{stats['W_V_r90']:.0f}/{stats['W_O_r90']:.0f}"
            )

    fig_path = plot_direction3_svd_summary(
        prefill_summary=prefill_summary,
        oracle_summary=oracle_summary,
        state_err_summaries=state_err_summaries,
        weight_records=weight_records,
        out_path=artifacts_dir / "direction3_role_prefill_svd.png",
    )
    if copy_figure_to is not None:
        copy_figure_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, copy_figure_to)

    report = {
        "elapsed_seconds": time.perf_counter() - t0,
        "trace_meta": trace_meta,
        "prefill_summary": [asdict(p) for p in prefill_summary],
        "oracle_summary": [asdict(o) for o in oracle_summary],
        "state_error_svd": {
            k: [asdict(r) for r in rows] for k, rows in state_err_summaries.items()
        },
        "weight_svd_by_depth": weight_depth_summary,
        "weight_svd_layers": [asdict(r) for r in weight_records],
    }
    json_path = artifacts_dir / "direction3_svd_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nCompleted all probes in {report['elapsed_seconds']:.1f}s")
    print(f"Saved JSON report -> {json_path}")
    print(f"Saved diagnostic figure -> {fig_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Direction 3 Role-Aware Prefill & Direction 2 SVD Probes.")
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument("--train-max-tokens", type=int, default=2048)
    parser.add_argument("--eval-max-tokens", type=int, default=6144)
    parser.add_argument("--max-chunk-len", type=int, default=1024)
    parser.add_argument("--copy-figure-to", type=Path, default=None)
    args = parser.parse_args()
    run_all_probes(
        n_train=args.n_train,
        train_max_tokens=args.train_max_tokens,
        eval_max_tokens=args.eval_max_tokens,
        max_chunk_len=args.max_chunk_len,
        copy_figure_to=args.copy_figure_to,
    )


if __name__ == "__main__":
    main()
