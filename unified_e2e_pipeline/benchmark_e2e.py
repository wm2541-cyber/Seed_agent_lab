"""End-to-end Apple Silicon M5 wall-clock benchmark (Prefill + Decode) for the Unified Pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx_lm
import numpy as np

from direction1_phase_shortlist.build_artifacts import ARTIFACTS_DIR as D1_ARTIFACTS
from direction1_phase_shortlist.phase_router import extract_phase_tag_ids
from direction1_phase_shortlist.shortlist_vocab import load_static_shortlist
from direction2_hybrid_prefill_align.aligned_prefill import compile_hybrid_aligner
from direction2_hybrid_prefill_align.kv_gdn_aligner import load_hybrid_aligner
from direction2_hybrid_prefill_align.train_aligners import ARTIFACTS_DIR as D2_ARTIFACTS
from seed_agent_lab.benchmark_closed_loop import extract_tool_prompt_ids
from seed_agent_lab.run_swe_rotation_probe import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR
from unified_e2e_pipeline import ensure_metal_gpu_arch
from unified_e2e_pipeline.pipeline import (
    UnifiedE2EResult,
    UnifiedPipelineConfig,
    UnifiedSpeculativePipeline,
    baseline_e2e_config,
    decode_only_d1_config,
    prefill_only_d2_config,
    unified_d23_d1_config,
    unified_d2_d1_config,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPO_FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class E2EModeAggregate:
    """Immutable aggregated E2E benchmark metrics for one pipeline configuration."""

    mode: str
    total_prompt_tokens: int
    total_gen_tokens: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    prefill_tok_s: float
    decode_tok_s: float
    prefill_speedup: float
    decode_speedup: float
    e2e_speedup: float
    exact_matches: int
    total_prompts: int
    mean_prefix_match_ratio: float
    tau: float


def _prefix_match_ratio(a: Sequence[int], b: Sequence[int]) -> float:
    if not b:
        return 1.0
    m = 0
    for x, y in zip(a, b):
        if x == y:
            m += 1
        else:
            break
    return m / len(b)


def plot_unified_e2e_summary(
    aggregates: Sequence[E2EModeAggregate],
    per_prompt: Sequence[dict],
    out_path: Path,
) -> Path:
    """Render a crisp 4-panel End-to-End M5 Speedup figure for the bilingual README."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), dpi=160)
    fig.suptitle(
        "Unified End-to-End Pipeline on Apple M5: Aligned Shallow Prefill (D2/D3) + Phase-Shortlist Decode (D1)",
        fontsize=13.2,
        fontweight="bold",
        y=0.98,
    )

    modes = [a.mode for a in aggregates]
    y_pos = np.arange(len(modes))

    # Panel A: Stacked End-to-End Wall-Clock Time (Prefill + Decode)
    ax = axes[0, 0]
    p_s = [a.prefill_seconds for a in aggregates]
    d_s = [a.decode_seconds for a in aggregates]
    tot_s = [a.total_seconds for a in aggregates]
    ax.barh(y_pos, p_s, label="Prefill (TTFT)", color="#3b82f6", height=0.58)
    ax.barh(y_pos, d_s, left=p_s, label="Decode", color="#10b981", height=0.58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modes, fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlabel("Total Wall-Clock Time Across 6 Held-Out SWE Prompts (s)")
    ax.set_title("A. End-to-End Wall-Clock Breakdown (Prefill + Decode)")
    ax.set_xlim(0, max(tot_s) * 1.28)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for idx, (t, a) in enumerate(zip(tot_s, aggregates)):
        ax.text(t + max(tot_s) * 0.015, idx, f"{t:.2f}s ({a.e2e_speedup:.2f}x)", va="center", fontsize=8.5, fontweight="bold")

    # Panel B: Component Speedups (Prefill vs Decode vs Total E2E)
    ax = axes[0, 1]
    non_base = [a for a in aggregates if a.mode != "Baseline_E2E"]
    x = np.arange(len(non_base))
    w = 0.25
    ax.bar(x - w, [a.prefill_speedup for a in non_base], width=w, label="Prefill Speedup", color="#3b82f6")
    ax.bar(x, [a.decode_speedup for a in non_base], width=w, label="Decode Speedup", color="#10b981")
    ax.bar(x + w, [a.e2e_speedup for a in non_base], width=w, label="Total E2E Speedup", color="#f59e0b")
    ax.axhline(1.0, color="#334155", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([a.mode for a in non_base], fontsize=8.5)
    ax.set_ylabel("Speedup vs Baseline_E2E (x)")
    ax.set_ylim(0, max(max(a.decode_speedup for a in non_base), max(a.e2e_speedup for a in non_base)) * 1.25)
    ax.set_title("B. Multiplicative Gain: Prefill × Decode -> E2E")
    ax.legend(fontsize=8.2, loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for i, a in enumerate(non_base):
        ax.text(i + w, a.e2e_speedup + 0.03, f"{a.e2e_speedup:.2f}x", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    # Panel C: Per-Prompt E2E Speedup (Baseline vs DecodeOnly_D1 vs Unified_D2_D1 vs Unified_D23_D1)
    ax = axes[1, 0]
    row_ids = sorted({r["row_idx"] for r in per_prompt})
    base_map = {r["row_idx"]: r["total_seconds"] for r in per_prompt if r["mode"] == "Baseline_E2E"}
    cmp_modes = ("DecodeOnly_D1", "PrefillOnly_D2", "Unified_D23_D1", "Unified_D2_D1")
    pal = ["#94a3b8", "#60a5fa", "#10b981", "#2563eb"]
    xr = np.arange(len(row_ids))
    wr = 0.2
    for idx_m, mname in enumerate(cmp_modes):
        mmap = {r["row_idx"]: r["total_seconds"] for r in per_prompt if r["mode"] == mname}
        vals = [base_map[rid] / mmap[rid] for rid in row_ids]
        ax.bar(xr + (idx_m - 1.5) * wr, vals, width=wr, label=mname, color=pal[idx_m])
    ax.axhline(1.0, color="#334155", linestyle=":", linewidth=1.0)
    ax.set_xticks(xr)
    ax.set_xticklabels([f"Row {rid}" for rid in row_ids], fontsize=9.0)
    ax.set_ylabel("Total E2E Speedup vs Baseline (x)")
    ax.set_ylim(0, 2.55)
    ax.set_title("C. Per-Prompt Total End-to-End Speedup (rows[12:18])")
    ax.legend(fontsize=8.0, loc="upper right", ncol=2, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel D: Throughput Summary (Prefill tok/s & Decode tok/s)
    ax = axes[1, 1]
    ax.bar(y_pos - 0.18, [a.prefill_tok_s for a in aggregates], width=0.35, label="Prefill (tok/s)", color="#3b82f6")
    ax.set_xticks(y_pos)
    ax.set_xticklabels(modes, fontsize=8.2, rotation=12)
    ax.set_ylabel("Prefill Throughput (tok/s)", color="#2563eb")
    ax.set_ylim(0, max(a.prefill_tok_s for a in aggregates) * 1.28)
    ax2 = ax.twinx()
    ax2.bar(y_pos + 0.18, [a.decode_tok_s for a in aggregates], width=0.35, label="Decode (tok/s)", color="#10b981")
    ax2.set_ylabel("Decode Throughput (tok/s)", color="#059669")
    ax2.set_ylim(0, max(a.decode_tok_s for a in aggregates) * 1.28)
    ax.set_title("D. Dual-Stage Throughput (Prefill vs Decode)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8.2, loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_e2e_benchmark(
    model_dir: Path = DEFAULT_MODEL_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
    w8_checkpoint: Path = CHECKPOINT_DIR / "layer8_ridge_asst.npz",
    shortlist_path: Path = D1_ARTIFACTS / "shortlist_8192.npz",
    aligner_path: Path = D2_ARTIFACTS / "hybrid_aligner_L16_r16.npz",
    artifacts_dir: Path = ARTIFACTS_DIR,
    max_prompt_tokens: int = 6144,
    max_new_tokens: int = 160,
    copy_figure_to: Path | None = None,
) -> dict:
    """Run full 5-mode End-to-End M5 benchmark on held-out SWE prompts rows[12:18]."""
    ensure_metal_gpu_arch()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8") if line.strip()]
    held_out = rows[12:18]

    model, tokenizer = mlx_lm.load(str(model_dir))
    phase_tags = extract_phase_tag_ids(tokenizer)
    eos_ids = frozenset(int(t) for t in getattr(tokenizer, "eos_token_ids", set()))

    w8 = np.load(w8_checkpoint)
    sl8 = load_static_shortlist(shortlist_path)
    aligner = load_hybrid_aligner(aligner_path)
    compiled_aligner = compile_hybrid_aligner(aligner, attn_rank=None)

    pipeline = UnifiedSpeculativePipeline(
        model=model,
        tokenizer=tokenizer,
        w8_weight=np.asarray(w8["weight"], dtype=np.float32),
        w8_bias=np.asarray(w8["bias"], dtype=np.float32),
        phase_tags=phase_tags,
        static_shortlist=sl8,
        compiled_prefill_aligner=compiled_aligner,
    )

    configs: tuple[UnifiedPipelineConfig, ...] = (
        baseline_e2e_config(),
        decode_only_d1_config(),
        prefill_only_d2_config(),
        unified_d23_d1_config(),
        unified_d2_d1_config(),
    )

    prompts: list[tuple[int, list[int]]] = []
    for offset, row in enumerate(held_out):
        pids = extract_tool_prompt_ids(tokenizer, row, max_prompt_tokens=max_prompt_tokens, enable_thinking=True)
        assert pids is not None
        prompts.append((12 + offset, pids))

    # Warmup kernels before timing
    _ = pipeline.generate(prompts[0][1][:256], max_new_tokens=8, eos_token_ids=eos_ids, config=baseline_e2e_config())
    _ = pipeline.generate(prompts[0][1][:256], max_new_tokens=8, eos_token_ids=eos_ids, config=unified_d2_d1_config())

    per_prompt_records: list[dict] = []
    by_mode: dict[str, list[dict]] = {c.name: [] for c in configs}

    for row_idx, pids in prompts:
        base_res: UnifiedE2EResult | None = None
        row_parts = [f"[Row {row_idx}] plen={len(pids)}"]
        for cfg in configs:
            res = pipeline.generate(pids, max_new_tokens=max_new_tokens, eos_token_ids=eos_ids, config=cfg)
            if base_res is None:
                base_res = res
            exact = res.tokens == base_res.tokens
            pmatch = _prefix_match_ratio(res.tokens, base_res.tokens)
            rec = {
                "row_idx": row_idx,
                "mode": cfg.name,
                "prompt_len": res.prompt_len,
                "gen_len": len(res.tokens),
                "prefill_seconds": res.prefill_seconds,
                "decode_seconds": res.decode_seconds,
                "total_seconds": res.total_seconds,
                "prefill_tok_s": res.prefill_tok_per_sec,
                "decode_tok_s": res.decode_tok_per_sec,
                "exact_match": exact,
                "prefix_match_ratio": pmatch,
                "verify_calls": res.verify_calls,
                "tau": res.mean_accepted_tokens,
            }
            per_prompt_records.append(rec)
            by_mode[cfg.name].append(rec)
            sp = base_res.total_seconds / res.total_seconds if res.total_seconds > 0 else 0.0
            row_parts.append(f"{cfg.name}={res.total_seconds:.2f}s({sp:.2f}x,P={res.prefill_seconds:.2f}/D={res.decode_seconds:.2f})")
        print(" | ".join(row_parts))

    base_rows = by_mode["Baseline_E2E"]
    base_p = sum(r["prefill_seconds"] for r in base_rows)
    base_d = sum(r["decode_seconds"] for r in base_rows)
    base_t = sum(r["total_seconds"] for r in base_rows)

    aggregates: list[E2EModeAggregate] = []
    for cfg in configs:
        rs = by_mode[cfg.name]
        tot_p_toks = sum(r["prompt_len"] for r in rs)
        tot_g_toks = sum(r["gen_len"] for r in rs)
        p_s = sum(r["prefill_seconds"] for r in rs)
        d_s = sum(r["decode_seconds"] for r in rs)
        t_s = sum(r["total_seconds"] for r in rs)
        vcalls = sum(r["verify_calls"] for r in rs)
        aggregates.append(
            E2EModeAggregate(
                mode=cfg.name,
                total_prompt_tokens=tot_p_toks,
                total_gen_tokens=tot_g_toks,
                prefill_seconds=p_s,
                decode_seconds=d_s,
                total_seconds=t_s,
                prefill_tok_s=tot_p_toks / p_s if p_s > 0 else 0.0,
                decode_tok_s=tot_g_toks / d_s if d_s > 0 else 0.0,
                prefill_speedup=base_p / p_s if p_s > 0 else 0.0,
                decode_speedup=base_d / d_s if d_s > 0 else 0.0,
                e2e_speedup=base_t / t_s if t_s > 0 else 0.0,
                exact_matches=sum(1 for r in rs if r["exact_match"]),
                total_prompts=len(rs),
                mean_prefix_match_ratio=float(np.mean([r["prefix_match_ratio"] for r in rs])),
                tau=tot_g_toks / vcalls if vcalls > 0 else 1.0,
            )
        )

    print("\n=== UNIFIED END-TO-END M5 BENCHMARK (rows[12:18], Prefill + Decode) ===")
    header = (
        f"{'Mode':<16} | {'Prefill(s)':>10} | {'Decode(s)':>9} | {'Total(s)':>8} | "
        f"{'P_Sp':>5} | {'D_Sp':>5} | {'E2E_Sp':>6} | {'P_tok/s':>7} | {'D_tok/s':>7} | {'Exact':>5} | {'Tau':>4}"
    )
    print(header)
    print("-" * len(header))
    for a in aggregates:
        print(
            f"{a.mode:<16} | {a.prefill_seconds:10.2f} | {a.decode_seconds:9.2f} | {a.total_seconds:8.2f} | "
            f"{a.prefill_speedup:4.2f}x | {a.decode_speedup:4.2f}x | {a.e2e_speedup:5.2f}x | "
            f"{a.prefill_tok_s:7.1f} | {a.decode_tok_s:7.2f} | {a.exact_matches}/{a.total_prompts:<3} | {a.tau:4.2f}"
        )

    fig_path = plot_unified_e2e_summary(aggregates, per_prompt_records, artifacts_dir / "unified_e2e_comparison.png")
    REPO_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, REPO_FIGURES_DIR / "unified_e2e_comparison.png")
    if copy_figure_to is not None:
        copy_figure_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fig_path, copy_figure_to)

    report = {
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "aggregates": [asdict(a) for a in aggregates],
        "per_prompt": per_prompt_records,
    }
    json_path = artifacts_dir / "unified_e2e_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report -> {json_path}")
    print(f"Saved E2E figure -> {fig_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Unified End-to-End M5 Benchmark (Prefill + Decode).")
    parser.add_argument("--max-prompt-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--copy-figure-to", type=Path, default=None)
    args = parser.parse_args()
    run_e2e_benchmark(
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        copy_figure_to=args.copy_figure_to,
    )


if __name__ == "__main__":
    main()
