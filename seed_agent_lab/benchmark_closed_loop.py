"""End-to-end closed-loop M5 benchmark: Baseline 32L vs LookupOnly vs L8 Ridge vs Hybrid speculative decoding."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx_lm
import numpy as np

from seed_agent_lab.closed_loop_engine import ClosedLoopResult, ClosedLoopSpeculativeEngine
from seed_agent_lab.run_swe_rotation_probe import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR
from seed_agent_lab.swe_loader import SWE_TOOLS_SCHEMA, normalize_swe_messages
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a closed-loop speculative decoding mode."""

    name: str
    use_neural: bool
    use_lookup: bool
    lookup_first: bool = True
    confidence_threshold: float = 0.70
    max_neural_len: int = 6
    max_lookup_len: int = 8
    ngram_size: int = 3


@dataclass(frozen=True)
class PromptEvalRecord:
    """Measured wall-clock result for one prompt and one decoding mode."""

    prompt_idx: int
    prompt_len: int
    mode_name: str
    gen_len: int
    exact_match: bool
    elapsed_seconds: float
    tokens_per_sec: float
    verify_calls: int
    mean_accepted_tokens: float
    total_neural_steps: int
    total_lookup_steps: int
    rollback_count: int
    draft_seconds: float
    verify_seconds: float
    rollback_seconds: float


DEFAULT_MODES: tuple[ModeConfig, ...] = (
    ModeConfig(name="LookupOnly", use_neural=False, use_lookup=True),
    ModeConfig(name="L8_Conf>=0.70", use_neural=True, use_lookup=False, confidence_threshold=0.70),
    ModeConfig(name="Hybrid_LookupFirst+L8(c>=0.70)", use_neural=True, use_lookup=True, confidence_threshold=0.70),
)


def extract_tool_prompt_ids(
    tokenizer,
    raw_row: dict,
    max_prompt_tokens: int = 2560,
    enable_thinking: bool = True,
) -> list[int] | None:
    """Slice a SWE-smith trajectory right before a later assistant tool_call turn within token budget."""
    raw_msgs = json.loads(raw_row["messages"]) if isinstance(raw_row["messages"], str) else raw_row["messages"]
    norm_msgs = normalize_swe_messages(raw_msgs)

    candidate_indices = [
        idx
        for idx, msg in enumerate(norm_msgs)
        if idx >= 2 and msg.get("role") == "assistant" and msg.get("tool_calls")
    ]
    best_ids: list[int] | None = None
    for cut_idx in candidate_indices:
        rendered = tokenizer.apply_chat_template(
            norm_msgs[:cut_idx],
            tools=SWE_TOOLS_SCHEMA,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        token_ids = tokenizer.encode(rendered)
        if len(token_ids) <= max_prompt_tokens:
            best_ids = token_ids
        elif best_ids is not None:
            break
    return best_ids


def _format_aggregate_rows(
    records: Sequence[PromptEvalRecord],
    mode_names: Sequence[str],
) -> list[str]:
    """Build formatted summary table lines from pure record collections."""
    by_mode: dict[str, list[PromptEvalRecord]] = {name: [] for name in mode_names}
    for rec in records:
        by_mode.setdefault(rec.mode_name, []).append(rec)

    base_recs = by_mode.get("Baseline_32L", [])
    base_total_toks = sum(r.gen_len for r in base_recs)
    base_total_secs = sum(r.elapsed_seconds for r in base_recs)
    base_tps = base_total_toks / base_total_secs if base_total_secs > 0 else 0.0

    lines = [
        "=== Closed-Loop M5 Benchmark Summary ===",
        f"{'Mode':<32} | {'Exact':<7} | {'Toks':<5} | {'Time(s)':<7} | {'Tok/s':<7} | {'Speedup':<7} | {'Tau':<5} | {'Draft%':<6} | {'Verify%':<7} | {'Roll%':<6}",
        "-" * 114,
    ]
    for name in mode_names:
        recs = by_mode.get(name, [])
        if not recs:
            continue
        tot_toks = sum(r.gen_len for r in recs)
        tot_secs = sum(r.elapsed_seconds for r in recs)
        tps = tot_toks / tot_secs if tot_secs > 0 else 0.0
        speedup = tps / base_tps if base_tps > 0 else 0.0
        tot_calls = sum(r.verify_calls for r in recs)
        tau = tot_toks / tot_calls if tot_calls > 0 else 0.0
        n_exact = sum(1 for r in recs if r.exact_match)
        draft_pct = 100.0 * sum(r.draft_seconds for r in recs) / tot_secs if tot_secs > 0 else 0.0
        verify_pct = 100.0 * sum(r.verify_seconds for r in recs) / tot_secs if tot_secs > 0 else 0.0
        roll_pct = 100.0 * sum(r.rollback_seconds for r in recs) / tot_secs if tot_secs > 0 else 0.0
        lines.append(
            f"{name:<32} | {n_exact}/{len(recs):<4} | {tot_toks:<5} | {tot_secs:<7.2f} | "
            f"{tps:<7.2f} | {speedup:<6.2f}x | {tau:<5.2f} | {draft_pct:<5.1f}% | {verify_pct:<6.1f}% | {roll_pct:<5.1f}%"
        )
    return lines


def run_benchmark(
    model_dir: Path,
    data_path: Path,
    ckpt_path: Path,
    max_new_tokens: int,
    max_prompt_tokens: int,
    enable_thinking: bool,
) -> None:
    """Run held-out closed-loop equivalence and real wall-clock throughput benchmark on M5."""
    rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8")]
    held_out_rows = rows[12:18]
    print(f"Loaded {len(held_out_rows)} held-out SWE-smith rows (rows[12:18])")

    model, tokenizer = mlx_lm.load(str(model_dir))
    ckpt = np.load(ckpt_path)
    engine = ClosedLoopSpeculativeEngine(
        model=model,
        w8_weight=ckpt["weight"],
        w8_bias=ckpt["bias"],
        draft_layer=int(ckpt["draft_layer"][0]),
    )
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])

    prompts: list[tuple[int, list[int]]] = []
    for offset, row in enumerate(held_out_rows, start=12):
        pids = extract_tool_prompt_ids(
            tokenizer,
            row,
            max_prompt_tokens=max_prompt_tokens,
            enable_thinking=enable_thinking,
        )
        if pids is not None:
            prompts.append((offset, pids))
    print(f"Extracted {len(prompts)} held-out tool-call prompts (thinking={enable_thinking}, max_new={max_new_tokens})")

    if not prompts:
        raise RuntimeError("No valid held-out prompts extracted.")

    # Warmup MLX Metal kernels for baseline, draft, verify, and GDN rollback before timing.
    warmup_ids = prompts[0][1]
    engine.generate_baseline(warmup_ids, max_new_tokens=16, eos_token_ids=eos_ids)
    engine.generate_speculative(
        warmup_ids,
        max_new_tokens=16,
        eos_token_ids=eos_ids,
        use_neural=True,
        use_lookup=True,
        confidence_threshold=0.0,
    )
    mx.clear_cache()

    records: list[PromptEvalRecord] = []
    for row_idx, prompt_ids in prompts:
        base_tokens, base_secs = engine.generate_baseline(
            prompt_ids, max_new_tokens=max_new_tokens, eos_token_ids=eos_ids
        )
        base_len = len(base_tokens)
        base_tps = base_len / base_secs if base_secs > 0 else 0.0
        records.append(
            PromptEvalRecord(
                prompt_idx=row_idx,
                prompt_len=len(prompt_ids),
                mode_name="Baseline_32L",
                gen_len=base_len,
                exact_match=True,
                elapsed_seconds=base_secs,
                tokens_per_sec=base_tps,
                verify_calls=base_len,
                mean_accepted_tokens=1.0,
                total_neural_steps=0,
                total_lookup_steps=0,
                rollback_count=0,
                draft_seconds=0.0,
                verify_seconds=base_secs,
                rollback_seconds=0.0,
            )
        )
        print(f"\n[Row {row_idx}] prompt_len={len(prompt_ids)}, generated={base_len} tokens | Baseline: {base_tps:.2f} tok/s ({base_secs:.2f}s)")

        for cfg in DEFAULT_MODES:
            res: ClosedLoopResult = engine.generate_speculative(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_ids,
                use_neural=cfg.use_neural,
                use_lookup=cfg.use_lookup,
                lookup_first=cfg.lookup_first,
                max_neural_len=cfg.max_neural_len,
                max_lookup_len=cfg.max_lookup_len,
                ngram_size=cfg.ngram_size,
                confidence_threshold=cfg.confidence_threshold,
            )
            exact = res.tokens == base_tokens
            rec = PromptEvalRecord(
                prompt_idx=row_idx,
                prompt_len=len(prompt_ids),
                mode_name=cfg.name,
                gen_len=len(res.tokens),
                exact_match=exact,
                elapsed_seconds=res.elapsed_seconds,
                tokens_per_sec=res.tokens_per_sec,
                verify_calls=res.verify_calls,
                mean_accepted_tokens=res.mean_accepted_tokens,
                total_neural_steps=res.total_neural_steps,
                total_lookup_steps=res.total_lookup_steps,
                rollback_count=res.rollback_count,
                draft_seconds=res.draft_seconds,
                verify_seconds=res.verify_seconds,
                rollback_seconds=res.rollback_seconds,
            )
            records.append(rec)
            speedup = res.tokens_per_sec / base_tps if base_tps > 0 else 0.0
            print(
                f"  {cfg.name:<30}: exact={exact} | {res.tokens_per_sec:5.2f} tok/s ({speedup:.2f}x) | "
                f"tau={res.mean_accepted_tokens:.2f} | calls={res.verify_calls} | "
                f"neural={res.total_neural_steps} lookup={res.total_lookup_steps} roll={res.rollback_count} | "
                f"t[D={res.draft_seconds:.2f} V={res.verify_seconds:.2f} R={res.rollback_seconds:.2f}]"
            )

    print()
    mode_order = ("Baseline_32L", *(m.name for m in DEFAULT_MODES))
    for line in _format_aggregate_rows(records, mode_order):
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--ckpt-path", type=Path, default=CHECKPOINT_DIR / "layer8_ridge_asst.npz")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=2560)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()
    run_benchmark(
        model_dir=args.model_dir,
        data_path=args.data_path,
        ckpt_path=args.ckpt_path,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        enable_thinking=not args.no_thinking,
    )


if __name__ == "__main__":
    main()
