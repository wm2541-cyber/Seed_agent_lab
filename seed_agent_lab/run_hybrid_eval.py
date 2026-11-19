"""Compare Layer-8 Linear Rotation, Prompt Lookup, and Dual-Engine Hybrid Drafting on SWE-smith edits."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx_lm
import numpy as np

from seed_agent_lab.analysis import classify_agent_token_texts, compute_layer_exit_metrics
from seed_agent_lab.hybrid_speculative import HybridSpecStats, simulate_hybrid_policy
from seed_agent_lab.rotation import apply_linear_rotation, fit_linear_rotation
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    RunningMetrics,
    _apply_lm_head,
    _prepare_trajectory,
    collect_normed_hidden_and_states,
)


def _aggregate_policy_stats(items: Sequence[HybridSpecStats]) -> HybridSpecStats:
    """Combine per-trajectory policy stats into an exact token-weighted aggregate."""
    if not items:
        return HybridSpecStats("empty", 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 0)
    total_tokens = sum(s.total_tokens for s in items)
    total_verifies = sum(s.verify_calls for s in items)
    total_neural = sum(s.avg_neural_steps * s.verify_calls for s in items)
    total_lookup = sum(s.avg_lookup_steps * s.verify_calls for s in items)
    total_cost = sum(s.avg_weight_read_cost * s.verify_calls for s in items)
    # Approximate weighted acceptance rate across runs by drafted step weights.
    total_drafts = total_neural + total_lookup
    weighted_acc = (
        sum(s.acceptance_rate * (s.avg_neural_steps + s.avg_lookup_steps) * s.verify_calls for s in items) / total_drafts
        if total_drafts > 0
        else 0.0
    )
    mean_acc = total_tokens / total_verifies
    avg_cost = total_cost / total_verifies
    return HybridSpecStats(
        policy_name=items[0].policy_name,
        mean_accepted_tokens=mean_acc,
        acceptance_rate=weighted_acc,
        avg_neural_steps=total_neural / total_verifies,
        avg_lookup_steps=total_lookup / total_verifies,
        avg_weight_read_cost=avg_cost,
        theoretical_speedup=mean_acc / avg_cost,
        verify_calls=total_verifies,
        total_tokens=total_tokens,
    )


def run_hybrid_eval(
    model_dir: Path,
    data_path: Path,
    draft_layer: int,
    n_train: int,
    max_tokens: int,
    ridge_lambda: float,
) -> None:
    """Fit W_k on training SWE trajectories and benchmark dual-engine drafting up to 3.5K tokens."""
    rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8")]
    model, tokenizer = mlx_lm.load(str(model_dir))
    num_layers = len(model.layers)
    depths = (draft_layer, num_layers)
    depth_ratio = draft_layer / num_layers

    prepared = []
    for row in rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is not None:
            prepared.append(item)

    train_items = prepared[:n_train]
    test_items = prepared[n_train:]
    print(f"Prepared trajectories: train={len(train_items)}, held_out={len(test_items)}, max_tokens={max_tokens}")

    train_src: list[np.ndarray] = []
    train_tgt: list[np.ndarray] = []
    for token_ids, _tags, _mask in train_items:
        normed, _ = collect_normed_hidden_and_states(model, token_ids[:-1], depths)
        train_src.append(normed[draft_layer])
        train_tgt.append(normed[num_layers])
        del normed
        mx.clear_cache()

    src_mat = np.concatenate(train_src, axis=0)
    tgt_mat = np.concatenate(train_tgt, axis=0)
    print(f"Fitting linear rotation W_{draft_layer} on {src_mat.shape[0]} tokens...")
    rot = fit_linear_rotation(src_mat, tgt_mat, ridge_lambda=ridge_lambda)
    del train_src, train_tgt, src_mat, tgt_mat
    gc.collect()

    rot_metrics: dict[str, RunningMetrics] = defaultdict(RunningMetrics)
    policy_runs: dict[str, list[HybridSpecStats]] = defaultdict(list)

    for token_ids, tags, asst_mask in test_items:
        input_ids = token_ids[:-1]
        target_tags = tags[1:]
        target_asst = asst_mask[1:]
        asst_positions = [i for i, flag in enumerate(target_asst) if flag]
        if not asst_positions:
            continue

        normed, _ = collect_normed_hidden_and_states(model, input_ids, depths)
        full_h = mx.array(normed[num_layers][asst_positions], dtype=mx.float16)
        full_logits_mx = _apply_lm_head(model, full_h)
        mx.eval(full_logits_mx)
        full_logits = np.array(full_logits_mx.astype(mx.float32), copy=True)
        full_top1 = np.argmax(full_logits, axis=-1).tolist()
        del full_h, full_logits_mx

        rot_h = apply_linear_rotation(rot, normed[draft_layer][asst_positions])
        rot_logits_mx = _apply_lm_head(model, mx.array(rot_h, dtype=mx.float16))
        mx.eval(rot_logits_mx)
        rot_logits = np.array(rot_logits_mx.astype(mx.float32), copy=True)
        rot_top1 = np.argmax(rot_logits, axis=-1).tolist()
        rot_p = np.exp(rot_logits - np.max(rot_logits, axis=-1, keepdims=True))
        rot_p /= np.sum(rot_p, axis=-1, keepdims=True)
        rot_max_probs = np.max(rot_p, axis=-1).tolist()
        del rot_logits_mx, rot_p, normed

        pos_to_local = {pos: j for j, pos in enumerate(asst_positions)}
        tag_local_idxs: dict[str, list[int]] = defaultdict(list)
        for pos in asst_positions:
            tag_local_idxs[target_tags[pos]].append(pos_to_local[pos])
        for tag, idxs in tag_local_idxs.items():
            rot_metrics[tag] = rot_metrics[tag].add(
                compute_layer_exit_metrics(draft_layer, rot_logits[idxs], full_logits[idxs])
            )
        del rot_logits, full_logits

        # Evaluate 5 policies on the exact same held-out trajectory!
        policies = [
            dict(policy_name="1_Lookup_Only(n=3,L=8)", use_neural=False, use_lookup=True, lookup_first=True, max_neural_len=0, max_lookup_len=8, confidence_threshold=0.0),
            dict(policy_name=f"2_L{draft_layer}_Only(L=6)", use_neural=True, use_lookup=False, lookup_first=False, max_neural_len=6, max_lookup_len=0, confidence_threshold=0.0),
            dict(policy_name=f"3_L{draft_layer}_Conf>=0.70(L=6)", use_neural=True, use_lookup=False, lookup_first=False, max_neural_len=6, max_lookup_len=0, confidence_threshold=0.70),
            dict(policy_name=f"4_Hybrid_LookupFirst+L{draft_layer}(c>=0.7)", use_neural=True, use_lookup=True, lookup_first=True, max_neural_len=6, max_lookup_len=8, confidence_threshold=0.70),
            dict(policy_name=f"5_Hybrid_L{draft_layer}_Conf>=0.7->Lookup", use_neural=True, use_lookup=True, lookup_first=False, max_neural_len=6, max_lookup_len=8, confidence_threshold=0.70),
        ]
        for cfg in policies:
            st = simulate_hybrid_policy(
                full_sequence_tokens=token_ids,
                assistant_positions=asst_positions,
                shallow_top1_tokens=rot_top1,
                target_top1_tokens=full_top1,
                shallow_max_probs=rot_max_probs,
                shallow_depth_ratio=depth_ratio,
                **cfg,
            )
            policy_runs[cfg["policy_name"]].append(st)

        gc.collect()
        mx.clear_cache()

    print(f"\n=== Layer {draft_layer}/{num_layers} ({100*depth_ratio:.0f}%) Rotated Accuracy across {max_tokens}-tok SWE turns ===")
    for tag in ("tool_boilerplate", "repeated_echo", "new_code_arg", "reasoning"):
        m = rot_metrics[tag].finalize(draft_layer)
        if m.count:
            print(f"  {tag:17s} (n={m.count:4d}): top1={m.top1_match_rate:.3f}, top5={m.top5_match_rate:.3f}, kl={m.mean_kl_nats:.2f}")

    print("\n=== Dual-Engine Speculative Comparison (All Held-Out Assistant Tokens) ===")
    for name, runs in policy_runs.items():
        agg = _aggregate_policy_stats(runs)
        print(
            f"  {agg.policy_name:40s} | tau={agg.mean_accepted_tokens:4.2f} | "
            f"acc={agg.acceptance_rate:.3f} | L{draft_layer}_steps={agg.avg_neural_steps:4.2f} | "
            f"lookup_steps={agg.avg_lookup_steps:4.2f} | cost={agg.avg_weight_read_cost:4.2f} | "
            f"speedup={agg.theoretical_speedup:4.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dual-Engine Layer-8 + Prompt Lookup on SWE-smith.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--draft-layer", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--ridge-lambda", type=float, default=50.0)
    args = parser.parse_args()
    run_hybrid_eval(
        model_dir=args.model_dir,
        data_path=args.data_path,
        draft_layer=args.draft_layer,
        n_train=args.n_train,
        max_tokens=args.max_tokens,
        ridge_lambda=args.ridge_lambda,
    )


if __name__ == "__main__":
    main()
