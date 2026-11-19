"""Evaluate Untrained Logit Lens vs Closed-Form Linear Rotation Probe on SWE-smith trajectories."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from seed_agent_lab.analysis import (
    DepthMetrics,
    SinkStats,
    classify_agent_token_texts,
    compute_layer_exit_metrics,
    compute_state_sink_stats,
    simulate_greedy_speculative,
    summarize_sink_stats,
)
from seed_agent_lab.rotation import LinearRotationMap, apply_linear_rotation, fit_linear_rotation
from seed_agent_lab.swe_loader import SWE_TOOLS_SCHEMA, assistant_token_mask, normalize_swe_messages

DEFAULT_MODEL_DIR = Path.home() / "Models" / "Qwen3.5-4B-MLX-8bit"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "swe_smith_tool_sample.jsonl"


@dataclass(frozen=True)
class RunningMetrics:
    """Immutable running sums for memory-safe streaming metric aggregation."""

    top1_hits: float = 0.0
    top5_hits: float = 0.0
    kl_sum: float = 0.0
    count: int = 0

    def add(self, batch: DepthMetrics) -> RunningMetrics:
        return RunningMetrics(
            top1_hits=self.top1_hits + batch.top1_match_rate * batch.count,
            top5_hits=self.top5_hits + batch.top5_match_rate * batch.count,
            kl_sum=self.kl_sum + batch.mean_kl_nats * batch.count,
            count=self.count + batch.count,
        )

    def finalize(self, depth: int) -> DepthMetrics:
        if self.count == 0:
            return DepthMetrics(depth=depth, top1_match_rate=0.0, top5_match_rate=0.0, mean_kl_nats=0.0, count=0)
        return DepthMetrics(
            depth=depth,
            top1_match_rate=self.top1_hits / self.count,
            top5_match_rate=self.top5_hits / self.count,
            mean_kl_nats=self.kl_sum / self.count,
            count=self.count,
        )


def _apply_lm_head(model, normed_hidden: mx.array) -> mx.array:
    """Project final-normed hidden states to vocabulary logits."""
    lm = model.language_model
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(normed_hidden)
    return lm.lm_head(normed_hidden)


def collect_normed_hidden_and_states(
    model,
    token_ids: Sequence[int],
    probe_depths: Sequence[int],
) -> tuple[dict[int, np.ndarray], tuple[tuple[int, np.ndarray], ...]]:
    """Run one forward pass, returning final-normed hidden states (T, D) per depth and linear states."""
    text_model = model.language_model.model
    cache = model.make_cache()
    inputs = mx.array([list(token_ids)], dtype=mx.int32)
    hidden = text_model.embed_tokens(inputs)

    fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx]) if text_model.fa_idx is not None else None
    ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx]) if text_model.ssm_idx is not None else None

    depth_set = set(probe_depths)
    normed_by_depth: dict[int, np.ndarray] = {}

    for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        depth = idx + 1
        if depth in depth_set:
            normed = text_model.norm(hidden)[0]
            mx.eval(normed)
            normed_by_depth[depth] = np.array(normed.astype(mx.float32), copy=True)

    linear_states: list[tuple[int, np.ndarray]] = []
    for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
        if layer.is_linear and layer_cache[1] is not None:
            mx.eval(layer_cache[1])
            state_np = np.array(layer_cache[1][0].astype(mx.float32), copy=True)
            linear_states.append((idx, state_np))

    del cache, hidden, fa_mask, ssm_mask
    mx.clear_cache()
    return normed_by_depth, tuple(linear_states)


def _prepare_trajectory(
    tokenizer,
    raw_row: dict,
    max_tokens: int,
) -> tuple[list[int], tuple[str, ...], tuple[bool, ...]] | None:
    """Render one SWE trajectory via Qwen3.5 chat template and return ids, semantic tags, assistant mask."""
    raw_msgs = json.loads(raw_row["messages"]) if isinstance(raw_row["messages"], str) else raw_row["messages"]
    norm_msgs = normalize_swe_messages(raw_msgs)
    if len(norm_msgs) < 4:
        return None
    rendered = tokenizer.apply_chat_template(
        norm_msgs,
        tools=SWE_TOOLS_SCHEMA,
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids = tokenizer.encode(rendered)[:max_tokens]
    if len(token_ids) < 128:
        return None
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]
    tags = classify_agent_token_texts(token_texts)
    asst_mask = assistant_token_mask(token_texts)
    return token_ids, tags, asst_mask


def run_swe_rotation_experiment(
    model_dir: Path,
    data_path: Path,
    depths: Sequence[int],
    n_train: int,
    max_tokens: int,
    ridge_lambda: float,
) -> None:
    """Fit closed-form rotation probes on training SWE trajectories and evaluate on held-out ones."""
    rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8")]
    print(f"Loaded {len(rows)} SWE trajectories from {data_path}")
    print(f"Loading model from: {model_dir}")
    model, tokenizer = mlx_lm.load(str(model_dir))

    num_layers = len(model.layers)
    valid_depths = tuple(sorted({d for d in depths if 1 <= d <= num_layers} | {num_layers}))
    shallow_depths = valid_depths[:-1]

    prepared = []
    for row in rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is not None:
            prepared.append(item)

    train_items = prepared[:n_train]
    test_items = prepared[n_train:]
    print(f"Prepared trajectories: train={len(train_items)}, held_out_test={len(test_items)} (max_tokens={max_tokens})")

    # 1. Extract hidden states on training split to fit W_k: Z_k -> Z_32.
    train_src: dict[int, list[np.ndarray]] = defaultdict(list)
    train_tgt: list[np.ndarray] = []

    for token_ids, _tags, _asst_mask in train_items:
        input_ids = token_ids[:-1]
        normed_by_depth, _ = collect_normed_hidden_and_states(model, input_ids, valid_depths)
        train_tgt.append(normed_by_depth[num_layers])
        for d in shallow_depths:
            train_src[d].append(normed_by_depth[d])

    target_mat = np.concatenate(train_tgt, axis=0)
    print(f"Fitting linear rotations on {target_mat.shape[0]} tokens (dim={target_mat.shape[1]}, ridge={ridge_lambda})...")
    rotations: dict[int, LinearRotationMap] = {}
    for d in shallow_depths:
        src_mat = np.concatenate(train_src[d], axis=0)
        rotations[d] = fit_linear_rotation(src_mat, target_mat, ridge_lambda=ridge_lambda)

    del train_src, train_tgt, target_mat
    gc.collect()

    # 2. Stream held-out evaluation to keep memory bounded on 16GB Apple Silicon.
    metrics_no_rot: dict[tuple[str, int], RunningMetrics] = defaultdict(RunningMetrics)
    metrics_rot: dict[tuple[str, int], RunningMetrics] = defaultdict(RunningMetrics)
    spec_records: dict[int, dict[str, list]] = {
        d: {"raw_top1": [], "rot_top1": [], "target_top1": [], "raw_probs": [], "rot_probs": []}
        for d in shallow_depths
    }
    svd_stats: list[SinkStats] = []

    for traj_idx, (token_ids, tags, asst_mask) in enumerate(test_items):
        input_ids = token_ids[:-1]
        target_tags = tags[1:]
        target_asst = asst_mask[1:]
        asst_positions = [i for i, flag in enumerate(target_asst) if flag]
        if not asst_positions:
            continue

        normed_by_depth, linear_states = collect_normed_hidden_and_states(model, input_ids, valid_depths)
        if traj_idx == 0:
            for _layer_idx, head_states in linear_states:
                svd_stats.extend(compute_state_sink_stats(h) for h in head_states)

        full_h = mx.array(normed_by_depth[num_layers][asst_positions], dtype=mx.float16)
        full_logits_mx = _apply_lm_head(model, full_h)
        mx.eval(full_logits_mx)
        full_logits = np.array(full_logits_mx.astype(mx.float32), copy=True)
        full_top1 = np.argmax(full_logits, axis=-1)
        del full_h, full_logits_mx

        pos_to_local = {pos: j for j, pos in enumerate(asst_positions)}
        tag_local_idxs: dict[str, list[int]] = defaultdict(list)
        for pos in asst_positions:
            tag_local_idxs[target_tags[pos]].append(pos_to_local[pos])

        for d in shallow_depths:
            raw_h_np = normed_by_depth[d][asst_positions]
            rot_h_np = apply_linear_rotation(rotations[d], raw_h_np)

            raw_logits_mx = _apply_lm_head(model, mx.array(raw_h_np, dtype=mx.float16))
            rot_logits_mx = _apply_lm_head(model, mx.array(rot_h_np, dtype=mx.float16))
            mx.eval(raw_logits_mx, rot_logits_mx)
            raw_logits = np.array(raw_logits_mx.astype(mx.float32), copy=True)
            rot_logits = np.array(rot_logits_mx.astype(mx.float32), copy=True)
            del raw_logits_mx, rot_logits_mx

            spec_records[d]["raw_top1"].extend(np.argmax(raw_logits, axis=-1).tolist())
            spec_records[d]["rot_top1"].extend(np.argmax(rot_logits, axis=-1).tolist())
            spec_records[d]["target_top1"].extend(full_top1.tolist())

            raw_p = np.exp(raw_logits - np.max(raw_logits, axis=-1, keepdims=True))
            raw_p /= np.sum(raw_p, axis=-1, keepdims=True)
            rot_p = np.exp(rot_logits - np.max(rot_logits, axis=-1, keepdims=True))
            rot_p /= np.sum(rot_p, axis=-1, keepdims=True)
            spec_records[d]["raw_probs"].extend(np.max(raw_p, axis=-1).tolist())
            spec_records[d]["rot_probs"].extend(np.max(rot_p, axis=-1).tolist())

            for tag, idxs in tag_local_idxs.items():
                metrics_no_rot[(tag, d)] = metrics_no_rot[(tag, d)].add(
                    compute_layer_exit_metrics(d, raw_logits[idxs], full_logits[idxs])
                )
                metrics_rot[(tag, d)] = metrics_rot[(tag, d)].add(
                    compute_layer_exit_metrics(d, rot_logits[idxs], full_logits[idxs])
                )

            del raw_logits, rot_logits, raw_p, rot_p
            mx.clear_cache()

        del full_logits, normed_by_depth
        gc.collect()
        mx.clear_cache()

    print("\n=== Held-Out SWE-smith Assistant Token Accuracy: No Rotation vs Linear Rotation ===")
    for tag in ("tool_boilerplate", "repeated_echo", "new_code_arg", "reasoning"):
        total_n = metrics_no_rot[(tag, shallow_depths[0])].count
        if total_n == 0:
            continue
        print(f"\n[{tag}] (held-out assistant tokens n={total_n})")
        for d in shallow_depths:
            m_no = metrics_no_rot[(tag, d)].finalize(d)
            m_rot = metrics_rot[(tag, d)].finalize(d)
            print(
                f"  layer {d:2d}/{num_layers} ({100*d/num_layers:4.0f}%): "
                f"NO_ROT top1={m_no.top1_match_rate:.3f} kl={m_no.mean_kl_nats:6.2f} | "
                f"ROTATED top1={m_rot.top1_match_rate:.3f} top5={m_rot.top5_match_rate:.3f} kl={m_rot.mean_kl_nats:5.2f}"
            )

    print("\n=== Simulated Greedy Speculative Acceptance (All Held-Out Assistant Tokens, gamma=6) ===")
    for d in shallow_depths:
        rec = spec_records[d]
        s_no = simulate_greedy_speculative(rec["raw_top1"], rec["target_top1"], rec["raw_probs"], max_draft_len=6)
        s_rot = simulate_greedy_speculative(rec["rot_top1"], rec["target_top1"], rec["rot_probs"], max_draft_len=6)
        s_rot_conf = simulate_greedy_speculative(
            rec["rot_top1"], rec["target_top1"], rec["rot_probs"], max_draft_len=6, confidence_threshold=0.70
        )
        print(
            f"  layer {d:2d}/{num_layers}: "
            f"no_rot tau={s_no.mean_accepted_tokens:.2f} | "
            f"rotated tau={s_rot.mean_accepted_tokens:.2f} (acc={s_rot.acceptance_rate:.3f}) | "
            f"rotated+conf>=0.7 tau={s_rot_conf.mean_accepted_tokens:.2f} (acc={s_rot_conf.acceptance_rate:.3f})"
        )

    if svd_stats:
        overall = summarize_sink_stats(svd_stats)
        print(
            f"\n=== State Sink SVD on real SWE trajectory ({len(svd_stats)} heads, max_tokens={max_tokens}) ===\n"
            f"  median_top1={overall['median_top1']:.3f}, pooled_top1={overall['pooled_top1']:.3f}, "
            f"median_rank90={overall['median_rank90']:.1f}, median_rank99={overall['median_rank99']:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWE-smith rotation probe on Qwen3.5-4B.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28, 32])
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--ridge-lambda", type=float, default=50.0)
    args = parser.parse_args()
    run_swe_rotation_experiment(
        model_dir=args.model_dir,
        data_path=args.data_path,
        depths=args.depths,
        n_train=args.n_train,
        max_tokens=args.max_tokens,
        ridge_lambda=args.ridge_lambda,
    )


if __name__ == "__main__":
    main()
