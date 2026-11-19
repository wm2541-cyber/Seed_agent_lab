"""Evaluate SEED State Sink (b + ax) compression and Adaptive Shallow Prefill on Qwen3.5-4B + SWE-smith."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from seed_agent_lab.analysis import compute_layer_exit_metrics
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    RunningMetrics,
    _apply_lm_head,
    _prepare_trajectory,
)
from seed_agent_lab.seed_ops import (
    attention_write_only,
    compress_matrix_svd,
    compress_seed_state,
    decompose_state_sink,
    gdn_extract_v_and_write,
)


def _clone_cache(cache: Sequence) -> list:
    """Create a deep, independent copy of an MLX prompt cache list."""
    cloned = []
    for c in cache:
        if isinstance(c, KVCache):
            nc = KVCache()
            if c.keys is not None and c.offset > 0:
                nc.keys = mx.array(c.keys[..., : c.offset, :])
                nc.values = mx.array(c.values[..., : c.offset, :])
                nc.offset = c.offset
            cloned.append(nc)
        elif isinstance(c, ArraysCache):
            nc = ArraysCache(len(c.cache))
            nc.cache = [mx.array(arr) if arr is not None else None for arr in c.cache]
            cloned.append(nc)
        else:
            raise TypeError(f"Unsupported cache type: {type(c)}")
    mx.eval([x for c in cloned for x in (c.cache if isinstance(c, ArraysCache) else (c.keys, c.values)) if x is not None])
    return cloned


def estimate_mean_values_and_sinks(
    model,
    train_token_seqs: Sequence[Sequence[int]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Estimate per-head mean value vector v_bar and mean sink coeff a_bar on training SWE traces."""
    text_model = model.language_model.model
    v_sums: dict[int, np.ndarray] = {}
    v_counts: dict[int, int] = defaultdict(int)
    a_sums: dict[int, np.ndarray] = {}
    n_seqs = 0

    for seq in train_token_seqs:
        cache = model.make_cache()
        hidden = text_model.embed_tokens(mx.array([list(seq)], dtype=mx.int32))
        fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])

        for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            if layer.is_linear:
                # Extract post-conv values v while running exact layer update via exact path + v probe.
                v_arr = gdn_extract_v_and_write(layer, hidden, layer_cache)
                mx.eval(v_arr)
                v_np = np.array(v_arr[0].astype(mx.float32), copy=True)  # (S, Hv, Dv)
                v_sum = np.sum(v_np, axis=0)  # (Hv, Dv)
                v_sums[idx] = v_sum if idx not in v_sums else (v_sums[idx] + v_sum)
                v_counts[idx] += v_np.shape[0]
                # Re-run full layer into a dummy throwaway cache? No—restore layer_cache and run layer normally!
                # Reset layer_cache[0], [1] to None before calling full layer so hidden advances identically.
                layer_cache.cache = [None, None]
                hidden = layer(hidden, mask=ssm_mask, cache=layer_cache)
            else:
                hidden = layer(hidden, mask=fa_mask, cache=layer_cache)

        mx.eval(hidden)
        n_seqs += 1
        del cache, hidden, fa_mask, ssm_mask
        mx.clear_cache()

    mean_v: dict[int, np.ndarray] = {idx: v_sums[idx] / v_counts[idx] for idx in v_sums}

    # Second quick pass to compute mean sink coefficients a_bar across training traces.
    for seq in train_token_seqs:
        cache = model.make_cache()
        _ = model(mx.array([list(seq)], dtype=mx.int32), cache=cache)
        for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
            if layer.is_linear and layer_cache[1] is not None:
                mx.eval(layer_cache[1])
                s_np = np.array(layer_cache[1][0].astype(mx.float32), copy=True)  # (Hv, Dv, Dk)
                a_heads = np.stack(
                    [decompose_state_sink(s_np[h], mean_v[idx][h]).sink_coeff for h in range(s_np.shape[0])],
                    axis=0,
                )
                a_sums[idx] = a_heads if idx not in a_sums else (a_sums[idx] + a_heads)
        del cache
        mx.clear_cache()

    mean_a: dict[int, np.ndarray] = {idx: a_sums[idx] / n_seqs for idx in a_sums}
    return mean_v, mean_a


def _run_chunks_with_shallow_prefill(
    model,
    chunks_and_depths: Sequence[tuple[Sequence[int], int]],
) -> list:
    """Prefill sequential prompt chunks where each chunk runs full layers up to anchor k, then write-only."""
    text_model = model.language_model.model
    cache = model.make_cache()
    num_layers = len(model.layers)

    for chunk_ids, anchor_k in chunks_and_depths:
        if not chunk_ids:
            continue
        hidden = text_model.embed_tokens(mx.array([list(chunk_ids)], dtype=mx.int32))
        fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])
        k_clamped = max(1, min(anchor_k, num_layers))

        for idx in range(k_clamped):
            layer = model.layers[idx]
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=cache[idx])

        for idx in range(k_clamped, num_layers):
            layer = model.layers[idx]
            if layer.is_linear:
                _ = gdn_extract_v_and_write(layer, hidden, cache[idx])
            else:
                attention_write_only(layer, hidden, cache[idx])

        mx.eval([x for c in cache for x in (c.cache if isinstance(c, ArraysCache) else (c.keys, c.values)) if x is not None])
        del hidden, fa_mask, ssm_mask
        mx.clear_cache()

    return cache


def _score_continuation(model, cache: Sequence, cont_ids: Sequence[int]) -> np.ndarray:
    """Run continuation tokens through all 32 layers using provided prompt cache and return logits."""
    logits_mx = model(mx.array([list(cont_ids)], dtype=mx.int32), cache=cache)[0]
    mx.eval(logits_mx)
    out = np.array(logits_mx.astype(mx.float32), copy=True)
    del logits_mx
    mx.clear_cache()
    return out


def _intervene_gdn_states(
    base_cache: Sequence,
    model,
    head_transform: Callable[[int, int, np.ndarray], np.ndarray],
) -> list:
    """Clone cache and transform every GatedDeltaNet head matrix S (Dv, Dk) cleanly."""
    cloned = _clone_cache(base_cache)
    for idx, (layer, layer_cache) in enumerate(zip(model.layers, cloned)):
        if layer.is_linear and layer_cache[1] is not None:
            orig_dtype = layer_cache[1].dtype
            s_np = np.array(layer_cache[1][0].astype(mx.float32), copy=True)
            new_heads = np.stack(
                [head_transform(idx, h, s_np[h]) for h in range(s_np.shape[0])],
                axis=0,
            )
            layer_cache[1] = mx.array(new_heads[None, ...], dtype=orig_dtype)
    mx.eval([c[1] for l, c in zip(model.layers, cloned) if l.is_linear and c[1] is not None])
    return cloned


def _find_last_assistant_span(asst_mask: Sequence[bool], min_prompt_len: int = 256) -> tuple[int, int] | None:
    """Return (start, end) of the last contiguous assistant span after min_prompt_len."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(asst_mask)
    while i < n:
        if asst_mask[i]:
            j = i
            while j < n and asst_mask[j]:
                j += 1
            if i >= min_prompt_len and (j - i) >= 16:
                spans.append((i, j))
            i = j
        else:
            i += 1
    return spans[-1] if spans else None


def run_seed_coding_eval(
    model_dir: Path,
    data_path: Path,
    n_train: int,
    n_test: int,
    max_tokens: int,
) -> None:
    """Run SEED State Sink + Shallow Prefill evaluations on SWE-smith Coding Agent continuations."""
    rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8")]
    model, tokenizer = mlx_lm.load(str(model_dir))
    num_layers = len(model.layers)

    prepared = []
    for row in rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is not None:
            prepared.append(item)

    train_seqs = [item[0] for item in prepared[:n_train]]
    test_items = prepared[n_train : n_train + n_test]
    print(f"Estimating mean values v_bar & sinks across {len(train_seqs)} SWE traces (up to {max_tokens} tokens)...")
    mean_v, mean_a = estimate_mean_values_and_sinks(model, train_seqs)

    # Track Decode state compression metrics & Prefill shallow-write metrics.
    state_metrics: dict[str, dict[str, RunningMetrics]] = defaultdict(lambda: defaultdict(RunningMetrics))
    prefill_metrics: dict[str, dict[str, RunningMetrics]] = defaultdict(lambda: defaultdict(RunningMetrics))

    for token_ids, tags, asst_mask in test_items:
        span = _find_last_assistant_span(asst_mask, min_prompt_len=256)
        if span is None:
            continue
        split_pt, span_end = span
        # Prompt ends before split_pt - 1; step 0 of continuation feeds token_ids[split_pt - 1]
        # to predict the first assistant token at split_pt.
        prompt_prefix = token_ids[: split_pt - 1]
        cont_inputs = token_ids[split_pt - 1 : span_end - 1]
        cont_tags = tags[split_pt:span_end]
        cont_asst = asst_mask[split_pt:span_end]
        if not cont_inputs:
            continue

        # Exact baseline: full 32-layer prefill on prompt_prefix, then continuation.
        exact_prompt_cache = _run_chunks_with_shallow_prefill(model, [(prompt_prefix, num_layers)])
        exact_cont_cache = _clone_cache(exact_prompt_cache)
        exact_logits = _score_continuation(model, exact_cont_cache, cont_inputs)
        del exact_cont_cache

        tag_buckets: dict[str, list[int]] = defaultdict(list)
        for pos, (tag, is_asst) in enumerate(zip(cont_tags, cont_asst)):
            if is_asst:
                tag_buckets["all_asst"].append(pos)
                if tag in ("tool_boilerplate", "repeated_echo", "new_code_arg", "reasoning"):
                    tag_buckets[tag].append(pos)

        # --- Experiment A: SEED State Sink (b + ax) Interventions ---
        state_policies: list[tuple[str, Callable[[int, int, np.ndarray], np.ndarray]]] = [
            ("Keep_Sink_Only (drop ax)", lambda l, h, s: np.outer(decompose_state_sink(s, mean_v[l][h]).unit_v_bar, decompose_state_sink(s, mean_v[l][h]).sink_coeff)),
            ("Remove_Sink (drop b)", lambda l, h, s: decompose_state_sink(s, mean_v[l][h]).content),
            ("Replace_Sink_Mean (swap b)", lambda l, h, s: np.outer(decompose_state_sink(s, mean_v[l][h]).unit_v_bar, mean_a[l][h]) + decompose_state_sink(s, mean_v[l][h]).content),
            ("Plain_SVD_R8", lambda l, h, s: compress_matrix_svd(s, rank=8)),
            ("SEED_b_ax_R8", lambda l, h, s: compress_seed_state(s, mean_v[l][h], rank=8)),
            ("Plain_SVD_R16", lambda l, h, s: compress_matrix_svd(s, rank=16)),
            ("SEED_b_ax_R16", lambda l, h, s: compress_seed_state(s, mean_v[l][h], rank=16)),
        ]

        for name, fn in state_policies:
            mod_cache = _intervene_gdn_states(exact_prompt_cache, model, fn)
            mod_logits = _score_continuation(model, mod_cache, cont_inputs)
            for bname, idxs in tag_buckets.items():
                if idxs:
                    state_metrics[name][bname] = state_metrics[name][bname].add(
                        compute_layer_exit_metrics(0, mod_logits[idxs], exact_logits[idxs])
                    )
            del mod_cache, mod_logits
            mx.clear_cache()

        del exact_prompt_cache
        mx.clear_cache()

        # --- Experiment B: Uniform vs Adaptive Shallow Prefill (Untrained Write Paths) ---
        cut_75 = int(len(prompt_prefix) * 0.75)
        prefill_configs: list[tuple[str, list[tuple[Sequence[int], int]]]] = [
            ("Uniform_k28 (88%)", [(prompt_prefix, 28)]),
            ("Uniform_k24 (75%)", [(prompt_prefix, 24)]),
            ("Uniform_k20 (62%)", [(prompt_prefix, 20)]),
            ("Uniform_k16 (50%)", [(prompt_prefix, 16)]),
            ("Adaptive_old16_recent24 (50%->75%)", [(prompt_prefix[:cut_75], 16), (prompt_prefix[cut_75:], 24)]),
        ]

        for pname, chunks in prefill_configs:
            p_cache = _run_chunks_with_shallow_prefill(model, chunks)
            p_logits = _score_continuation(model, p_cache, cont_inputs)
            for bname, idxs in tag_buckets.items():
                if idxs:
                    prefill_metrics[pname][bname] = prefill_metrics[pname][bname].add(
                        compute_layer_exit_metrics(0, p_logits[idxs], exact_logits[idxs])
                    )
            del p_cache, p_logits
            mx.clear_cache()

        del exact_logits
        gc.collect()
        mx.clear_cache()

    print("\n=== Experiment A: SEED b + ax State Interventions on Held-Out SWE Continuations ===")
    for name in (
        "Keep_Sink_Only (drop ax)",
        "Remove_Sink (drop b)",
        "Replace_Sink_Mean (swap b)",
        "Plain_SVD_R8",
        "SEED_b_ax_R8",
        "Plain_SVD_R16",
        "SEED_b_ax_R16",
    ):
        all_m = state_metrics[name]["all_asst"].finalize(0)
        tb_m = state_metrics[name]["tool_boilerplate"].finalize(0)
        nc_m = state_metrics[name]["new_code_arg"].finalize(0)
        print(
            f"  {name:28s} | all_asst(n={all_m.count:3d}): kl={all_m.mean_kl_nats:6.4f} top1={all_m.top1_match_rate:.3f} | "
            f"tool_bp(n={tb_m.count:3d}): kl={tb_m.mean_kl_nats:6.4f} top1={tb_m.top1_match_rate:.3f} | "
            f"code(n={nc_m.count:3d}): top1={nc_m.top1_match_rate:.3f}"
        )

    print("\n=== Experiment B: Untrained Shallow Prefill (Uniform vs Adaptive Old->Recent) ===")
    for pname in (
        "Uniform_k28 (88%)",
        "Uniform_k24 (75%)",
        "Uniform_k20 (62%)",
        "Uniform_k16 (50%)",
        "Adaptive_old16_recent24 (50%->75%)",
    ):
        all_m = prefill_metrics[pname]["all_asst"].finalize(0)
        tb_m = prefill_metrics[pname]["tool_boilerplate"].finalize(0)
        nc_m = prefill_metrics[pname]["new_code_arg"].finalize(0)
        print(
            f"  {pname:36s} | all_asst(n={all_m.count:3d}): kl={all_m.mean_kl_nats:6.4f} top1={all_m.top1_match_rate:.3f} | "
            f"tool_bp(n={tb_m.count:3d}): top1={tb_m.top1_match_rate:.3f} | "
            f"code(n={nc_m.count:3d}): top1={nc_m.top1_match_rate:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SEED State Sink and Adaptive Shallow Prefill on SWE-smith.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-train", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=3072)
    args = parser.parse_args()
    run_seed_coding_eval(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_train=args.n_train,
        n_test=args.n_test,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
