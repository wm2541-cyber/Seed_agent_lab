"""Train KL-distilled Layer-8 Draft Adapters on SWE-smith and benchmark against Ridge W_8."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten

from seed_agent_lab.analysis import compute_layer_exit_metrics
from seed_agent_lab.draft_adapter import (
    LinearDraftAdapter,
    SwiGLUDraftAdapter,
    init_ridge_accumulator,
    solve_ridge_accumulator,
    update_ridge_accumulator,
)
from seed_agent_lab.hybrid_speculative import HybridSpecStats, simulate_hybrid_policy
from seed_agent_lab.rotation import apply_linear_rotation
from seed_agent_lab.run_hybrid_eval import _aggregate_policy_stats
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    RunningMetrics,
    _apply_lm_head,
    _prepare_trajectory,
    collect_normed_hidden_and_states,
)

EXTRA_TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "swe_smith_extra_train.jsonl"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _kl_train_adapter(
    adapter: nn.Module,
    h8_train: np.ndarray,
    h32_train: np.ndarray,
    lm_head_w_f16: mx.array,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> nn.Module:
    """Distill Layer-8 adapter toward Layer-32 full-vocabulary distribution using AdamW."""
    lm_head_t = lm_head_w_f16.T  # (D, V)
    n_samples = h8_train.shape[0]
    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = epochs * steps_per_epoch

    schedule = optim.cosine_decay(lr, total_steps, end=lr * 0.05)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=weight_decay)

    def loss_fn(model_adapter: nn.Module, x_b: mx.array, y_b: mx.array) -> mx.array:
        t_logits = (y_b.astype(mx.float16) @ lm_head_t).astype(mx.float32)
        t_logp = mx.stop_gradient(t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True))
        pred_h = model_adapter(x_b).astype(mx.float16)
        s_logits = (pred_h @ lm_head_t).astype(mx.float32)
        s_logp = s_logits - mx.logsumexp(s_logits, axis=-1, keepdims=True)
        return mx.mean(mx.sum(mx.exp(t_logp) * (t_logp - s_logp), axis=-1))

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)
    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            b_idx = perm[step * batch_size : (step + 1) * batch_size]
            x_b = mx.array(h8_train[b_idx], dtype=mx.float32)
            y_b = mx.array(h32_train[b_idx], dtype=mx.float32)
            loss, grads = loss_and_grad(adapter, x_b, y_b)
            optimizer.update(adapter, grads)
            mx.eval(adapter.parameters(), optimizer.state, loss)
            epoch_loss += float(loss.item())
        print(f"    epoch {epoch + 1}/{epochs}: mean KL={epoch_loss / steps_per_epoch:.4f} nats ({time.perf_counter() - t0:.1f}s)")
    mx.clear_cache()
    return adapter


def _evaluate_predictor(
    name: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    model,
    cached_test_data: list[tuple[list[int], list[int], tuple[str, ...], np.ndarray, np.ndarray]],
    draft_layer: int,
    num_layers: int,
) -> None:
    """Evaluate one predictor on the exact same frozen 6 held-out SWE trajectories."""
    depth_ratio = draft_layer / num_layers
    rot_metrics: dict[str, RunningMetrics] = defaultdict(RunningMetrics)
    policy_runs: dict[str, list[HybridSpecStats]] = defaultdict(list)

    for token_ids, asst_positions, target_tags, h8_asst, full_logits in cached_test_data:
        full_top1 = np.argmax(full_logits, axis=-1).tolist()
        pred_h = predict_fn(h8_asst)
        pred_logits_mx = _apply_lm_head(model, mx.array(pred_h, dtype=mx.float16))
        mx.eval(pred_logits_mx)
        pred_logits = np.array(pred_logits_mx.astype(mx.float32), copy=True)
        del pred_logits_mx

        pred_top1 = np.argmax(pred_logits, axis=-1).tolist()
        pred_p = np.exp(pred_logits - np.max(pred_logits, axis=-1, keepdims=True))
        pred_p /= np.sum(pred_p, axis=-1, keepdims=True)
        pred_max_probs = np.max(pred_p, axis=-1).tolist()
        del pred_p

        pos_to_local = {pos: j for j, pos in enumerate(asst_positions)}
        tag_local_idxs: dict[str, list[int]] = defaultdict(list)
        for pos in asst_positions:
            tag_local_idxs[target_tags[pos]].append(pos_to_local[pos])

        for tag, idxs in tag_local_idxs.items():
            rot_metrics[tag] = rot_metrics[tag].add(
                compute_layer_exit_metrics(draft_layer, pred_logits[idxs], full_logits[idxs])
            )
        del pred_logits

        for pname, use_lookup, conf in (
            ("L8_Conf>=0.70", False, 0.70),
            ("Hybrid_LookupFirst+L8(c>=0.70)", True, 0.70),
            ("Hybrid_LookupFirst+L8(c>=0.60)", True, 0.60),
        ):
            st = simulate_hybrid_policy(
                policy_name=pname,
                full_sequence_tokens=token_ids,
                assistant_positions=asst_positions,
                shallow_top1_tokens=pred_top1,
                target_top1_tokens=full_top1,
                shallow_max_probs=pred_max_probs,
                shallow_depth_ratio=depth_ratio,
                use_neural=True,
                use_lookup=use_lookup,
                lookup_first=True,
                max_neural_len=6,
                max_lookup_len=8,
                confidence_threshold=conf,
            )
            policy_runs[pname].append(st)

    print(f"\n=== [{name}] Held-Out Layer-{draft_layer} Accuracy ===")
    for tag in ("tool_boilerplate", "repeated_echo", "new_code_arg", "reasoning"):
        m = rot_metrics[tag].finalize(draft_layer)
        if m.count:
            print(f"  {tag:17s} (n={m.count:4d}): top1={m.top1_match_rate:.3f}, top5={m.top5_match_rate:.3f}, kl={m.mean_kl_nats:.2f}")
    for pname, runs in policy_runs.items():
        agg = _aggregate_policy_stats(runs)
        print(
            f"  -> {agg.policy_name:32s} | tau={agg.mean_accepted_tokens:4.2f} | "
            f"acc={agg.acceptance_rate:.3f} | L{draft_layer}={agg.avg_neural_steps:4.2f} | "
            f"lookup={agg.avg_lookup_steps:4.2f} | cost={agg.avg_weight_read_cost:4.2f} | "
            f"speedup={agg.theoretical_speedup:4.2f}x"
        )


def run_train_and_eval(
    model_dir: Path,
    base_data_path: Path,
    extra_train_path: Path,
    draft_layer: int,
    max_tokens: int,
    ridge_lambda: float,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    """Compare Ridge-All, Ridge-Asst, KL-Linear, and KL-SwiGLU(512) on held-out SWE traces."""
    base_rows = [json.loads(line) for line in base_data_path.open("r", encoding="utf-8")]
    extra_rows = [json.loads(line) for line in extra_train_path.open("r", encoding="utf-8")] if extra_train_path.exists() else []
    train_rows = base_rows[:12] + extra_rows
    test_rows = base_rows[12:18]  # Exact same 6 held-out test trajectories!

    model, tokenizer = mlx_lm.load(str(model_dir))
    num_layers = len(model.layers)
    hidden_dim = model.language_model.args.hidden_size
    depths = (draft_layer, num_layers)

    # 1. Extract streaming Ridge covariances + assistant hidden pairs across all training traces.
    acc_all = init_ridge_accumulator(hidden_dim)
    acc_asst = init_ridge_accumulator(hidden_dim)
    asst_h8_list: list[np.ndarray] = []
    asst_h32_list: list[np.ndarray] = []
    n_train_used = 0

    for row in train_rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is None:
            continue
        token_ids, _tags, asst_mask = item
        asst_pos = [i for i, flag in enumerate(asst_mask[1:]) if flag]
        normed, _ = collect_normed_hidden_and_states(model, token_ids[:-1], depths)
        h8 = normed[draft_layer]
        h32 = normed[num_layers]
        acc_all = update_ridge_accumulator(acc_all, h8, h32)
        if asst_pos:
            h8_a = h8[asst_pos]
            h32_a = h32[asst_pos]
            acc_asst = update_ridge_accumulator(acc_asst, h8_a, h32_a)
            asst_h8_list.append(h8_a.astype(np.float16))
            asst_h32_list.append(h32_a.astype(np.float16))
        n_train_used += 1
        del normed
        mx.clear_cache()

    h8_asst_train = np.concatenate(asst_h8_list, axis=0)
    h32_asst_train = np.concatenate(asst_h32_list, axis=0)
    del asst_h8_list, asst_h32_list
    gc.collect()

    print(
        f"Training corpus ready: {n_train_used} SWE trajectories | "
        f"all_tokens={acc_all.count} | assistant_tokens={h8_asst_train.shape[0]}"
    )

    ridge_all = solve_ridge_accumulator(acc_all, ridge_lambda=ridge_lambda)
    ridge_asst = solve_ridge_accumulator(acc_asst, ridge_lambda=ridge_lambda)

    # 2. Pre-extract held-out test assistant data once so all 4 variants evaluate identically & fast.
    cached_test_data = []
    for row in test_rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is None:
            continue
        token_ids, tags, asst_mask = item
        target_tags = tags[1:]
        asst_pos = [i for i, flag in enumerate(asst_mask[1:]) if flag]
        if not asst_pos:
            continue
        normed, _ = collect_normed_hidden_and_states(model, token_ids[:-1], depths)
        full_logits_mx = _apply_lm_head(model, mx.array(normed[num_layers][asst_pos], dtype=mx.float16))
        mx.eval(full_logits_mx)
        full_logits = np.array(full_logits_mx.astype(mx.float32), copy=True)
        h8_asst = normed[draft_layer][asst_pos]
        cached_test_data.append((token_ids, asst_pos, target_tags, h8_asst, full_logits))
        del normed, full_logits_mx
        mx.clear_cache()

    # Evaluate closed-form baselines first.
    _evaluate_predictor("1_Ridge_AllTokens", lambda x: apply_linear_rotation(ridge_all, x), model, cached_test_data, draft_layer, num_layers)
    _evaluate_predictor("2_Ridge_AssistantOnly", lambda x: apply_linear_rotation(ridge_asst, x), model, cached_test_data, draft_layer, num_layers)

    # 3. Dequantize frozen LM head once for fast KL distillation.
    emb = model.language_model.model.embed_tokens
    lm_head_w_f16 = mx.dequantize(
        emb.weight, emb.scales, emb.biases, group_size=emb.group_size, bits=emb.bits
    ).astype(mx.float16)
    mx.eval(lm_head_w_f16)

    print("\nTraining 3_KL_Linear_W8 (0 extra inference params)...")
    kl_lin = LinearDraftAdapter(hidden_dim, init_map=ridge_asst)
    _kl_train_adapter(kl_lin, h8_asst_train, h32_asst_train, lm_head_w_f16, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=1e-4)
    _evaluate_predictor(
        "3_KL_Linear_W8",
        lambda x: np.array(kl_lin(mx.array(x, dtype=mx.float32)), copy=True),
        model,
        cached_test_data,
        draft_layer,
        num_layers,
    )

    print("\nTraining 4_KL_SwiGLU_Adapter (bottleneck=512, ~3.9M params)...")
    kl_swi = SwiGLUDraftAdapter(hidden_dim, bottleneck_dim=512, init_map=ridge_asst)
    _kl_train_adapter(kl_swi, h8_asst_train, h32_asst_train, lm_head_w_f16, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=1e-3)
    _evaluate_predictor(
        "4_KL_SwiGLU_Adapter(r=512)",
        lambda x: np.array(kl_swi(mx.array(x, dtype=mx.float32)), copy=True),
        model,
        cached_test_data,
        draft_layer,
        num_layers,
    )

    # Save trained SwiGLU adapter weights for upcoming end-to-end closed-loop benchmark!
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"layer{draft_layer}_swiglu_adapter.safetensors"
    flat_weights = {k: v.astype(mx.float16) for k, v in tree_flatten(kl_swi.parameters())}
    mx.save_safetensors(str(ckpt_path), flat_weights)
    print(f"\nSaved trained adapter to: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate KL-distilled Layer-8 Draft Adapters.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--extra-train", type=Path, default=EXTRA_TRAIN_PATH)
    parser.add_argument("--draft-layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--ridge-lambda", type=float, default=50.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    run_train_and_eval(
        model_dir=args.model_dir,
        base_data_path=args.base_data,
        extra_train_path=args.extra_train,
        draft_layer=args.draft_layer,
        max_tokens=args.max_tokens,
        ridge_lambda=args.ridge_lambda,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
