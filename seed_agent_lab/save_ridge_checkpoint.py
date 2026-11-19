"""Fit and save Assistant-only closed-form Ridge rotation W_k for fast closed-loop loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np

from seed_agent_lab.draft_adapter import (
    init_ridge_accumulator,
    solve_ridge_accumulator,
    update_ridge_accumulator,
)
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    _prepare_trajectory,
    collect_normed_hidden_and_states,
)
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR, EXTRA_TRAIN_PATH


def save_ridge_asst(
    model_dir: Path,
    base_data: Path,
    extra_train: Path,
    draft_layer: int,
    max_tokens: int,
    ridge_lambda: float,
    out_path: Path,
) -> None:
    """Solve Assistant-only Ridge W_k across SWE training trajectories and save as .npz."""
    base_rows = [json.loads(line) for line in base_data.open("r", encoding="utf-8")]
    extra_rows = [json.loads(line) for line in extra_train.open("r", encoding="utf-8")] if extra_train.exists() else []
    train_rows = base_rows[:12] + extra_rows

    model, tokenizer = mlx_lm.load(str(model_dir))
    num_layers = len(model.layers)
    hidden_dim = model.language_model.args.hidden_size
    depths = (draft_layer, num_layers)

    acc_asst = init_ridge_accumulator(hidden_dim)
    n_used = 0
    for row in train_rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is None:
            continue
        token_ids, _tags, asst_mask = item
        asst_pos = [i for i, flag in enumerate(asst_mask[1:]) if flag]
        if not asst_pos:
            continue
        normed, _ = collect_normed_hidden_and_states(model, token_ids[:-1], depths)
        acc_asst = update_ridge_accumulator(acc_asst, normed[draft_layer][asst_pos], normed[num_layers][asst_pos])
        n_used += 1
        del normed
        mx.clear_cache()

    rot = solve_ridge_accumulator(acc_asst, ridge_lambda=ridge_lambda)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        weight=rot.weight_with_bias[:-1, :].astype(np.float32),
        bias=rot.weight_with_bias[-1:, :].astype(np.float32),
        draft_layer=np.array([draft_layer]),
        n_trajectories=np.array([n_used]),
        n_tokens=np.array([acc_asst.count]),
    )
    print(f"Saved Ridge_AssistantOnly (W_{draft_layer}) from {n_used} traces ({acc_asst.count} tokens) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--ridge-lambda", type=float, default=50.0)
    args = parser.parse_args()
    out = CHECKPOINT_DIR / f"layer{args.draft_layer}_ridge_asst.npz"
    save_ridge_asst(
        model_dir=DEFAULT_MODEL_DIR,
        base_data=DEFAULT_DATA_PATH,
        extra_train=EXTRA_TRAIN_PATH,
        draft_layer=args.draft_layer,
        max_tokens=args.max_tokens,
        ridge_lambda=args.ridge_lambda,
        out_path=out,
    )


if __name__ == "__main__":
    main()
