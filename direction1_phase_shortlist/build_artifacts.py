"""Build Direction 1 static shortlists (4K/8K/16K/32K) and logZ Ridge calibrator strictly from the 48 SWE training traces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx_lm
from mlx_lm.models.base import create_attention_mask
import numpy as np

from direction1_phase_shortlist import ensure_metal_gpu_arch
from direction1_phase_shortlist.logz_calibrator import (
    init_logz_accumulator,
    save_logz_calibrator,
    solve_logz_calibrator,
    update_logz_accumulator,
)
from direction1_phase_shortlist.phase_router import extract_phase_tag_ids
from direction1_phase_shortlist.shortlist_vocab import build_static_shortlist, save_static_shortlist
from seed_agent_lab.run_swe_rotation_probe import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    _prepare_trajectory,
)
from seed_agent_lab.train_and_eval_adapter import CHECKPOINT_DIR, EXTRA_TRAIN_PATH

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_SHORTLIST_SIZES: tuple[int, ...] = (4096, 8192, 16384, 32768)


def _forward_to_layer_normed(model, token_ids: Sequence[int], stop_layer: int) -> mx.array:
    """Run layers 0..stop_layer-1 on a single sequence and return final-RMSNormed hidden states (T, D)."""
    text_model = model.language_model.model
    hidden = text_model.embed_tokens(mx.array([list(token_ids)], dtype=mx.int32))
    fa_mask = create_attention_mask(hidden, None)
    for idx in range(stop_layer):
        layer = model.layers[idx]
        mask = None if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=None)
    return text_model.norm(hidden)[0]  # (T, D)


def _chunked_full_logz(embed_tokens, rotated_bf16: mx.array, chunk_size: int = 128) -> np.ndarray:
    """Compute exact full-vocabulary logsumexp in float32 using bounded-memory chunks."""
    n_tokens = int(rotated_bf16.shape[0])
    chunks: list[np.ndarray] = []
    for start in range(0, n_tokens, chunk_size):
        sub = rotated_bf16[start : start + chunk_size]
        logits_f32 = embed_tokens.as_linear(sub).astype(mx.float32)
        logz = mx.logsumexp(logits_f32, axis=-1)
        mx.eval(logz)
        chunks.append(np.asarray(logz, dtype=np.float64))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float64)


def _collect_pinned_ids(tokenizer) -> tuple[int, ...]:
    """Purely gather structural control & EOS token IDs that must always reside in the static shortlist."""
    tags = extract_phase_tag_ids(tokenizer)
    pinned: list[int] = [
        tid
        for tid in (
            tags.think_start,
            tags.think_end,
            tags.tool_call_start,
            tags.tool_call_end,
            tags.im_end,
        )
        if tid is not None
    ]
    for eos_id in sorted(getattr(tokenizer, "eos_token_ids", set()) or ()):
        pinned.append(int(eos_id))
    for text in ("\n", "\n\n", "<function=", "</function>", "<parameter=", "</parameter>"):
        pinned.extend(int(t) for t in tokenizer.encode(text, add_special_tokens=False))
    # Deduplicate preserving order
    seen: set[int] = set()
    ordered: list[int] = []
    for tid in pinned:
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return tuple(ordered)


def build_direction1_artifacts(
    model_dir: Path = DEFAULT_MODEL_DIR,
    base_data: Path = DEFAULT_DATA_PATH,
    extra_train: Path = EXTRA_TRAIN_PATH,
    w8_checkpoint: Path = CHECKPOINT_DIR / "layer8_ridge_asst.npz",
    out_dir: Path = ARTIFACTS_DIR,
    shortlist_sizes: Sequence[int] = DEFAULT_SHORTLIST_SIZES,
    draft_layer: int = 8,
    max_tokens: int = 3584,
    logz_ridge_lambda: float = 2000.0,
) -> dict[str, Path]:
    """Fit static shortlists and streaming Ridge logZ calibrator strictly on the 48 training traces."""
    ensure_metal_gpu_arch()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = [json.loads(line) for line in base_data.open("r", encoding="utf-8")]
    extra_rows = [json.loads(line) for line in extra_train.open("r", encoding="utf-8")] if extra_train.exists() else []
    train_rows = base_rows[:12] + extra_rows

    w8_npz = np.load(w8_checkpoint)
    w8_weight = mx.array(w8_npz["weight"], dtype=mx.bfloat16)
    w8_bias = mx.array(w8_npz["bias"], dtype=mx.bfloat16)
    mx.eval(w8_weight, w8_bias)

    model, tokenizer = mlx_lm.load(str(model_dir))
    embed_tokens = model.language_model.model.embed_tokens
    vocab_size = int(embed_tokens.weight.shape[0])
    hidden_dim = int(model.language_model.args.hidden_size)
    pinned_ids = _collect_pinned_ids(tokenizer)

    asst_counter: Counter[int] = Counter()
    all_counter: Counter[int] = Counter()
    logz_acc = init_logz_accumulator(hidden_dim)

    n_used = 0
    for row in train_rows:
        item = _prepare_trajectory(tokenizer, row, max_tokens=max_tokens)
        if item is None:
            continue
        token_ids, _tags, asst_mask = item
        all_counter.update(int(t) for t in token_ids)

        asst_pos = [i for i, flag in enumerate(asst_mask[1:]) if flag]
        if not asst_pos:
            continue
        for pos in asst_pos:
            asst_counter[int(token_ids[pos + 1])] += 1

        normed_h8 = _forward_to_layer_normed(model, token_ids[:-1], stop_layer=draft_layer)
        pos_mx = mx.array(asst_pos, dtype=mx.int32)
        h8_asst = normed_h8[pos_mx]
        rotated_bf16 = (h8_asst @ w8_weight) + w8_bias
        mx.eval(h8_asst, rotated_bf16)

        full_logz = _chunked_full_logz(embed_tokens, rotated_bf16, chunk_size=128)
        # Fit logZ on frozen RMSNormed h8 (avoids nested W_8 in-sample vs out-of-sample norm inflation).
        h8_f32_np = np.asarray(h8_asst.astype(mx.float32), dtype=np.float32)
        logz_acc = update_logz_accumulator(logz_acc, h8_f32_np, full_logz)

        n_used += 1
        del normed_h8, h8_asst, rotated_bf16
        mx.clear_cache()

    saved_paths: dict[str, Path] = {}
    for size in shortlist_sizes:
        sl = build_static_shortlist(
            vocab_size=vocab_size,
            target_size=int(size),
            assistant_counts=asst_counter,
            all_token_counts=all_counter,
            pinned_token_ids=pinned_ids,
            metadata={
                "source": "swe_smith_48_train",
                "n_trajectories": str(n_used),
                "unique_asst_tokens": str(len(asst_counter)),
                "unique_all_tokens": str(len(all_counter)),
                "target_size": str(size),
            },
        )
        sl_path = save_static_shortlist(out_dir / f"shortlist_{size}.npz", sl)
        saved_paths[f"shortlist_{size}"] = sl_path
        print(
            f"Saved static shortlist K={size:5d} "
            f"(unique_asst={len(asst_counter)}, unique_all={len(all_counter)}) -> {sl_path}"
        )

    calibrator = solve_logz_calibrator(logz_acc, ridge_lambda=logz_ridge_lambda)
    cal_path = save_logz_calibrator(out_dir / f"layer{draft_layer}_logz_calibrator.npz", calibrator)
    saved_paths["logz_calibrator"] = cal_path
    print(
        f"Saved LogZCalibrator from {n_used} train traces ({calibrator.n_samples} assistant tokens, "
        f"norm_weight={calibrator.norm_weight:.4f}, bias={calibrator.bias:.4f}) -> {cal_path}"
    )
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Direction 1 shortlist vocabularies & logZ calibrator.")
    parser.add_argument("--draft-layer", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--logz-ridge-lambda", type=float, default=2000.0)
    args = parser.parse_args()
    build_direction1_artifacts(
        draft_layer=args.draft_layer,
        max_tokens=args.max_tokens,
        logz_ridge_lambda=args.logz_ridge_lambda,
    )


if __name__ == "__main__":
    main()
