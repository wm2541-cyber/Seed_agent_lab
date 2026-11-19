"""Run local MVP-0 shallow-exit and State Sink SVD probes on Qwen3.5-4B (MLX)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from seed_agent_lab.analysis import (
    SinkStats,
    classify_agent_token_texts,
    compute_layer_exit_metrics,
    compute_state_sink_stats,
    summarize_sink_stats,
)

DEFAULT_MODEL_DIR = Path.home() / "Models" / "Qwen3.5-4B-MLX-8bit"

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read lines from a source file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["file_path", "start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact code snippet in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
]

_APP_PY_SNIPPET = (
    "def create_pool(dsn: str):\n"
    "    conn = sqlite3.connect(dsn)\n"
    "    conn.row_factory = sqlite3.Row\n"
    "    return conn\n"
)

_UPDATED_SNIPPET = (
    "def create_pool(dsn: str, timeout: float = 10.0):\n"
    "    conn = sqlite3.connect(dsn, timeout=timeout)\n"
    "    conn.row_factory = sqlite3.Row\n"
    "    return conn\n"
)

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a concise coding agent."},
    {"role": "user", "content": "Inspect src/services/db_pool.py and add a configurable timeout parameter."},
    {
        "role": "assistant",
        "reasoning_content": "",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": {"file_path": "src/services/db_pool.py", "start_line": 1, "end_line": 40},
                },
            }
        ],
    },
    {"role": "tool", "content": _APP_PY_SNIPPET},
    {
        "role": "assistant",
        "reasoning_content": "I see create_pool opens a sqlite3 connection without an explicit timeout. I will add a timeout parameter.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "arguments": {
                        "file_path": "src/services/db_pool.py",
                        "old_string": _APP_PY_SNIPPET,
                        "new_string": _UPDATED_SNIPPET,
                    },
                },
            }
        ],
    },
    {"role": "tool", "content": "Successfully updated src/services/db_pool.py."},
    {
        "role": "assistant",
        "reasoning_content": "Now let me re-read the file to verify the edit.",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": {"file_path": "src/services/db_pool.py", "start_line": 1, "end_line": 40},
                },
            }
        ],
    },
]


def _project_logits(model, hidden_states: mx.array) -> mx.array:
    """Apply final RMSNorm and LM head without mutating model weights."""
    lm = model.language_model
    normed = lm.model.norm(hidden_states)
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(normed)
    return lm.lm_head(normed)


def collect_layer_logits_and_states(
    model,
    token_ids: Sequence[int],
    probe_depths: Sequence[int],
) -> tuple[dict[int, np.ndarray], tuple[tuple[int, np.ndarray], ...]]:
    """Run one forward pass, recording logits at requested depths and final GatedDeltaNet states."""
    text_model = model.language_model.model
    cache = model.make_cache()
    inputs = mx.array([list(token_ids)], dtype=mx.int32)
    hidden = text_model.embed_tokens(inputs)

    fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx]) if text_model.fa_idx is not None else None
    ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx]) if text_model.ssm_idx is not None else None

    depth_set = set(probe_depths)
    depth_logits: dict[int, np.ndarray] = {}

    for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        depth = idx + 1
        if depth in depth_set:
            logits = _project_logits(model, hidden)[0]
            mx.eval(logits)
            depth_logits[depth] = np.array(logits.astype(mx.float32), copy=False)

    linear_states: list[tuple[int, np.ndarray]] = []
    for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
        if layer.is_linear and layer_cache[1] is not None:
            mx.eval(layer_cache[1])
            state_np = np.array(layer_cache[1][0].astype(mx.float32), copy=False)
            linear_states.append((idx, state_np))

    return depth_logits, tuple(linear_states)


def run_probes(model_dir: Path, depths: Sequence[int]) -> None:
    """Load Qwen3.5-4B, verify native tool-call generation, and run MVP-0 + State Sink SVD."""
    print(f"Loading model from: {model_dir}")
    model, tokenizer = mlx_lm.load(str(model_dir))

    # 1. Verify native tool-calling generation with Qwen3.5 chat_template.
    live_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Read lines 1 to 50 of src/main.py."}],
        tools=SAMPLE_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    live_reply = mlx_lm.generate(model, tokenizer, prompt=live_prompt, max_tokens=96, verbose=False)
    print("\n=== Live Tool-Call Smoke Test (enable_thinking=False) ===")
    print(live_reply.strip())

    # 2. Render full multi-turn Coding Agent trace using Qwen3.5's exact chat_template.
    rendered_trace = tokenizer.apply_chat_template(
        SAMPLE_MESSAGES,
        tools=SAMPLE_TOOLS,
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids = tokenizer.encode(rendered_trace)
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]
    tags = classify_agent_token_texts(token_texts)

    # Shift by 1: token[t] predicts token[t+1]
    input_ids = token_ids[:-1]
    target_tags = tags[1:]

    num_layers = len(model.layers)
    valid_depths = tuple(sorted({d for d in depths if 1 <= d <= num_layers} | {num_layers}))
    depth_logits, linear_states = collect_layer_logits_and_states(model, input_ids, valid_depths)
    full_logits = depth_logits[num_layers]

    print(f"\n=== MVP-0: Shallow Exit Agreement vs Full Model ({num_layers} layers, {len(input_ids)} tokens) ===")
    by_tag_indices: dict[str, list[int]] = defaultdict(list)
    for pos, tag in enumerate(target_tags):
        by_tag_indices[tag].append(pos)

    for tag in ("tool_boilerplate", "repeated_echo", "new_code_arg", "reasoning"):
        idxs = by_tag_indices.get(tag, [])
        if not idxs:
            continue
        print(f"\n[{tag}] (n={len(idxs)} tokens)")
        for d in valid_depths[:-1]:
            m = compute_layer_exit_metrics(d, depth_logits[d][idxs], full_logits[idxs])
            print(
                f"  layer {d:2d}/{num_layers} ({100*d/num_layers:4.0f}%): "
                f"top1={m.top1_match_rate:.3f}, top5={m.top5_match_rate:.3f}, kl={m.mean_kl_nats:.3f} nats"
            )

    print(f"\n=== State Sink SVD across {len(linear_states)} GatedDeltaNet layers ===")
    all_head_stats: list[SinkStats] = []
    for layer_idx, state_heads in linear_states:
        layer_stats = tuple(compute_state_sink_stats(head_mat) for head_mat in state_heads)
        all_head_stats.extend(layer_stats)
        if layer_idx in (0, 8, 16, 24, 30):
            summary = summarize_sink_stats(layer_stats)
            print(
                f"  layer {layer_idx:2d}: median_top1={summary['median_top1']:.3f}, "
                f"pooled_top1={summary['pooled_top1']:.3f}, "
                f"rank90={summary['median_rank90']:.1f}, rank99={summary['median_rank99']:.1f}"
            )

    overall = summarize_sink_stats(all_head_stats)
    print(
        f"\n  OVERALL ({len(all_head_stats)} heads): "
        f"median_top1={overall['median_top1']:.3f}, "
        f"pooled_top1={overall['pooled_top1']:.3f}, "
        f"median_rank90={overall['median_rank90']:.1f}, "
        f"median_rank99={overall['median_rank99']:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen3.5-4B shallow exit and State Sink SVD.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28, 32])
    args = parser.parse_args()
    run_probes(args.model_dir, args.depths)


if __name__ == "__main__":
    main()
