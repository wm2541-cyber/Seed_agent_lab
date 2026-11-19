"""Pure analysis functions for State Sink SVD, shallow exit metrics, and speculative drafting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SinkStats:
    """Immutable SVD summary for one recurrent state matrix."""

    top1_energy_share: float
    top2_energy_share: float
    rank_90_energy: int
    rank_99_energy: int
    frobenius_norm_sq: float


@dataclass(frozen=True)
class DepthMetrics:
    """Immutable comparison between shallow-layer logits and full-model logits."""

    depth: int
    top1_match_rate: float
    top5_match_rate: float
    mean_kl_nats: float
    count: int


@dataclass(frozen=True)
class SpeculativeTraceStats:
    """Immutable summary of speculative acceptance under a draft-verify policy."""

    mean_accepted_tokens: float
    acceptance_rate: float
    verify_calls: int
    total_tokens: int


def compute_state_sink_stats(state_matrix: np.ndarray) -> SinkStats:
    """Compute singular energy shares and effective ranks of a 2D state matrix."""
    mat = np.asarray(state_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {mat.shape}")

    singular_values = np.linalg.svd(mat, compute_uv=False)
    energies = singular_values**2
    total_energy = float(np.sum(energies))
    if total_energy <= 0.0:
        return SinkStats(0.0, 0.0, 0, 0, 0.0)

    cumulative = np.cumsum(energies) / total_energy
    top1 = float(energies[0] / total_energy)
    top2 = float(energies[1] / total_energy) if len(energies) > 1 else 0.0
    rank_90 = int(np.searchsorted(cumulative, 0.90, side="left")) + 1
    rank_99 = int(np.searchsorted(cumulative, 0.99, side="left")) + 1
    return SinkStats(
        top1_energy_share=top1,
        top2_energy_share=top2,
        rank_90_energy=rank_90,
        rank_99_energy=rank_99,
        frobenius_norm_sq=total_energy,
    )


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def compute_layer_exit_metrics(
    depth: int,
    shallow_logits: np.ndarray,
    full_logits: np.ndarray,
) -> DepthMetrics:
    """Compare shallow early-exit logits against full-model logits."""
    s_arr = np.asarray(shallow_logits, dtype=np.float64)
    f_arr = np.asarray(full_logits, dtype=np.float64)
    if s_arr.shape != f_arr.shape or s_arr.ndim != 2:
        raise ValueError(f"Shape mismatch: {s_arr.shape} vs {f_arr.shape}")

    n_tokens = s_arr.shape[0]
    if n_tokens == 0:
        return DepthMetrics(depth=depth, top1_match_rate=0.0, top5_match_rate=0.0, mean_kl_nats=0.0, count=0)

    target_top1 = np.argmax(f_arr, axis=-1)
    shallow_top1 = np.argmax(s_arr, axis=-1)
    top1_rate = float(np.mean(shallow_top1 == target_top1))

    top5_indices = np.argpartition(s_arr, -min(5, s_arr.shape[-1]), axis=-1)[:, -5:]
    top5_hits = np.any(top5_indices == target_top1[:, None], axis=-1)
    top5_rate = float(np.mean(top5_hits))

    log_p = _log_softmax(f_arr)
    log_q = _log_softmax(s_arr)
    p = np.exp(log_p)
    kl_per_token = np.sum(p * (log_p - log_q), axis=-1)
    return DepthMetrics(
        depth=depth,
        top1_match_rate=top1_rate,
        top5_match_rate=top5_rate,
        mean_kl_nats=float(np.mean(kl_per_token)),
        count=n_tokens,
    )


_XML_TOOL_TAG_RE = re.compile(
    r"</?tool_call>|</?function(?:=[^>\n]+)?\s*>|</?parameter(?:=[^>\n]+)?\s*>"
)
_PARAM_VALUE_RE = re.compile(r"<parameter=[^>\n]+>\n?(.*?)\n?</parameter>", re.DOTALL)
_TOOL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def classify_agent_token_texts(token_texts: Sequence[str]) -> tuple[str, ...]:
    """Tag tokens into tool_boilerplate, repeated_echo, new_code_arg, or reasoning."""
    full_text = "".join(token_texts)
    if not full_text:
        return tuple()

    char_labels = ["reasoning"] * len(full_text)
    for block in _TOOL_BLOCK_RE.finditer(full_text):
        prior_text = full_text[: block.start()]
        for i in range(block.start(), block.end()):
            char_labels[i] = "new_code_arg"

        for param_match in _PARAM_VALUE_RE.finditer(block.group(0)):
            val_start = block.start() + param_match.start(1)
            val_text = param_match.group(1)
            # Mark lines/values that already appeared earlier in context as repeated_echo.
            line_cursor = 0
            for line in val_text.splitlines(keepends=True):
                stripped = line.strip()
                if len(stripped) >= 3 and stripped in prior_text:
                    for j in range(val_start + line_cursor, val_start + line_cursor + len(line)):
                        char_labels[j] = "repeated_echo"
                line_cursor += len(line)

    for tag_match in _XML_TOOL_TAG_RE.finditer(full_text):
        for i in range(tag_match.start(), tag_match.end()):
            char_labels[i] = "tool_boilerplate"

    for i, ch in enumerate(full_text):
        if char_labels[i] in ("new_code_arg", "repeated_echo") and ch in '{}[]":,\n\t ':
            if char_labels[i] == "new_code_arg":
                char_labels[i] = "tool_boilerplate"

    priority = ("new_code_arg", "repeated_echo", "tool_boilerplate", "reasoning")
    tags: list[str] = []
    cursor = 0
    for tok in token_texts:
        end = cursor + len(tok)
        span = char_labels[cursor:end]
        cursor = end
        chosen = "reasoning"
        for label in priority:
            if label in span:
                chosen = label
                break
        tags.append(chosen)
    return tuple(tags)


def simulate_greedy_speculative(
    draft_top1_tokens: Sequence[int],
    target_top1_tokens: Sequence[int],
    draft_max_probs: Sequence[float],
    max_draft_len: int = 6,
    confidence_threshold: float = 0.0,
) -> SpeculativeTraceStats:
    """Simulate greedy draft-verify acceptance length with optional confidence early-stop."""
    n = len(target_top1_tokens)
    if len(draft_top1_tokens) != n or len(draft_max_probs) != n:
        raise ValueError("All input sequences must have equal length.")
    if n == 0:
        return SpeculativeTraceStats(0.0, 0.0, 0, 0)

    idx = 0
    verify_calls = 0
    accepted_from_draft = 0
    proposed_from_draft = 0

    while idx < n:
        draft_steps = 0
        proposed_indices: list[int] = []
        while draft_steps < max_draft_len and (idx + draft_steps) < n:
            cur = idx + draft_steps
            proposed_indices.append(cur)
            draft_steps += 1
            if confidence_threshold > 0.0 and draft_max_probs[cur] < confidence_threshold:
                break

        verify_calls += 1
        proposed_from_draft += len(proposed_indices)

        matched = 0
        for p_idx in proposed_indices:
            if draft_top1_tokens[p_idx] == target_top1_tokens[p_idx]:
                matched += 1
            else:
                break

        accepted_from_draft += matched
        idx += matched + 1

    mean_accepted = float(n / verify_calls)
    acc_rate = float(accepted_from_draft / proposed_from_draft) if proposed_from_draft > 0 else 0.0
    return SpeculativeTraceStats(
        mean_accepted_tokens=mean_accepted,
        acceptance_rate=acc_rate,
        verify_calls=verify_calls,
        total_tokens=n,
    )


def summarize_sink_stats(stats: Iterable[SinkStats]) -> Mapping[str, float]:
    """Compute median and pooled top-1 energy share across a collection of head states."""
    items = tuple(stats)
    if not items:
        return {"median_top1": 0.0, "pooled_top1": 0.0, "median_rank90": 0.0, "median_rank99": 0.0}

    top1s = np.array([s.top1_energy_share for s in items], dtype=np.float64)
    energies = np.array([s.frobenius_norm_sq for s in items], dtype=np.float64)
    rank90s = np.array([s.rank_90_energy for s in items], dtype=np.float64)
    rank99s = np.array([s.rank_99_energy for s in items], dtype=np.float64)
    total_e = float(np.sum(energies))
    pooled = float(np.sum(top1s * energies) / total_e) if total_e > 0.0 else 0.0
    return {
        "median_top1": float(np.median(top1s)),
        "pooled_top1": pooled,
        "median_rank90": float(np.median(rank90s)),
        "median_rank99": float(np.median(rank99s)),
    }
