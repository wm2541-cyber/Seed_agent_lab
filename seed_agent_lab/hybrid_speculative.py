"""Pure simulation of Hybrid Layer-8 Linear Rotation + Prompt Lookup Speculative Decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HybridSpecStats:
    """Immutable Roofline and acceptance metrics for a draft-verify policy."""

    policy_name: str
    mean_accepted_tokens: float
    acceptance_rate: float
    avg_neural_steps: float
    avg_lookup_steps: float
    avg_weight_read_cost: float
    theoretical_speedup: float
    verify_calls: int
    total_tokens: int


def find_prompt_lookup_draft(
    history_tokens: Sequence[int],
    ngram_size: int = 3,
    max_draft_len: int = 8,
) -> tuple[int, ...]:
    """Find the most recent exact suffix N-gram match in prior history and copy following tokens."""
    n = len(history_tokens)
    if n < ngram_size + 1 or max_draft_len <= 0:
        return ()

    needle = tuple(history_tokens[-ngram_size:])
    # Search backwards in strictly earlier context so we never match the needle's own position.
    search_limit = n - ngram_size
    for start in range(search_limit - 1, -1, -1):
        if tuple(history_tokens[start : start + ngram_size]) == needle:
            copy_start = start + ngram_size
            copy_end = min(copy_start + max_draft_len, search_limit)
            if copy_end > copy_start:
                return tuple(history_tokens[copy_start:copy_end])
    return ()


def _burst_weight_cost(
    neural_steps: int,
    lookup_steps: int,
    shallow_depth_ratio: float,
) -> float:
    """Compute exact full-model-equivalent weight read cost for one draft-verify cycle."""
    if lookup_steps > 0 or neural_steps == 0:
        # Tokens produced purely via lookup still require running layers 0..k-1 during Verify.
        return shallow_depth_ratio * neural_steps + 1.0
    # When all drafted tokens already ran layers 0..k-1, Verify only runs layers k..L-1.
    return shallow_depth_ratio * neural_steps + (1.0 - shallow_depth_ratio)


def simulate_hybrid_policy(
    policy_name: str,
    full_sequence_tokens: Sequence[int],
    assistant_positions: Sequence[int],
    shallow_top1_tokens: Sequence[int],
    target_top1_tokens: Sequence[int],
    shallow_max_probs: Sequence[float],
    shallow_depth_ratio: float = 0.25,
    use_neural: bool = True,
    use_lookup: bool = True,
    lookup_first: bool = True,
    max_neural_len: int = 6,
    max_lookup_len: int = 8,
    ngram_size: int = 3,
    confidence_threshold: float = 0.70,
) -> HybridSpecStats:
    """Simulate greedy draft-verify on assistant turns with contiguous verified tokens."""
    n_asst = len(assistant_positions)
    if n_asst == 0:
        return HybridSpecStats(policy_name, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 0)

    if not (len(shallow_top1_tokens) == len(target_top1_tokens) == len(shallow_max_probs) == n_asst):
        raise ValueError("Assistant-aligned arrays must have identical lengths.")

    verify_calls = 0
    accepted_draft_tokens = 0
    proposed_draft_tokens = 0
    total_neural_steps = 0
    total_lookup_steps = 0
    total_weight_cost = 0.0

    idx = 0
    while idx < n_asst:
        # Determine how far this contiguous assistant turn extends before the next user/tool turn.
        burst_end = idx
        while burst_end + 1 < n_asst and assistant_positions[burst_end + 1] == assistant_positions[burst_end] + 1:
            burst_end += 1
        max_avail = (burst_end - idx) + 1

        proposed: list[int] = []
        neural_steps = 0
        lookup_steps = 0

        cur_seq_pos = assistant_positions[idx]
        # History up to and including the token just emitted at cur_seq_pos (predicting cur_seq_pos + 1).
        history_buf = list(full_sequence_tokens[: cur_seq_pos + 1])

        def _try_lookup() -> bool:
            nonlocal lookup_steps
            copied = find_prompt_lookup_draft(
                history_buf,
                ngram_size=ngram_size,
                max_draft_len=min(max_lookup_len, max_avail - len(proposed)),
            )
            if not copied:
                return False
            for tok in copied:
                proposed.append(tok)
                history_buf.append(tok)
                lookup_steps += 1
            return True

        if use_lookup and lookup_first:
            _try_lookup()

        if use_neural and len(proposed) == 0:
            while len(proposed) < min(max_neural_len, max_avail):
                pos_idx = idx + len(proposed)
                tok = int(shallow_top1_tokens[pos_idx])
                prob = float(shallow_max_probs[pos_idx])
                proposed.append(tok)
                history_buf.append(tok)
                neural_steps += 1
                if confidence_threshold > 0.0 and prob < confidence_threshold:
                    if use_lookup and not lookup_first and len(proposed) < max_avail:
                        _try_lookup()
                    break

        verify_calls += 1
        proposed_draft_tokens += len(proposed)
        total_neural_steps += neural_steps
        total_lookup_steps += lookup_steps
        total_weight_cost += _burst_weight_cost(neural_steps, lookup_steps, shallow_depth_ratio)

        matched = 0
        for step_offset, draft_tok in enumerate(proposed):
            if draft_tok == int(target_top1_tokens[idx + step_offset]):
                matched += 1
            else:
                break

        accepted_draft_tokens += matched
        # Advance by matched tokens + 1 verified correction/next token from target model.
        idx += min(matched + 1, max_avail)

    mean_accepted = float(n_asst / verify_calls)
    acc_rate = float(accepted_draft_tokens / proposed_draft_tokens) if proposed_draft_tokens > 0 else 0.0
    avg_cost = float(total_weight_cost / verify_calls)
    speedup = float(mean_accepted / avg_cost) if avg_cost > 0 else 1.0
    return HybridSpecStats(
        policy_name=policy_name,
        mean_accepted_tokens=mean_accepted,
        acceptance_rate=acc_rate,
        avg_neural_steps=float(total_neural_steps / verify_calls),
        avg_lookup_steps=float(total_lookup_steps / verify_calls),
        avg_weight_read_cost=avg_cost,
        theoretical_speedup=speedup,
        verify_calls=verify_calls,
        total_tokens=n_asst,
    )
