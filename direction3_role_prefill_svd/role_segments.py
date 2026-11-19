"""Pure functional role parsing and controlled depth-schedule builders for SWE-smith trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class TokenRole(str, Enum):
    """Semantic role of each prompt token in a multi-turn Qwen3.5 SWE trajectory."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class RoleTokenSpan:
    """Immutable contiguous span of tokens sharing the same TokenRole."""

    role: TokenRole
    start: int
    end: int
    tool_turn_index: int | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class DepthSchedule:
    """Immutable chunked prefill depth plan with exact FLOPs-saving telemetry."""

    policy_name: str
    chunks: tuple[tuple[tuple[int, ...], int], ...]
    per_token_depths: tuple[int, ...]
    full_depth: int
    skipped_layer_fraction: float
    skipped_token_count: int
    skipped_by_role: tuple[tuple[str, int], ...]
    mean_skipped_rel_pos: float


def parse_role_tags(
    token_texts: Sequence[str],
    token_ids: Sequence[int] | None = None,
    im_start_id: int = 248045,
    im_end_id: int = 248046,
    tool_resp_start_id: int = 248066,
    tool_resp_end_id: int = 248067,
) -> tuple[TokenRole, ...]:
    """Purely assign each token to SYSTEM, USER, ASSISTANT, or TOOL.

    In Qwen3.5 chat templates, tool outputs live inside `<|im_start|>user\n<tool_response>...</tool_response><|im_end|>`.
    Tokens inside `<tool_response>...</tool_response>` are classified strictly as `TokenRole.TOOL`,
    separating bash/editor logs cleanly from genuine `TokenRole.USER` task instructions.
    """
    n = len(token_texts)
    if token_ids is not None and len(token_ids) != n:
        raise ValueError(f"Mismatched lengths: {len(token_texts)} texts vs {len(token_ids)} ids")

    roles: list[TokenRole] = []
    turn_role = TokenRole.SYSTEM
    expect_role_header = False
    in_tool_response = False

    for idx, text in enumerate(token_texts):
        tid = int(token_ids[idx]) if token_ids is not None else None
        is_im_start = (tid == im_start_id) if tid is not None else (text == "<|im_start|>")
        is_im_end = (tid == im_end_id) if tid is not None else (text == "<|im_end|>")
        is_tr_start = (tid == tool_resp_start_id) if tid is not None else ("<tool_response>" in text)
        is_tr_end = (tid == tool_resp_end_id) if tid is not None else ("</tool_response>" in text)

        if is_im_start:
            expect_role_header = True
            in_tool_response = False
            # Look ahead one token to label the turn delimiter with the upcoming turn role.
            next_text = token_texts[idx + 1].strip() if idx + 1 < n else ""
            if next_text.startswith("system"):
                turn_role = TokenRole.SYSTEM
            elif next_text.startswith("assistant"):
                turn_role = TokenRole.ASSISTANT
            elif next_text.startswith("user"):
                turn_role = TokenRole.USER
            elif next_text.startswith("tool"):
                turn_role = TokenRole.TOOL
            roles.append(turn_role)
            continue

        if expect_role_header:
            stripped = text.strip()
            if stripped.startswith("system"):
                turn_role = TokenRole.SYSTEM
                expect_role_header = False
            elif stripped.startswith("assistant"):
                turn_role = TokenRole.ASSISTANT
                expect_role_header = False
            elif stripped.startswith("user"):
                turn_role = TokenRole.USER
                expect_role_header = False
            elif stripped.startswith("tool"):
                turn_role = TokenRole.TOOL
                expect_role_header = False
            roles.append(turn_role)
            continue

        if is_tr_start:
            in_tool_response = True
            roles.append(TokenRole.TOOL)
            continue

        if is_tr_end:
            roles.append(TokenRole.TOOL)
            in_tool_response = False
            continue

        if is_im_end:
            roles.append(TokenRole.TOOL if in_tool_response else turn_role)
            in_tool_response = False
            continue

        roles.append(TokenRole.TOOL if in_tool_response else turn_role)

    return tuple(roles)


def build_role_spans(roles: Sequence[TokenRole]) -> tuple[RoleTokenSpan, ...]:
    """Purely group consecutive identical roles into spans, numbering each TOOL turn sequentially."""
    if not roles:
        return ()
    spans: list[RoleTokenSpan] = []
    start = 0
    tool_idx = 0
    cur = roles[0]
    for i in range(1, len(roles)):
        if roles[i] is not cur:
            t_idx = tool_idx if cur is TokenRole.TOOL else None
            if cur is TokenRole.TOOL:
                tool_idx += 1
            spans.append(RoleTokenSpan(role=cur, start=start, end=i, tool_turn_index=t_idx))
            start = i
            cur = roles[i]
    t_idx = tool_idx if cur is TokenRole.TOOL else None
    spans.append(RoleTokenSpan(role=cur, start=start, end=len(roles), tool_turn_index=t_idx))
    return tuple(spans)


def _merge_depths_into_chunks(
    token_ids: Sequence[int],
    depths: Sequence[int],
    max_chunk_len: int,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Purely pack consecutive equal-depth tokens into bounded-size prefill chunks."""
    if not token_ids:
        return ()
    if max_chunk_len <= 0:
        raise ValueError(f"max_chunk_len must be positive, got {max_chunk_len}")
    chunks: list[tuple[tuple[int, ...], int]] = []
    start = 0
    cur_d = int(depths[0])
    for i in range(1, len(token_ids)):
        d = int(depths[i])
        if d != cur_d or (i - start) >= max_chunk_len:
            chunks.append((tuple(int(t) for t in token_ids[start:i]), cur_d))
            start = i
            cur_d = d
    chunks.append((tuple(int(t) for t in token_ids[start:]), cur_d))
    return tuple(chunks)


def _select_matched_indices(
    roles: Sequence[TokenRole],
    target_roles: frozenset[TokenRole],
    fallback_roles: frozenset[TokenRole],
    quota: int,
    prefer_recent: bool = False,
) -> frozenset[int]:
    """Purely select exactly `quota` token indices from `target_roles` (with deterministic fallback)."""
    primary = [i for i, r in enumerate(roles) if r in target_roles]
    fallback = [i for i, r in enumerate(roles) if r in fallback_roles]
    if prefer_recent:
        primary = list(reversed(primary))
        fallback = list(reversed(fallback))
    ordered = primary + fallback
    return frozenset(ordered[: max(0, quota)])


def build_depth_schedule(
    token_ids: Sequence[int],
    roles: Sequence[TokenRole],
    policy_name: str,
    shallow_k: int = 16,
    full_depth: int = 32,
    recent_window: int = 256,
    max_chunk_len: int = 1024,
) -> DepthSchedule:
    """Purely construct a role-aware or matched-budget prefill depth schedule.

    Supported `policy_name` values:
      - `Full_32L`
      - `Uniform_k16` / `Uniform_k24` (uses `shallow_k`)
      - `Skip_AllTool_k16` / `Skip_AllTool_k20` (all TOOL tokens at `shallow_k`, others at `full_depth`)
      - `Skip_OldTool_k16` (all TOOL spans except the final TOOL span at `shallow_k`)
      - `Adaptive_old16_recent24` (older tokens at `shallow_k`, final `recent_window` tokens at 24)
      - `Matched_SkipTool_k16` (exact `m = min(n_tool, n_non_tool)` TOOL tokens at `shallow_k`)
      - `Matched_SkipUserAsst_k16` (exact same `m` USER/ASSISTANT tokens at `shallow_k`, all TOOL at `full_depth`)
      - `Matched_SkipUserAsstRecent_k16` (exact same `m` most-recent USER/ASSISTANT tokens at `shallow_k`)
    """
    n = len(token_ids)
    if len(roles) != n:
        raise ValueError(f"Mismatched lengths: {n} token_ids vs {len(roles)} roles")
    if n == 0:
        return DepthSchedule(
            policy_name=policy_name,
            chunks=(),
            per_token_depths=(),
            full_depth=full_depth,
            skipped_layer_fraction=0.0,
            skipped_token_count=0,
            skipped_by_role=(),
            mean_skipped_rel_pos=0.0,
        )

    spans = build_role_spans(roles)
    tool_spans = [sp for sp in spans if sp.role is TokenRole.TOOL]
    n_tool = sum(sp.length for sp in tool_spans)
    n_non_tool = n - n_tool
    matched_m = min(n_tool, n_non_tool)

    if policy_name == "Full_32L":
        depths = [full_depth] * n
    elif policy_name.startswith("Uniform_k"):
        depths = [shallow_k] * n
    elif policy_name.startswith("Skip_AllTool_k"):
        depths = [shallow_k if r is TokenRole.TOOL else full_depth for r in roles]
    elif policy_name.startswith("Skip_OldTool_k"):
        last_tool_idx = tool_spans[-1].tool_turn_index if tool_spans else None
        old_tool_indices: set[int] = set()
        for sp in tool_spans:
            if sp.tool_turn_index != last_tool_idx:
                old_tool_indices.update(range(sp.start, sp.end))
        depths = [shallow_k if i in old_tool_indices else full_depth for i in range(n)]
    elif policy_name == "Adaptive_old16_recent24":
        recent_start = max(0, n - recent_window)
        recent_depth = min(24, full_depth)
        depths = [shallow_k if i < recent_start else recent_depth for i in range(n)]
    elif policy_name == "Matched_SkipTool_k16":
        chosen = _select_matched_indices(
            roles,
            target_roles=frozenset({TokenRole.TOOL}),
            fallback_roles=frozenset(),
            quota=matched_m,
            prefer_recent=False,
        )
        depths = [shallow_k if i in chosen else full_depth for i in range(n)]
    elif policy_name == "Matched_SkipUserAsst_k16":
        chosen = _select_matched_indices(
            roles,
            target_roles=frozenset({TokenRole.USER, TokenRole.ASSISTANT}),
            fallback_roles=frozenset({TokenRole.SYSTEM}),
            quota=matched_m,
            prefer_recent=False,
        )
        depths = [shallow_k if i in chosen else full_depth for i in range(n)]
    elif policy_name == "Matched_SkipUserAsstRecent_k16":
        chosen = _select_matched_indices(
            roles,
            target_roles=frozenset({TokenRole.USER, TokenRole.ASSISTANT}),
            fallback_roles=frozenset({TokenRole.SYSTEM}),
            quota=matched_m,
            prefer_recent=True,
        )
        depths = [shallow_k if i in chosen else full_depth for i in range(n)]
    else:
        raise ValueError(f"Unknown prefill depth policy: {policy_name}")

    chunks = _merge_depths_into_chunks(token_ids, depths, max_chunk_len=max_chunk_len)
    total_layer_ops = n * full_depth
    executed_layer_ops = sum(depths)
    skipped_layer_fraction = (total_layer_ops - executed_layer_ops) / total_layer_ops if total_layer_ops > 0 else 0.0

    skipped_positions = [i for i, d in enumerate(depths) if d < full_depth]
    by_role_counts = {r.value: 0 for r in TokenRole}
    for pos in skipped_positions:
        by_role_counts[roles[pos].value] += 1
    mean_rel_pos = (
        sum(pos / max(1, n - 1) for pos in skipped_positions) / len(skipped_positions)
        if skipped_positions
        else 0.0
    )

    return DepthSchedule(
        policy_name=policy_name,
        chunks=chunks,
        per_token_depths=tuple(depths),
        full_depth=full_depth,
        skipped_layer_fraction=skipped_layer_fraction,
        skipped_token_count=len(skipped_positions),
        skipped_by_role=tuple(sorted(by_role_counts.items())),
        mean_skipped_rel_pos=mean_rel_pos,
    )
