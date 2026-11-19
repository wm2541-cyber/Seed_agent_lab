"""Unit tests for pure functional role segmentation and matched-budget depth schedules."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pytest

from direction3_role_prefill_svd.role_segments import (
    TokenRole,
    build_depth_schedule,
    build_role_spans,
    parse_role_tags,
)


def _sample_multi_turn_tokens() -> tuple[list[int], list[str]]:
    # Simulated Qwen3.5 multi-turn sequence: system -> user -> assistant -> tool_1 -> assistant -> tool_2
    items = [
        (248045, "<|im_start|>"), (1, "system"), (2, "\n"), (3, "sys"), (248046, "<|im_end|>"),
        (248045, "<|im_start|>"), (4, "user"), (2, "\n"), (5, "fix"), (6, "bug"), (248046, "<|im_end|>"),
        (248045, "<|im_start|>"), (7, "assistant"), (2, "\n"), (8, "call1"), (248046, "<|im_end|>"),
        (248045, "<|im_start|>"), (4, "user"), (2, "\n"),
        (248066, "<tool_response>"), (9, "ls"), (10, "a.py"), (248067, "</tool_response>"), (248046, "<|im_end|>"),
        (248045, "<|im_start|>"), (7, "assistant"), (2, "\n"), (11, "call2"), (248046, "<|im_end|>"),
        (248045, "<|im_start|>"), (4, "user"), (2, "\n"),
        (248066, "<tool_response>"), (12, "def"), (13, "f"), (14, "pass"), (248067, "</tool_response>"), (248046, "<|im_end|>"),
    ]
    return [t for t, _ in items], [s for _, s in items]


def test_parse_role_tags_separates_tool_responses_from_user_prompt() -> None:
    tids, texts = _sample_multi_turn_tokens()
    texts_before = list(texts)
    roles = parse_role_tags(texts, token_ids=tids)
    assert texts == texts_before
    assert len(roles) == len(tids)

    spans = build_role_spans(roles)
    tool_spans = [sp for sp in spans if sp.role is TokenRole.TOOL]
    assert len(tool_spans) == 2
    assert [sp.tool_turn_index for sp in tool_spans] == [0, 1]
    assert tool_spans[0].length == 4  # <tool_response>, ls, a.py, </tool_response>
    assert tool_spans[1].length == 5  # <tool_response>, def, f, pass, </tool_response>
    assert any(sp.role is TokenRole.SYSTEM for sp in spans)
    assert any(sp.role is TokenRole.USER for sp in spans)
    assert any(sp.role is TokenRole.ASSISTANT for sp in spans)


def test_depth_schedules_skip_old_tool_and_match_counterfactual_budget() -> None:
    tids, texts = _sample_multi_turn_tokens()
    roles = parse_role_tags(texts, token_ids=tids)

    all_tool = build_depth_schedule(tids, roles, "Skip_AllTool_k16", shallow_k=16, full_depth=32)
    old_tool = build_depth_schedule(tids, roles, "Skip_OldTool_k16", shallow_k=16, full_depth=32)
    m_tool = build_depth_schedule(tids, roles, "Matched_SkipTool_k16", shallow_k=16, full_depth=32)
    m_non = build_depth_schedule(tids, roles, "Matched_SkipUserAsst_k16", shallow_k=16, full_depth=32)
    m_non_rec = build_depth_schedule(tids, roles, "Matched_SkipUserAsstRecent_k16", shallow_k=16, full_depth=32)

    assert all_tool.skipped_token_count == 9
    assert old_tool.skipped_token_count == 4  # skips only turn 0, keeps turn 1 at 32L!
    # Exact token-budget & FLOP-fraction equality across matched counterfactuals!
    assert m_tool.skipped_token_count == m_non.skipped_token_count == m_non_rec.skipped_token_count == 9
    assert m_tool.skipped_layer_fraction == pytest.approx(m_non.skipped_layer_fraction)
    assert dict(m_tool.skipped_by_role)["tool"] == 9
    assert dict(m_non.skipped_by_role)["tool"] == 0
    assert m_non_rec.mean_skipped_rel_pos > m_non.mean_skipped_rel_pos

    # Chunks reconstruct exact token sequence and obey max_chunk_len
    capped = build_depth_schedule(tids, roles, "Uniform_k16", shallow_k=16, full_depth=32, max_chunk_len=5)
    assert all(len(c_ids) <= 5 for c_ids, _ in capped.chunks)
    rebuilt = [tok for c_ids, _ in capped.chunks for tok in c_ids]
    assert rebuilt == tids

    with pytest.raises(FrozenInstanceError):
        all_tool.policy_name = "mutated"  # type: ignore[misc]
