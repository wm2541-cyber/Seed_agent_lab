"""Unit tests for pure functional shortlist vocabulary construction and coverage evaluation."""

from __future__ import annotations

from pathlib import Path
import pytest

from direction1_phase_shortlist.shortlist_vocab import (
    build_dynamic_shortlist,
    build_static_shortlist,
    count_token_frequencies,
    evaluate_shortlist_coverage,
    load_static_shortlist,
    save_static_shortlist,
)


def test_count_token_frequencies_masked_and_pure() -> None:
    seqs = [[10, 20, 10, 30], [20, 40]]
    masks = [[False, True, True, False], [True, False]]
    seqs_before = [list(s) for s in seqs]

    masked = count_token_frequencies(seqs, masks=masks)
    assert masked == {10: 1, 20: 2}
    unmasked = count_token_frequencies(seqs)
    assert unmasked == {10: 2, 20: 2, 30: 1, 40: 1}
    assert seqs == seqs_before

    with pytest.raises(ValueError):
        count_token_frequencies(seqs, masks=[[True]])


def test_build_static_shortlist_tier_priority_and_bpe_fallback() -> None:
    asst_counts = {50: 3, 40: 10, 60: 3}
    all_counts = {70: 100, 40: 20, 80: 5}
    pinned = [99, 98, 99]

    sl = build_static_shortlist(
        vocab_size=120,
        target_size=10,
        assistant_counts=asst_counts,
        all_token_counts=all_counts,
        pinned_token_ids=pinned,
        metadata={"split": "train"},
    )
    # Tier 1: 99, 98
    # Tier 2 (asst desc, tie asc): 40 (10), 50 (3), 60 (3)
    # Tier 3 (all desc excluding seen): 70 (100), 80 (5)
    # Tier 4 (low BPE IDs): 0, 1, 2
    assert sl.token_ids == (99, 98, 40, 50, 60, 70, 80, 0, 1, 2)
    assert sl.size == 10
    assert len(set(sl.token_ids)) == 10


def test_dynamic_shortlist_and_coverage_and_roundtrip(tmp_path: Path) -> None:
    sl = build_static_shortlist(
        vocab_size=100,
        target_size=4,
        assistant_counts={10: 5, 20: 4},
        all_token_counts={30: 2},
        pinned_token_ids=[90],
    )
    assert sl.token_ids == (90, 10, 20, 30)

    prompt = [10, 55, 55, 66, 90, 77]
    dyn = build_dynamic_shortlist(sl, prompt)
    assert dyn.extra_prompt_ids == (55, 66, 77)
    assert dyn.all_ids == (90, 10, 20, 30, 55, 66, 77)
    assert dyn.total_size == 7

    cov = evaluate_shortlist_coverage(sl, prompt, [10, 55, 66, 88])
    assert cov.total_tokens == 4
    assert cov.covered_by_static == 1  # token 10
    assert cov.covered_by_dynamic == 3  # tokens 10, 55, 66
    assert cov.oov_tokens == 1  # token 88
    assert cov.static_coverage == pytest.approx(0.25)
    assert cov.dynamic_coverage == pytest.approx(0.75)
    assert cov.oov_rate == pytest.approx(0.25)

    saved = save_static_shortlist(tmp_path / "sl.npz", sl)
    reloaded = load_static_shortlist(saved)
    assert reloaded == sl
