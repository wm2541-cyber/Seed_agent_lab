"""Unit tests for Prompt Lookup + Layer-8 Hybrid Speculative simulation."""

import math

from seed_agent_lab.hybrid_speculative import find_prompt_lookup_draft, simulate_hybrid_policy


def test_find_prompt_lookup_draft_matches_most_recent() -> None:
    # Earlier context contains [10, 20, 30, 40, 50, 60]; current suffix is [10, 20, 30]
    history = [1, 2, 10, 20, 30, 40, 50, 60, 99, 10, 20, 30]
    draft = find_prompt_lookup_draft(history, ngram_size=3, max_draft_len=4)
    assert draft == (40, 50, 60, 99)


def test_simulate_hybrid_policy_lookup_boosts_copy() -> None:
    # Tokens 0..5 are prompt: [100, 101, 102, 103, 104, 105]
    # Assistant starts at index 5 predicting [100, 101, 102, 103, 104, 105] (exact copy!)
    # Notice full_sequence_tokens[pos] is input token at pos; target_top1 at pos is full_sequence_tokens[pos+1]
    full_seq = [100, 101, 102, 103, 104, 105, 100, 101, 102, 103, 104, 105]
    # Assistant positions predicting index 6..11 -> input positions 5..10; wait needle needs 3 matching tokens inside history:
    # At pos 8 (after 100, 101, 102 have been emitted), lookup matches [100, 101, 102] and copies [103, 104, 105]!
    asst_pos = [5, 6, 7, 8, 9, 10]
    target = [100, 101, 102, 103, 104, 105]
    shallow = [999, 999, 999, 999, 999, 999]  # neural fails completely
    probs = [0.2] * 6

    stats = simulate_hybrid_policy(
        policy_name="lookup_only",
        full_sequence_tokens=full_seq,
        assistant_positions=asst_pos,
        shallow_top1_tokens=shallow,
        target_top1_tokens=target,
        shallow_max_probs=probs,
        use_neural=False,
        use_lookup=True,
        max_lookup_len=6,
        ngram_size=3,
    )
    # pos 5->100 (miss), pos 6->101 (miss), pos 7->102 (miss), pos 8: suffix is [100,101,102] -> copies [103,104,105] all match!
    assert stats.verify_calls == 4
    assert math.isclose(stats.mean_accepted_tokens, 6 / 4)
    assert stats.avg_lookup_steps > 0
