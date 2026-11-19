"""Unit tests for pure analysis helpers in seed_agent_lab."""

import math
import numpy as np

from seed_agent_lab.analysis import (
    classify_agent_token_texts,
    compute_layer_exit_metrics,
    compute_state_sink_stats,
    simulate_greedy_speculative,
    summarize_sink_stats,
)


def test_state_sink_rank_one_dominant() -> None:
    rng = np.random.default_rng(42)
    u = rng.normal(size=(64, 1))
    v = rng.normal(size=(1, 64))
    sink = 20.0 * (u @ v)
    noise = 0.05 * rng.normal(size=(64, 64))
    stats = compute_state_sink_stats(sink + noise)

    assert stats.top1_energy_share > 0.99
    assert stats.rank_90_energy == 1
    assert stats.rank_99_energy == 1


def test_layer_exit_metrics_exact_match() -> None:
    logits = np.array([[10.0, 0.0, -5.0], [0.0, 8.0, 1.0]])
    metrics = compute_layer_exit_metrics(depth=16, shallow_logits=logits, full_logits=logits)
    assert metrics.depth == 16
    assert math.isclose(metrics.top1_match_rate, 1.0)
    assert math.isclose(metrics.top5_match_rate, 1.0)
    assert math.isclose(metrics.mean_kl_nats, 0.0, abs_tol=1e-9)


def test_classify_agent_token_texts_qwen35_coding_echo() -> None:
    tokens = [
        "Inspect ",
        "src/app.py",
        " first. ",
        "<tool_call>",
        "\n",
        "<function=edit_file>",
        "\n",
        "<parameter=file_path>",
        "\n",
        "src/app.py",
        "\n",
        "</parameter>",
        "\n",
        "<parameter=new_string>",
        "\n",
        "pool_size=8",
        "\n",
        "</parameter>",
        "\n",
        "</function>",
        "\n",
        "</tool_call>",
        " Done",
    ]
    tags = classify_agent_token_texts(tokens)
    assert tags == (
        "reasoning",
        "reasoning",
        "reasoning",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "repeated_echo",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "new_code_arg",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "tool_boilerplate",
        "reasoning",
    )


def test_simulate_greedy_speculative() -> None:
    draft = [1, 2, 3, 4, 99, 6]
    target = [1, 2, 3, 4, 5, 6]
    probs = [0.95] * 6
    stats = simulate_greedy_speculative(draft, target, probs, max_draft_len=6)
    assert stats.verify_calls == 2
    assert math.isclose(stats.mean_accepted_tokens, 3.0)


def test_summarize_sink_stats() -> None:
    s1 = compute_state_sink_stats(np.diag([10.0, 0.0, 0.0]))
    s2 = compute_state_sink_stats(np.diag([1.0, 1.0, 0.0]))
    summary = summarize_sink_stats((s1, s2))
    assert summary["median_top1"] == 0.75
    assert summary["pooled_top1"] > 0.98
