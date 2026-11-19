"""Unit tests for pure functional phase router and policy presets."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pytest

from direction1_phase_shortlist.phase_router import (
    GenerationPhase,
    PhaseTagIds,
    advance_phase,
    extract_phase_tag_ids,
    infer_phase_from_tokens,
    lookup_only_routing,
    phase_adaptive_routing,
    phase_gated_routing,
    policy_for_phase,
    step_phase,
    uniform_hybrid_routing,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self._map = {
            "<think>": [248068],
            "</think>": [248069],
            "<tool_call>": [248058],
            "</tool_call>": [248059],
            "<|im_end|>": [248046],
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._map.get(text, [1, 2]))


def test_extract_phase_tag_ids_resolves_atomic_tags() -> None:
    tags = extract_phase_tag_ids(_DummyTokenizer())
    assert tags == PhaseTagIds(
        think_start=248068,
        think_end=248069,
        tool_call_start=248058,
        tool_call_end=248059,
        im_end=248046,
    )


def test_step_and_infer_phase_on_qwen_prompt_templates() -> None:
    tags = extract_phase_tag_ids(_DummyTokenizer())

    # Qwen3.5 thinking-enabled prompt ends with `<|im_start|>assistant\n<think>\n`
    thinking_prompt = [10, 20, 248046, 198, 248045, 74455, 198, 248068, 198]
    original_copy = list(thinking_prompt)
    assert infer_phase_from_tokens(thinking_prompt, tags) is GenerationPhase.THINKING
    assert thinking_prompt == original_copy  # no mutation

    # Multi-turn trace with closed thinking and closed tool_call followed by non-thinking turn
    response_prompt = [248068, 100, 248069, 248058, 200, 248059, 248046, 198]
    assert infer_phase_from_tokens(response_prompt, tags) is GenerationPhase.RESPONSE

    # Inside an open tool_call
    tool_stream = [248068, 11, 248069, 198, 248058, 42]
    assert infer_phase_from_tokens(tool_stream, tags) is GenerationPhase.TOOL_CALL

    # Closing tool_call returns to RESPONSE; im_end resets any phase to RESPONSE
    assert step_phase(GenerationPhase.TOOL_CALL, 248059, tags) is GenerationPhase.RESPONSE
    assert step_phase(GenerationPhase.THINKING, 248046, tags) is GenerationPhase.RESPONSE
    assert advance_phase(GenerationPhase.THINKING, [5, 6, 248069, 248058], tags) is GenerationPhase.TOOL_CALL


def test_routing_presets_and_immutability() -> None:
    gated = phase_gated_routing(confidence_threshold=0.72, max_neural_len=5)
    assert not policy_for_phase(gated, GenerationPhase.THINKING).use_neural
    assert policy_for_phase(gated, GenerationPhase.THINKING).use_lookup
    assert policy_for_phase(gated, GenerationPhase.RESPONSE).use_neural
    assert policy_for_phase(gated, GenerationPhase.TOOL_CALL).confidence_threshold == 0.72

    adaptive = phase_adaptive_routing(think_confidence_threshold=0.88, think_max_neural_len=2)
    assert policy_for_phase(adaptive, GenerationPhase.THINKING).use_neural
    assert policy_for_phase(adaptive, GenerationPhase.THINKING).confidence_threshold == 0.88
    assert policy_for_phase(adaptive, GenerationPhase.THINKING).max_neural_len == 2

    lookup = lookup_only_routing()
    assert not policy_for_phase(lookup, GenerationPhase.TOOL_CALL).use_neural

    uniform = uniform_hybrid_routing()
    assert policy_for_phase(uniform, GenerationPhase.THINKING) == policy_for_phase(uniform, GenerationPhase.TOOL_CALL)

    with pytest.raises(FrozenInstanceError):
        gated.thinking = lookup.thinking  # type: ignore[misc]
