"""Pure functional phase state machine and routing policies for Qwen3.5 agent generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Protocol, Sequence


class _TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class GenerationPhase(str, Enum):
    """Semantic decoding phase tracked deterministically from emitted tokens."""

    THINKING = "thinking"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class PhaseTagIds:
    """Immutable token IDs for atomic Qwen3.5 structural phase delimiters."""

    think_start: int | None
    think_end: int | None
    tool_call_start: int | None
    tool_call_end: int | None
    im_end: int | None


@dataclass(frozen=True)
class PhaseDraftPolicy:
    """Immutable speculative drafting controls for a single generation phase."""

    use_lookup: bool = True
    use_neural: bool = True
    lookup_first: bool = True
    max_neural_len: int = 6
    max_lookup_len: int = 8
    ngram_size: int = 3
    confidence_threshold: float = 0.70


@dataclass(frozen=True)
class RoutingConfig:
    """Immutable mapping from every GenerationPhase to its PhaseDraftPolicy."""

    thinking: PhaseDraftPolicy
    response: PhaseDraftPolicy
    tool_call: PhaseDraftPolicy


def _encode_single_tag(tokenizer: _TokenizerLike, text: str) -> int | None:
    """Return the atomic token ID if `text` encodes to exactly one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def extract_phase_tag_ids(tokenizer: _TokenizerLike) -> PhaseTagIds:
    """Purely resolve Qwen3.5 structural phase delimiter token IDs from a tokenizer."""
    return PhaseTagIds(
        think_start=_encode_single_tag(tokenizer, "<think>"),
        think_end=_encode_single_tag(tokenizer, "</think>"),
        tool_call_start=_encode_single_tag(tokenizer, "<tool_call>"),
        tool_call_end=_encode_single_tag(tokenizer, "</tool_call>"),
        im_end=_encode_single_tag(tokenizer, "<|im_end|>"),
    )


def step_phase(
    phase: GenerationPhase,
    token_id: int,
    tags: PhaseTagIds,
) -> GenerationPhase:
    """Pure state transition for one committed token."""
    if tags.im_end is not None and token_id == tags.im_end:
        return GenerationPhase.RESPONSE
    if tags.think_start is not None and token_id == tags.think_start:
        return GenerationPhase.THINKING
    if tags.think_end is not None and token_id == tags.think_end:
        return GenerationPhase.RESPONSE
    if tags.tool_call_start is not None and token_id == tags.tool_call_start:
        return GenerationPhase.TOOL_CALL
    if tags.tool_call_end is not None and token_id == tags.tool_call_end:
        return GenerationPhase.RESPONSE
    return phase


def advance_phase(
    initial_phase: GenerationPhase,
    token_ids: Sequence[int],
    tags: PhaseTagIds,
) -> GenerationPhase:
    """Pure left-fold over a sequence of committed token IDs."""
    return reduce(lambda cur, tok: step_phase(cur, int(tok), tags), token_ids, initial_phase)


def infer_phase_from_tokens(
    prompt_ids: Sequence[int],
    tags: PhaseTagIds,
) -> GenerationPhase:
    """Infer the active generation phase at the end of `prompt_ids`."""
    return advance_phase(GenerationPhase.RESPONSE, prompt_ids, tags)


def policy_for_phase(
    config: RoutingConfig,
    phase: GenerationPhase,
) -> PhaseDraftPolicy:
    """Pure selector returning the immutable policy active for `phase`."""
    if phase is GenerationPhase.THINKING:
        return config.thinking
    if phase is GenerationPhase.TOOL_CALL:
        return config.tool_call
    return config.response


def lookup_only_routing(
    max_lookup_len: int = 8,
    ngram_size: int = 3,
) -> RoutingConfig:
    """Pure preset: zero-neural Prompt Lookup across all phases."""
    policy = PhaseDraftPolicy(
        use_lookup=True,
        use_neural=False,
        lookup_first=True,
        max_neural_len=0,
        max_lookup_len=max_lookup_len,
        ngram_size=ngram_size,
        confidence_threshold=0.0,
    )
    return RoutingConfig(thinking=policy, response=policy, tool_call=policy)


def uniform_hybrid_routing(
    confidence_threshold: float = 0.70,
    max_neural_len: int = 6,
    max_lookup_len: int = 8,
    ngram_size: int = 3,
) -> RoutingConfig:
    """Pure preset: identical Lookup-First + Neural L8 policy across all phases."""
    policy = PhaseDraftPolicy(
        use_lookup=True,
        use_neural=True,
        lookup_first=True,
        max_neural_len=max_neural_len,
        max_lookup_len=max_lookup_len,
        ngram_size=ngram_size,
        confidence_threshold=confidence_threshold,
    )
    return RoutingConfig(thinking=policy, response=policy, tool_call=policy)


def phase_gated_routing(
    confidence_threshold: float = 0.70,
    max_neural_len: int = 6,
    max_lookup_len: int = 8,
    ngram_size: int = 3,
) -> RoutingConfig:
    """Pure preset: skip L8 neural drafting inside <think> (LookupOnly), enable full Hybrid outside."""
    think_policy = PhaseDraftPolicy(
        use_lookup=True,
        use_neural=False,
        lookup_first=True,
        max_neural_len=0,
        max_lookup_len=max_lookup_len,
        ngram_size=ngram_size,
        confidence_threshold=0.0,
    )
    active_policy = PhaseDraftPolicy(
        use_lookup=True,
        use_neural=True,
        lookup_first=True,
        max_neural_len=max_neural_len,
        max_lookup_len=max_lookup_len,
        ngram_size=ngram_size,
        confidence_threshold=confidence_threshold,
    )
    return RoutingConfig(thinking=think_policy, response=active_policy, tool_call=active_policy)


def phase_adaptive_routing(
    think_confidence_threshold: float = 0.85,
    think_max_neural_len: int = 3,
    active_confidence_threshold: float = 0.70,
    active_max_neural_len: int = 6,
    max_lookup_len: int = 8,
    ngram_size: int = 3,
) -> RoutingConfig:
    """Pure preset: high-confidence conservative L8 inside <think>, standard Hybrid outside."""
    think_policy = PhaseDraftPolicy(
        use_lookup=True,
        use_neural=True,
        lookup_first=True,
        max_neural_len=think_max_neural_len,
        max_lookup_len=max_lookup_len,
        ngram_size=ngram_size,
        confidence_threshold=think_confidence_threshold,
    )
    active_policy = PhaseDraftPolicy(
        use_lookup=True,
        use_neural=True,
        lookup_first=True,
        max_neural_len=active_max_neural_len,
        max_lookup_len=max_lookup_len,
        ngram_size=ngram_size,
        confidence_threshold=active_confidence_threshold,
    )
    return RoutingConfig(thinking=think_policy, response=active_policy, tool_call=active_policy)
