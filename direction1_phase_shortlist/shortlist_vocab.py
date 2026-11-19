"""Pure functional shortlist vocabulary construction, dynamic prompt union, and coverage analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StaticShortlist:
    """Immutable static Top-K vocabulary derived purely from training corpus + BPE prior."""

    token_ids: tuple[int, ...]
    vocab_size: int
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def size(self) -> int:
        return len(self.token_ids)

    @property
    def token_set(self) -> frozenset[int]:
        return frozenset(self.token_ids)


@dataclass(frozen=True)
class DynamicShortlist:
    """Immutable prompt-conditioned shortlist = Static Top-K union unique unseen prompt token IDs."""

    static_ids: tuple[int, ...]
    extra_prompt_ids: tuple[int, ...]
    all_ids: tuple[int, ...]

    @property
    def total_size(self) -> int:
        return len(self.all_ids)

    @property
    def extra_size(self) -> int:
        return len(self.extra_prompt_ids)


@dataclass(frozen=True)
class ShortlistCoverageStats:
    """Immutable coverage & OOV statistics on an evaluation token stream."""

    shortlist_size: int
    total_tokens: int
    covered_by_static: int
    covered_by_dynamic: int
    oov_tokens: int

    @property
    def static_coverage(self) -> float:
        return self.covered_by_static / self.total_tokens if self.total_tokens > 0 else 0.0

    @property
    def dynamic_coverage(self) -> float:
        return self.covered_by_dynamic / self.total_tokens if self.total_tokens > 0 else 0.0

    @property
    def oov_rate(self) -> float:
        return self.oov_tokens / self.total_tokens if self.total_tokens > 0 else 0.0


def count_token_frequencies(
    token_sequences: Sequence[Sequence[int]],
    masks: Sequence[Sequence[bool]] | None = None,
) -> Mapping[int, int]:
    """Purely count token frequencies across sequences with optional boolean masks."""
    counter: Counter[int] = Counter()
    if masks is None:
        for seq in token_sequences:
            counter.update(int(tok) for tok in seq)
    else:
        if len(token_sequences) != len(masks):
            raise ValueError(f"Mismatched lengths: {len(token_sequences)} sequences vs {len(masks)} masks")
        for seq, mask in zip(token_sequences, masks):
            if len(seq) != len(mask):
                raise ValueError(f"Mismatched sequence/mask length: {len(seq)} vs {len(mask)}")
            counter.update(int(tok) for tok, keep in zip(seq, mask) if keep)
    return dict(counter)


def _rank_by_frequency_desc(counts: Mapping[int, int], vocab_size: int) -> tuple[int, ...]:
    """Return valid token IDs sorted by descending frequency, breaking ties by ascending token ID."""
    valid_items = [
        (int(tok), int(cnt))
        for tok, cnt in counts.items()
        if cnt > 0 and 0 <= int(tok) < vocab_size
    ]
    valid_items.sort(key=lambda item: (-item[1], item[0]))
    return tuple(tok for tok, _ in valid_items)


def build_static_shortlist(
    vocab_size: int,
    target_size: int,
    assistant_counts: Mapping[int, int],
    all_token_counts: Mapping[int, int],
    pinned_token_ids: Sequence[int] = (),
    metadata: Mapping[str, str] | None = None,
) -> StaticShortlist:
    """Build a deterministic static Top-K shortlist without test leakage.

    Priority tiers:
      1. `pinned_token_ids` (special structural tokens like <think>, </think>, <tool_call>, EOS)
      2. Training assistant-turn tokens ordered by descending frequency
      3. Training full-trajectory tokens ordered by descending frequency
      4. Low-ID BPE merge prior (0, 1, 2, ...) to reach `target_size`
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if target_size > vocab_size:
        raise ValueError(f"target_size ({target_size}) cannot exceed vocab_size ({vocab_size})")

    selected: list[int] = []
    seen: set[int] = set()

    def _extend_unique(candidates: Sequence[int]) -> None:
        for tok in candidates:
            tok_i = int(tok)
            if 0 <= tok_i < vocab_size and tok_i not in seen:
                selected.append(tok_i)
                seen.add(tok_i)
                if len(selected) >= target_size:
                    return

    _extend_unique(pinned_token_ids)
    if len(selected) < target_size:
        _extend_unique(_rank_by_frequency_desc(assistant_counts, vocab_size))
    if len(selected) < target_size:
        _extend_unique(_rank_by_frequency_desc(all_token_counts, vocab_size))
    if len(selected) < target_size:
        for tok_i in range(vocab_size):
            if tok_i not in seen:
                selected.append(tok_i)
                seen.add(tok_i)
                if len(selected) >= target_size:
                    break

    meta_items = tuple(sorted((str(k), str(v)) for k, v in (metadata or {}).items()))
    return StaticShortlist(
        token_ids=tuple(selected),
        vocab_size=vocab_size,
        metadata=meta_items,
    )


def build_dynamic_shortlist(
    static_shortlist: StaticShortlist,
    prompt_ids: Sequence[int],
    static_token_set: frozenset[int] | None = None,
) -> DynamicShortlist:
    """Purely form `StaticShortlist ∪ unique(prompt_ids)` preserving deterministic order."""
    base_set = static_token_set if static_token_set is not None else static_shortlist.token_set
    extra_seen: set[int] = set()
    extra_list: list[int] = []
    for raw_tok in prompt_ids:
        tok = int(raw_tok)
        if 0 <= tok < static_shortlist.vocab_size and tok not in base_set and tok not in extra_seen:
            extra_seen.add(tok)
            extra_list.append(tok)
    extra_tuple = tuple(extra_list)
    return DynamicShortlist(
        static_ids=static_shortlist.token_ids,
        extra_prompt_ids=extra_tuple,
        all_ids=static_shortlist.token_ids + extra_tuple,
    )


def evaluate_shortlist_coverage(
    static_shortlist: StaticShortlist,
    prompt_ids: Sequence[int],
    target_token_ids: Sequence[int],
) -> ShortlistCoverageStats:
    """Purely evaluate static vs dynamic shortlist coverage on held-out target tokens."""
    static_set = static_shortlist.token_set
    dynamic = build_dynamic_shortlist(static_shortlist, prompt_ids, static_token_set=static_set)
    dynamic_set = frozenset(dynamic.all_ids)

    total = len(target_token_ids)
    cov_static = sum(1 for tok in target_token_ids if int(tok) in static_set)
    cov_dynamic = sum(1 for tok in target_token_ids if int(tok) in dynamic_set)
    return ShortlistCoverageStats(
        shortlist_size=dynamic.total_size,
        total_tokens=total,
        covered_by_static=cov_static,
        covered_by_dynamic=cov_dynamic,
        oov_tokens=total - cov_dynamic,
    )


def save_static_shortlist(path: Path, shortlist: StaticShortlist) -> Path:
    """Persist a StaticShortlist cleanly as compressed `.npz`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_keys = np.array([k for k, _ in shortlist.metadata], dtype=np.str_)
    meta_vals = np.array([v for _, v in shortlist.metadata], dtype=np.str_)
    np.savez_compressed(
        path,
        token_ids=np.asarray(shortlist.token_ids, dtype=np.int32),
        vocab_size=np.asarray([shortlist.vocab_size], dtype=np.int64),
        meta_keys=meta_keys,
        meta_vals=meta_vals,
    )
    return path


def load_static_shortlist(path: Path) -> StaticShortlist:
    """Load an immutable StaticShortlist from `.npz`."""
    data = np.load(path)
    token_ids = tuple(int(x) for x in data["token_ids"].tolist())
    vocab_size = int(data["vocab_size"][0])
    keys = data["meta_keys"].tolist() if "meta_keys" in data else []
    vals = data["meta_vals"].tolist() if "meta_vals" in data else []
    metadata = tuple(zip((str(k) for k in keys), (str(v) for v in vals)))
    return StaticShortlist(
        token_ids=token_ids,
        vocab_size=vocab_size,
        metadata=metadata,
    )
