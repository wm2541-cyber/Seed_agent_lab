"""Upgraded Direction 2: 4-Layer Deep Full-Attention KV Alignment + Frozen-b / Rank-r ax GDN Alignment."""

from __future__ import annotations

import os

from direction2_hybrid_prefill_align.kv_gdn_aligner import (
    AttnKVMap,
    GDNHeadValueMap,
    HybridPrefillAligner,
    load_hybrid_aligner,
    save_hybrid_aligner,
)
from direction2_hybrid_prefill_align.streaming_ridge import (
    AffineLinearMap,
    LowRankAffineMap,
    SharedGramAccumulator,
    apply_affine_numpy,
    apply_low_rank_numpy,
    factorize_affine_svd,
    init_shared_gram,
    project_out_head_sinks,
    solve_shared_gram_ridge,
    update_shared_gram,
)


def ensure_metal_gpu_arch(default_arch: str = "g16g") -> str:
    """Explicitly configure MLX Metal GPU arch if unset (avoids NAX JIT issue on macOS 26.1 / M5)."""
    return os.environ.setdefault("MLX_METAL_GPU_ARCH", default_arch)


__all__ = [
    "AffineLinearMap",
    "AttnKVMap",
    "GDNHeadValueMap",
    "HybridPrefillAligner",
    "LowRankAffineMap",
    "SharedGramAccumulator",
    "apply_affine_numpy",
    "apply_low_rank_numpy",
    "ensure_metal_gpu_arch",
    "factorize_affine_svd",
    "init_shared_gram",
    "load_hybrid_aligner",
    "project_out_head_sinks",
    "save_hybrid_aligner",
    "solve_shared_gram_ridge",
    "update_shared_gram",
]
