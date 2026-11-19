"""Direction 3: Role-Aware Shallow Prefill & Pretrained/State SVD Diagnostics for Direction 2."""

from __future__ import annotations

import os

from direction3_role_prefill_svd.role_segments import (
    DepthSchedule,
    RoleTokenSpan,
    TokenRole,
    build_depth_schedule,
    build_role_spans,
    parse_role_tags,
)
from direction3_role_prefill_svd.svd_analysis import (
    SingularSpectrum,
    StateErrorDecomposition,
    WeightSinkAlignment,
    analyze_head_weight_sink_alignment,
    compute_singular_spectrum,
    decompose_state_error,
    low_rank_replace_content_error,
)


def ensure_metal_gpu_arch(default_arch: str = "g16g") -> str:
    """Explicitly configure MLX Metal GPU arch if unset (avoids NAX JIT issue on macOS 26.1 / M5)."""
    return os.environ.setdefault("MLX_METAL_GPU_ARCH", default_arch)


__all__ = [
    "DepthSchedule",
    "RoleTokenSpan",
    "SingularSpectrum",
    "StateErrorDecomposition",
    "TokenRole",
    "WeightSinkAlignment",
    "analyze_head_weight_sink_alignment",
    "build_depth_schedule",
    "build_role_spans",
    "compute_singular_spectrum",
    "decompose_state_error",
    "ensure_metal_gpu_arch",
    "low_rank_replace_content_error",
    "parse_role_tags",
]
