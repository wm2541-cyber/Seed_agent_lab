"""Unified End-to-End Inference Pipeline: Direction 2/3 Aligned Shallow Prefill + Direction 1 Phase-Shortlist Decode."""

from __future__ import annotations

import os

from unified_e2e_pipeline.pipeline import (
    UnifiedE2EResult,
    UnifiedPipelineConfig,
    UnifiedSpeculativePipeline,
    baseline_e2e_config,
    decode_only_d1_config,
    prefill_only_d2_config,
    unified_d23_d1_config,
    unified_d2_d1_config,
)


def ensure_metal_gpu_arch(default_arch: str = "g16g") -> str:
    """Explicitly configure MLX Metal GPU arch if unset (avoids NAX JIT issue on macOS 26.1 / M5)."""
    return os.environ.setdefault("MLX_METAL_GPU_ARCH", default_arch)


__all__ = [
    "UnifiedE2EResult",
    "UnifiedPipelineConfig",
    "UnifiedSpeculativePipeline",
    "baseline_e2e_config",
    "decode_only_d1_config",
    "ensure_metal_gpu_arch",
    "prefill_only_d2_config",
    "unified_d23_d1_config",
    "unified_d2_d1_config",
]
