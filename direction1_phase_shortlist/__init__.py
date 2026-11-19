"""Direction 1: Phase-Aware Routing + Shortlist Vocabulary + logZ Calibration for Closed-Loop Speculative Decoding."""

from __future__ import annotations

import os

from direction1_phase_shortlist.logz_calibrator import (
    LogZCalibrator,
    LogZMetrics,
    StreamingLogZAccumulator,
    calibrated_max_prob_numpy,
    evaluate_logz_calibrator,
    extract_logz_features,
    init_logz_accumulator,
    load_logz_calibrator,
    predict_logz_numpy,
    save_logz_calibrator,
    solve_logz_calibrator,
    update_logz_accumulator,
)
from direction1_phase_shortlist.phase_router import (
    GenerationPhase,
    PhaseDraftPolicy,
    PhaseTagIds,
    RoutingConfig,
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
from direction1_phase_shortlist.shortlist_vocab import (
    DynamicShortlist,
    ShortlistCoverageStats,
    StaticShortlist,
    build_dynamic_shortlist,
    build_static_shortlist,
    count_token_frequencies,
    evaluate_shortlist_coverage,
    load_static_shortlist,
    save_static_shortlist,
)


def ensure_metal_gpu_arch(default_arch: str = "g16g") -> str:
    """Explicitly configure MLX Metal GPU arch if unset (avoids NAX JIT issue on macOS 26.1 / M5)."""
    return os.environ.setdefault("MLX_METAL_GPU_ARCH", default_arch)


__all__ = [
    "DynamicShortlist",
    "GenerationPhase",
    "LogZCalibrator",
    "LogZMetrics",
    "PhaseDraftPolicy",
    "PhaseTagIds",
    "RoutingConfig",
    "ShortlistCoverageStats",
    "StaticShortlist",
    "StreamingLogZAccumulator",
    "advance_phase",
    "build_dynamic_shortlist",
    "build_static_shortlist",
    "calibrated_max_prob_numpy",
    "count_token_frequencies",
    "ensure_metal_gpu_arch",
    "evaluate_logz_calibrator",
    "evaluate_shortlist_coverage",
    "extract_logz_features",
    "extract_phase_tag_ids",
    "infer_phase_from_tokens",
    "init_logz_accumulator",
    "load_logz_calibrator",
    "load_static_shortlist",
    "lookup_only_routing",
    "phase_adaptive_routing",
    "phase_gated_routing",
    "policy_for_phase",
    "predict_logz_numpy",
    "save_logz_calibrator",
    "save_static_shortlist",
    "solve_logz_calibrator",
    "step_phase",
    "uniform_hybrid_routing",
    "update_logz_accumulator",
]
