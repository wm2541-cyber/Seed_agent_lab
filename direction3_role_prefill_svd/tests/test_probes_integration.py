"""Integration tests for MLX QuantizedLinear dequantization, chunked shallow prefill equivalence, and oracle cache merging."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
import pytest

from collections import defaultdict
from direction3_role_prefill_svd import ensure_metal_gpu_arch
from direction3_role_prefill_svd.role_prefill_probe import (
    _build_surgical_oracle_cache,
    _finalize_policy_metrics,
    _update_bucket_metrics,
)
from direction3_role_prefill_svd.role_segments import DepthSchedule
from direction3_role_prefill_svd.weight_svd_probe import dequantize_linear_weight
from seed_agent_lab.run_seed_coding_eval import _run_chunks_with_shallow_prefill, _score_continuation
from seed_agent_lab.run_swe_rotation_probe import DEFAULT_MODEL_DIR, RunningMetrics


def test_bucket_metrics_aggregation_exact() -> None:
    logits = np.array([[5.0, 1.0, 0.0], [0.0, 6.0, 1.0]], dtype=np.float32)
    buckets: dict[str, RunningMetrics] = defaultdict(RunningMetrics)
    _update_bucket_metrics(buckets, {"all_asst": [0, 1], "reasoning": [0]}, logits, logits)
    sched = DepthSchedule("Test", (), (16, 32), 32, 0.25, 1, (("tool", 1),), 0.5)
    summary = _finalize_policy_metrics("Test", buckets, [sched])
    assert summary.all_asst_n == 2
    assert summary.all_asst_kl < 1e-6
    assert summary.all_asst_top1 == pytest.approx(1.0)
    assert dict(summary.by_category)["reasoning"][0] == 1


def test_dequantize_linear_weight_matches_quantized_matmul() -> None:
    ensure_metal_gpu_arch()
    mx.random.seed(5)
    base = nn.Linear(128, 64, bias=False)
    base.weight = base.weight.astype(mx.bfloat16)
    q_lin = nn.QuantizedLinear.from_linear(base, group_size=64, bits=8)
    mx.eval(q_lin.weight, q_lin.scales, q_lin.biases)

    w_np = dequantize_linear_weight(q_lin)
    assert w_np.shape == (64, 128)

    x = mx.random.normal((4, 128)).astype(mx.bfloat16)
    y_q = np.array(q_lin(x).astype(mx.float32))
    y_d = np.array(x.astype(mx.float32)) @ w_np.T
    np.testing.assert_allclose(y_q, y_d, atol=2e-2, rtol=1e-2)


def test_chunked_shallow_prefill_and_full_oracle_restoration_on_qwen35() -> None:
    ensure_metal_gpu_arch()
    if not DEFAULT_MODEL_DIR.exists():
        pytest.skip("Qwen3.5-4B-MLX-8bit not available")

    model, tokenizer = mlx_lm.load(str(DEFAULT_MODEL_DIR))
    prompt = tokenizer.encode("def add(a, b):\n    return a + b\n\n# Now write multiply:\n")
    cont = tokenizer.encode("def multiply(a, b):\n    return a * b")

    # 1. Splitting a shallow k=16 chunk in half must preserve logits up to native bfloat16 GEMM ULP noise
    mid = len(prompt) // 2
    c_single = _run_chunks_with_shallow_prefill(model, [(prompt, 16)])
    c_split = _run_chunks_with_shallow_prefill(model, [(prompt[:mid], 16), (prompt[mid:], 16)])
    lg_single = _score_continuation(model, c_single, cont)
    lg_split = _score_continuation(model, c_split, cont)
    assert float(np.mean(np.abs(lg_single - lg_split))) < 0.05
    assert float(np.max(np.abs(lg_single - lg_split))) < 0.75
    np.testing.assert_array_equal(np.argmax(lg_single, axis=-1), np.argmax(lg_split, axis=-1))

    # 2. Restoring BOTH AttnKV and FullGDN from exact_cache onto shallow_cache recovers exact 32L logits!
    c_exact = _run_chunks_with_shallow_prefill(model, [(prompt, 32)])
    c_shallow = _run_chunks_with_shallow_prefill(model, [(prompt, 16)])
    c_restored = _build_surgical_oracle_cache(
        model=model,
        shallow_cache=c_shallow,
        exact_cache=c_exact,
        mean_v={},
        mean_a={},
        start_layer=16,
        restore_attn_kv=True,
        restore_full_gdn=True,
        use_mean_sink_b=False,
        oracle_ax_rank=None,
    )
    lg_exact = _score_continuation(model, c_exact, cont)
    lg_restored = _score_continuation(model, c_restored, cont)
    np.testing.assert_allclose(lg_restored, lg_exact, atol=1e-4)
    mx.clear_cache()
