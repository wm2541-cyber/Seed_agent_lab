# Direction 1: Phase-Aware Routing + Shortlist Vocabulary + `logZ` Calibration

Self-contained implementation, unit/integration tests, training-split artifacts, and Apple Silicon M5 closed-loop benchmark for accelerating `Qwen3.5-4B-MLX-8bit` greedy speculative decoding on `SWE-bench/SWE-smith-trajectories`.

## Folder Structure

```text
direction1_phase_shortlist/
├── __init__.py                 # Public functional API + explicit Metal arch helper
├── phase_router.py             # Pure immutable phase FSM (<think> / response / <tool_call>) & routing presets
├── shortlist_vocab.py          # Pure 4-tier Top-K shortlist builder (zero test leakage) + dynamic prompt union
├── logz_calibrator.py          # Pure streaming Ridge calibrator predicting full-vocab logsumexp (logZ)
├── quantized_head.py           # Bit-exact row-sliced QuantizedEmbedding LM head via mx.quantized_matmul
├── engine.py                   # Closed-loop speculative engine (reuses exact GDN + KVCache rollback)
├── build_artifacts.py          # Builds 4K/8K/16K/32K shortlists & normed_h8 logZ calibrator from 48 train traces
├── benchmark.py                # Held-out offline diagnostics + 10-mode M5 closed-loop wall-clock benchmark
├── README.md                   # This document
├── artifacts/
│   ├── shortlist_{4096,8192,16384,32768}.npz
│   ├── layer8_logz_calibrator.npz
│   ├── benchmark_results.json
│   └── direction1_comparison.png
└── tests/
    ├── test_phase_router.py
    ├── test_shortlist_vocab.py
    ├── test_logz_calibrator.py
    ├── test_quantized_head.py
    └── test_engine.py
```

## Core Design Principles

- **Functional Core, Imperative Shell**: All routing, vocabulary construction, Ridge accumulation/solving, probability calibration, and coverage/metric aggregation are pure functions over `@dataclass(frozen=True)` immutable structures with zero implicit global mutable state. Stateful MLX KV/GDN caches are isolated strictly inside `engine.py`.
- **Zero Test Leakage**: Static shortlists and the `LogZCalibrator` are constructed strictly from the 48 training trajectories (`swe_smith_tool_sample.jsonl[:12]` + `swe_smith_extra_train.jsonl`, `10,908` assistant tokens). All evaluations run on frozen held-out `rows[12:18]`.
- **Bit-Exact Quantized Sub-Head**: `Qwen3.5-4B-MLX-8bit` quantizes `embed_tokens` (`248,320 × 2,560`, `bits=8, group_size=64`) independently along the hidden axis per row. Slicing `(weight, scales, biases)` by shortlist IDs reproduces full `embed_tokens.as_linear` with **`max_abs_diff = 0.0`** while cutting `LM_head` latency **11.5x–20.3x** (`5.575 ms -> 0.484 ms` at `K=8192`).
- **Frozen `normed_h8` for `logZ` Calibration**: Fitting `logZ` on rotated `W_8(normed_h8)` suffers nested out-of-sample norm inflation (`||h||: 145.24 -> 150.93`, held-out `R² = 0.3386`, `bias = +1.57`). Fitting directly on frozen RMSNormed `normed_h8` (`||h8||: 119.14 vs 119.45`) raises held-out **`R²` to `0.8011`** (`MAE = 1.3686`, `bias = +0.3530`).

## Reproducible Commands

Run from the repository root (`seed_agent_lab/`):

```bash
# 1. Run Direction 1 unit + integration tests (14 tests)
MLX_METAL_GPU_ARCH=g16g .venv/bin/pytest direction1_phase_shortlist/tests -v

# 2. Run entire repository test suite (36 tests)
MLX_METAL_GPU_ARCH=g16g .venv/bin/pytest -v

# 3. Rebuild static shortlists & normed_h8 logZ calibrator from 48 training traces
MLX_METAL_GPU_ARCH=g16g .venv/bin/python -m direction1_phase_shortlist.build_artifacts

# 4. Run full held-out offline diagnostics + 10-mode M5 closed-loop benchmark
MLX_METAL_GPU_ARCH=g16g .venv/bin/python -m direction1_phase_shortlist.benchmark --max-new-tokens 160
```

## Verified Apple Silicon M5 Results (`rows[12:18]`, `max_new_tokens=160`)

### Part A — Held-Out Shortlist Coverage & Head Latency (`n = 1,413` assistant tokens)

- `LogZCalibrator` (frozen `normed_h8`, `lambda = 2000`): `MAE = 1.3686`, `RMSE = 2.2574`, `R² = 0.8011`, `bias = +0.3530`
- `K = 8,192` (`mean dynamic size = 8,204.8`): `StaticCov = 97.66%`, **`DynCov = 99.43%`**, `OOV = 0.57%` (`8 / 1,413`), `Head = 0.484 ms vs 5.575 ms` (**`11.5x` faster**)
- `K = 16,384` (`mean dynamic size = 16,392.3`): `StaticCov = 98.23%`, **`DynCov = 99.58%`**, `OOV = 0.42%` (`7 / 1,413`), `Head = 0.673 ms vs 5.575 ms` (**`8.3x` faster**)

### Part B — Closed-Loop M5 Wall-Clock Throughput

| Mode | Exact | Time (s) | tok/s | Speedup | Tau | Draft% | Verify% | Roll% | Neural (think/other) | SubHead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_32L` | 6/6 | 19.50 | 24.00 | 1.00x | 1.00 | 0.0% | 100.0% | 0.0% | 0 / 0 | 0.00 |
| `LookupOnly` | 6/6 | 13.32 | 35.14 | 1.46x | 1.50 | 0.2% | 99.3% | 0.5% | 0 / 0 | 0.00 |
| `Hybrid_FullVocab` | 6/6 | 12.79 | 36.59 | 1.52x | 2.54 | 45.9% | 53.4% | 0.7% | 117 / 289 | 0.00 |
| `PhaseGate_FullVocab` | 6/6 | 11.92 | 39.25 | 1.64x | 2.39 | 33.8% | 65.7% | 0.6% | 0 / 273 | 0.00 |
| `Hybrid_SL8K_Guard` | 6/6 | 11.17 | 41.90 | 1.75x | 2.22 | 29.2% | 69.9% | 0.8% | 98 / 256 | 14.73 |
| **`PhaseGate_SL8K_NoCal`** | **6/6** | **10.49** | **44.63** | **1.86x** | **2.39** | **24.2%** | **75.1%** | **0.6%** | **0 / 273** | **6.65** |
| `PhaseGate_SL8K_Guard` | 6/6 | 11.54 | 40.56 | 1.69x | 2.17 | 21.3% | 77.9% | 0.7% | 0 / 249 | 8.94 |
| `PhaseGate_SL16K_Guard` | 6/6 | 11.45 | 40.86 | 1.70x | 2.17 | 21.7% | 77.6% | 0.7% | 0 / 249 | 38.02 |
| `PhaseAdapt_SL8K_Guard` | 6/6 | 11.18 | 41.84 | 1.74x | 2.21 | 28.5% | 70.7% | 0.8% | 91 / 252 | 11.91 |
| `PhaseGate_SL8K_Strict` | 5/6* | 12.10 | 38.85 | 1.62x | 1.81 | 16.3% | 83.1% | 0.6% | 0 / 209 | 11.03 |

*Note: Single non-exact match in `PhaseGate_SL8K_Strict` arises from native `bfloat16` `S=1` vs `S>1` 1-ULP tie-breaking when aggressive `d=0.0` calibration shortens a verify chunk right at a zero-margin tie; all `NoCal` and `Guard` shortlist modes achieved `6/6` exact greedy match.*

## Key Takeaways

1. **Go-Criterion Exceeded (`1.86x` overall, `4/6` prompts `> 2.0x`)**: `PhaseGate_SL8K_NoCal` reaches **`44.63 tok/s` (`1.86x`)** with **`6/6` exact match**, peaking at **`2.41x` (`55.84 tok/s`)** on Row 14 and fixing long-`<think>` Row 16 from `1.22x` (`Hybrid_FullVocab`) up to `1.49x`.
2. **Phase-Aware Routing Isolates Free Gain**: Simply skipping Layer-8 neural draft inside `<think>...</think>` (`PhaseGate_FullVocab`) eliminates `117` wasted thinking draft steps and jumps throughput from `1.52x -> 1.64x`.
3. **Why `NoCal` Beat `logZ` Calibration on `Top-8K ∪ unique(prompt_ids)`**: Dynamic prompt union already achieves **`99.43%`** token coverage (`OOV = 0.57%`, only `8` tokens out of `1,413`, and naive false high confidence affects only `4 / 1,413 = 0.28%` of steps). Even an unbiased `R² = 0.80` linear `logZ` proxy (`MAE = 1.37`) loses more speed from false early-stops on in-vocab tokens (`8.8%` recall drop with `Guard`, `33.6%` with `Strict`) than it saves on the rare `0.28%` OOV events.
