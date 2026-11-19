# Direction 3 & Direction 2 Prep: Role-Aware Shallow Prefill + State/Weight SVD Blueprint

Self-contained probe suite verifying the **"Bash/Tool Output Information Dilution"** hypothesis on `Qwen3.5-4B-MLX-8bit` + `SWE-bench/SWE-smith-trajectories` (`rows[12:18]`, `max_tokens=6144`), coupled with full 32-layer pretrained weight SVD, deep GDN state-error (`E_b` vs `E_ax`) SVD, and causal oracle attribution to guide **Direction 2**.

## Folder Structure

```text
direction3_role_prefill_svd/
├── __init__.py               # Pure API exports + explicit Metal arch helper
├── role_segments.py          # Pure Qwen3.5 role parser (system/user/assistant/tool) & matched-budget depth schedules
├── svd_analysis.py           # Pure SVD spectra, Pythagorean state-error (E_b vs E_ax) split, & weight-sink alignment
├── role_prefill_probe.py     # Probe 1 (Role-Aware Prefill), Probe 2A (State-Error SVD), Probe 2B (Causal Oracles)
├── weight_svd_probe.py       # Probe 3 (Full 32-layer dequantized GDN + Full-Attn weight SVD & Sink alignment)
├── run_probes.py             # End-to-end CLI runner generating JSON report + 6-panel diagnostic figure
├── README.md                 # This document
├── artifacts/
│   ├── direction3_svd_report.json
│   └── direction3_role_prefill_svd.png
└── tests/
    ├── test_role_segments.py
    ├── test_svd_analysis.py
    └── test_probes_integration.py
```

## Reproducible Commands

Run from the repository root (`seed_agent_lab/`):

```bash
# 1. Run Direction 3 unit + integration tests (8 tests)
MLX_METAL_GPU_ARCH=g16g .venv/bin/pytest direction3_role_prefill_svd/tests -v

# 2. Run full three-probe diagnostic suite (rows[:12] sink train, rows[12:18] held-out eval)
MLX_METAL_GPU_ARCH=g16g .venv/bin/python -m direction3_role_prefill_svd.run_probes \
  --n-train 12 --train-max-tokens 2048 --eval-max-tokens 6144 --max-chunk-len 1024
```

## Verified Key Results (`rows[12:18]`, `n = 436` held-out assistant continuation tokens)

### Probe 1 — Role-Aware Shallow Prefill vs Matched-Budget Counterfactuals

In multi-turn SWE trajectories (`max_tokens=6144`), `<tool_response>` logs account for **`21.9%–78.7%`** of the prompt (`Row 13: 74.2%`, `Row 15: 70.2%`, `Row 16: 78.7%`).

| Policy | Skipped Layers | Mean Skipped Tokens | KL vs 32L | Top-1 | Top-5 | `reasoning` KL/T1 | `new_code_arg` KL/T1 | `tool_boilerplate` KL/T1 | `repeated_echo` KL/T1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Uniform_k16` | 50.0% | 3501.5 | 0.7292 | 82.6% | 95.2% | 0.486 / 86.3% | 0.186 / 92.3% | 0.966 / 79.4% | 0.635 / 81.2% |
| `Uniform_k24` | 25.0% | 3501.5 | 0.1035 | 96.8% | 99.5% | 0.034 / 96.3% | 0.000 / 100.0% | 0.183 / 96.3% | 0.013 / 100.0% |
| `Adaptive_old16_recent24` | 47.5% | 3501.5 | 0.6292 | 84.6% | 95.4% | 0.316 / 88.8% | 0.190 / 92.3% | 0.924 / 80.4% | 0.483 / 87.5% |
| **`Skip_AllTool_k16`** | **25.4%** | **2218.7** | **0.0332** | **96.8%** | **100.0%** | **0.038 / 94.4%** | **0.000 / 100.0%** | **0.001 / 99.5%** | **0.170 / 91.7%** |
| **`Skip_OldTool_k16`** | **15.1%** | **1448.0** | **0.0031** | **99.8%** | **100.0%** | **0.005 / 100.0%** | **0.000 / 100.0%** | **0.000 / 100.0%** | **0.013 / 97.9%** |
| **`Skip_AllTool_k20`** | **19.0%** | **2218.7** | **0.0128** | **99.1%** | **100.0%** | **0.019 / 98.1%** | **0.000 / 100.0%** | **0.000 / 100.0%** | **0.050 / 97.9%** |
| `Matched_SkipTool_k16` | **13.2%** | **899.0** | **0.0308** | **97.0%** | **100.0%** | 0.032 / 94.4% | 0.001 / 100.0% | 0.000 / 100.0% | 0.170 / 91.7% |
| `Matched_SkipUserAsst_k16` | **13.2%** | **899.0** | **0.4270** | **91.3%** | **96.8%** | 0.274 / 93.2% | 0.005 / 100.0% | 0.662 / 87.4% | 0.006 / 100.0% |
| `Matched_SkipUserAsstRecent_k16` | **13.2%** | **899.0** | **0.4546** | **90.4%** | **96.8%** | 0.346 / 90.7% | 0.005 / 100.0% | 0.663 / 87.4% | 0.009 / 100.0% |

- At **identical FLOPs skipped (`13.2%`, `899.0` tokens)**, shallow-prefilling `USER/ASSISTANT` is **`13.9x–14.8x` worse in KL** (`0.4270–0.4546` vs `0.0308`) than shallow-prefilling `TOOL` outputs.
- At **`25.4%` layer compute saved**, `Skip_AllTool_k16` (`KL = 0.0332`) is **`3.1x` more accurate** than `Uniform_k24` (`KL = 0.1035`).
- Keeping only the most recent `<tool_response>` at 32L (`Skip_OldTool_k16`) achieves **`KL = 0.0031` (`99.8%` Top-1, `100.0%` Top-5)** with zero training.

### Probe 2B — Causal Oracle Attribution for Direction 2 (Crucial Finding!)

| Intervention on `Uniform_k16` | KL vs 32L | Top-1 | Top-5 |
| :--- | :---: | :---: | :---: |
| `Uniform_k16` (raw baseline) | 0.7292 | 82.6% | 95.2% |
| `Oracle_GDN_Only` (restore ALL 12 deep GDN states `L16..L30`, leave 4 Attn KV shallow) | 0.6342 | 84.4% | 95.4% |
| **`Oracle_AttnKV_Only`** (restore ONLY the 4 deep Full-Attn KV caches `L19,23,27,31`, leave 12 GDN shallow!) | **0.0088** | **98.9%** | **100.0%** |
| `Uniform_k16 + MeanSink_b + ax_R16` (without Attn KV) | 0.6557 | 84.2% | 95.4% |
| **`Oracle_AttnKV + MeanSink_b + ax_R8`** | **0.0023** | **99.8%** | **100.0%** |
| **`Oracle_AttnKV + MeanSink_b + ax_R16`** | **0.0017** | **99.5%** | **100.0%** |

**Critical Takeaway for Direction 2**: **`98.8%` of `Uniform_k16`'s KL loss (`0.7292 -> 0.0088`) originates in the 4 deep Full-Attention KV caches (`L19, L23, L27, L31`)**, NOT in the 12 deep GDN layers (`Oracle_GDN_Only` hits a hard ceiling at `KL = 0.6342`). Once the 4 deep Attn KV layers are aligned, anchoring `b = mean_b` and learning a tiny **Rank-8** `ax` correction finishes the remaining GDN residual (`0.0088 -> 0.0023`, `99.8%` Top-1).

### Probe 2A & Probe 3 — State-Error & Pretrained Weight SVD Blueprint

- **Deep GDN State Error (`Uniform_k16`, `L16..L30`)**: `ContentShare = 58.7%`, `SinkShare = 32.9%`, **`E_ax effRank = 9.2`**, `r80 = 6.2`, `r90 = 10.4`, `r95 = 15.2` (out of 128) -> confirms **Rank 8–16** suffices for deep GDN `ax` alignment.
- **Pretrained Weight SVD across all 32 layers**:
  - **Deep Full-Attention (`L19, 23, 27, 31`) has the LOWEST-RANK KV projections in the entire model**: `W_K effRank = 394.8 (r90 = 384)`, `W_V effRank = 569.1 (r90 = 505)` (dropping sharply from shallow `555.1 / 666.6`).
  - **Deep GDN (`L16..L30`) weights are much higher-rank**: `W_V effRank = 1626.2 (r90 = 1472)`, `W_out effRank = 1606.5 (r90 = 1451)`, per-head `W_V effRank = 113.9 / 128`.
  - **Rank-1 Sink vs Weight Subspace**: Empirical sink `v_hat` strongly overlaps `W_V`'s top-4 left singular subspace (`47.2%` shallow / `42.5%` deep vs `3.1%` random), while `W_out` attenuates it (`gain = 0.287–0.360`) and removing `v_hat` from `W_V` preserves weight rank (`113.9 -> 113.7`), confirming the Rank-1 sink is an activation-accumulation mode rather than weight collapse.
