# Upgraded Direction 2: 4-Layer Deep Full-Attention KV + Frozen-`b` Rank-`r` `ax` Prefill Alignment

Self-contained implementation, unit/integration tests, closed-form shared-Gram Ridge training pipeline, and Apple Silicon M5 wall-clock Prefill benchmark built directly from the [Direction 3 + SVD Blueprint](../direction3_role_prefill_svd/README.md).

## Folder Structure

```text
direction2_hybrid_prefill_align/
├── __init__.py               # Pure functional API exports + explicit Metal arch helper
├── streaming_ridge.py        # Pure immutable shared-Gram Cholesky Ridge, SVD low-rank factorization, sink projection
├── kv_gdn_aligner.py         # Pure immutable AttnKVMap / GDNHeadValueMap / HybridPrefillAligner + .npz serialization
├── aligned_prefill.py        # Fast MLX bfloat16 aligned shallow-prefill engine (4-layer Attn KV + 11-layer GDN)
├── train_aligners.py         # Single-pass shared-Gram training over 48 SWE train traces (zero test leakage)
├── eval_and_benchmark.py     # Held-out rows[12:18] (max_tokens=6144) KL/Top-1 + real M5 wall-clock Prefill benchmark
├── README.md                 # This document
├── artifacts/
│   ├── hybrid_aligner_L16_r16.npz
│   ├── direction2_eval_results.json
│   └── direction2_hybrid_align.png
└── tests/
    ├── test_streaming_ridge.py
    ├── test_kv_gdn_aligner.py
    └── test_aligned_prefill.py
```

## Reproducible Commands

Run from the repository root (`seed_agent_lab/`):

```bash
# 1. Run Direction 2 unit + integration tests (6 tests)
MLX_METAL_GPU_ARCH=g16g .venv/bin/pytest direction2_hybrid_prefill_align/tests -v

# 2. Fit all 30 alignment maps (4 Attn K/V + 11 GDN V/K) via single shared-Gram Cholesky on 48 train traces
MLX_METAL_GPU_ARCH=g16g .venv/bin/python -m direction2_hybrid_prefill_align.train_aligners \
  --anchor-layer 16 --max-tokens 2560 --ridge-lambda 200.0 --gdn-rank 16

# 3. Evaluate KL/Top-1 & real M5 wall-clock Prefill speedup on held-out rows[12:18] (21,009 prefix tokens)
MLX_METAL_GPU_ARCH=g16g .venv/bin/python -m direction2_hybrid_prefill_align.eval_and_benchmark \
  --eval-max-tokens 6144 --max-chunk-len 1024
```

## Verified Apple Silicon M5 Results (`rows[12:18]`, `21,009` Total Prefix Tokens, `436` Continuation Tokens)

Trained strictly on the 48 SWE training traces (`122,880` tokens; single shared-Gram Cholesky solve took **`14.19s`** with zero SGD).

| Mode | Prefill Time (s) | Prefill tok/s | Real M5 Speedup | KL vs 32L | Top-1 | Top-5 | `reasKL` | `codeKL` | `toolKL` | `echoKL` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Full_32L` | 49.01 | 428.7 | 1.00x | 0.0000 | 100.0% | 100.0% | 0.000 | 0.000 | 0.000 | 0.000 |
| `Uniform_k24_Raw` | 40.54 | 518.2 | 1.21x | 0.1035 | 96.8% | 99.5% | 0.034 | 0.000 | 0.183 | 0.013 |
| `Uniform_k16_Raw` | 30.32 | 692.9 | 1.62x | 0.7292 | 82.6% | 95.2% | 0.486 | 0.186 | 0.966 | 0.635 |
| `U16_GDN_FreezeB_Ax_Only` | 31.65 | 663.8 | 1.55x | 0.7341 | 82.8% | 94.7% | 0.521 | 0.072 | 0.956 | 0.637 |
| **`U16_AttnKV_FullRidge`** | **30.35** | **692.3** | **1.61x** | **0.0667** | **96.6%** | **100.0%** | **0.138** | **0.002** | **0.003** | **0.130** |
| **`U16_AttnKV_SVD_R256`** | **30.22** | **695.3** | **1.62x** | **0.0900** | **95.2%** | **100.0%** | **0.170** | **0.003** | **0.011** | **0.198** |
| `U16_AttnKV + FreezeB_Ax_R8` | 30.25 | 694.6 | 1.62x | 0.0703 | 95.6% | 100.0% | 0.148 | 0.002 | 0.003 | 0.129 |
| `U16_AttnKV + FreezeB_Ax_R16` | 30.72 | 684.0 | 1.60x | 0.0703 | 96.3% | 100.0% | 0.146 | 0.002 | 0.003 | 0.134 |
| `U16_AttnKV + FreezeB_AxK_R16` | 31.01 | 677.4 | 1.58x | 0.0667 | 96.6% | 100.0% | 0.137 | 0.002 | 0.003 | 0.131 |
| `U16_AttnKV + FitFullV_R16` | 30.56 | 687.4 | 1.60x | 0.0690 | 96.3% | 100.0% | 0.145 | 0.002 | 0.003 | 0.126 |
| `Skip_AllTool_k16_Raw` | 37.90 | 554.3 | 1.29x | 0.0332 | 96.8% | 100.0% | 0.038 | 0.000 | 0.001 | 0.170 |
| **`Skip_AllTool_k16 + D2_AttnKV`** | **39.33** | **534.2** | **1.25x** | **0.0196** | **97.5%** | **100.0%** | **0.027** | **0.000** | **0.001** | **0.082** |
| **`Skip_AllTool_k16 + D2_Full`** | **39.44** | **532.7** | **1.24x** | **0.0202** | **97.7%** | **100.0%** | **0.028** | **0.001** | **0.002** | **0.081** |

## Key Takeaways

1. **10.9x KL Reduction at Zero Extra Latency (`1.61x` Real M5 Speedup)**: Aligning ONLY the 4 deep Full-Attention KV layers (`L19, L23, L27, L31`) via closed-form Ridge slashes `Uniform_k16` KL from **`0.7292 -> 0.0667`** (`Top-1: 82.6% -> 96.6%`, `Top-5: 100.0%`, `codeKL: 0.002`, `toolKL: 0.003`) while running in **`30.35s` vs `30.32s`** (`Uniform_k16_Raw`)—**33% faster AND 1.55x lower KL than `Uniform_k24_Raw`** (`40.54s`, `KL = 0.1035`).
2. **SVD Compression (`U16_AttnKV_SVD_R256`)**: Truncating the 4 Attn KV maps to Rank-256 runs at **`1.62x` (`30.22s`, faster than raw `Uniform_k16`)** while STILL beating `Uniform_k24_Raw` (`KL = 0.0900 vs 0.1035`, `Top-5 = 100.0%`).
3. **Direction 2 + Direction 3 Synergy**: Applying `D2_AttnKV` only to `<tool_response>` chunks cuts `Skip_AllTool_k16`'s sole remaining `repeated_echo` KL in half (`0.170 -> 0.082`), reaching **`KL = 0.0196`, `Top-1 = 97.5%–97.7%`, `Top-5 = 100.0%`**.
4. **Why GDN `FreezeB_FitAx` Didn't Add More Gain on Top of Linear `AttnKV`**: As Probe 2B predicted, raw GDN on `Uniform_k16` only accounts for `0.0088` KL; the remaining `0.0667` KL is predominantly the linear-map residual of the 4 deep Attn KV layers (`0.0667` vs `0.0088` oracle), and token-level per-head GDN `v` is high-rank (`56.3%` energy at Rank-16, whereas sequence-accumulated state `E_ax` is Rank-9).
