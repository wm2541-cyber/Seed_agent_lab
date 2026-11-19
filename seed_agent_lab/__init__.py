"""Local lab for SEED State Sink analysis and Agent shallow drafting on Qwen3.5-4B."""

import os

# Avoid macOS 26.1 Metal NAX compiler bug on Apple M5 by defaulting to Steel g16g path.
os.environ.setdefault("MLX_METAL_GPU_ARCH", "g16g")
