"""Numerical defaults for paired FP32 framework comparisons."""
import jittor as jt

def configure_fp32():
    # CUDA fast-math substitutes approximate transcendental functions.
    # Official PyTorch FP32 tanh uses the precise implementation.
    if hasattr(jt.flags, "nvcc_flags"):
        jt.flags.nvcc_flags=jt.flags.nvcc_flags.replace("--use_fast_math","")
