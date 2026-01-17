# ================================
# File: src/utils/gpu_diagnostics.py
# ================================
"""
GPU diagnostics utility.
Run directly or import assert_cuda_ready().
"""

import subprocess
import torch


def assert_cuda_ready() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    _ = torch.empty((1,), device="cuda")
    torch.cuda.synchronize()


def print_diagnostics() -> None:
    print("=== GPU DIAGNOSTICS ===")
    print(f"Torch Version: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hwaccels"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        print("FFmpeg HW Accels:")
        print(out.strip())
    except Exception:
        print("FFmpeg not found or no hwaccels available")

    print("=======================")


if __name__ == "__main__":
    assert_cuda_ready()
    print_diagnostics()
