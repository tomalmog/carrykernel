"""Nsight Compute profiling of the CUDA fused GDN-state kernel.

Runs `ncu` over a short driver that launches the FP32 baseline and the INT8+EF
kernel, and reports the metrics that matter for a memory-bound kernel:

  * dram__bytes_read / dram__bytes_write   -- actual HBM traffic (vs our model)
  * gpu__time_duration                     -- kernel time
  * l1tex / lts hit rates                  -- whether the SMEM staging works
  * sm__warps_active                       -- achieved occupancy
  * smsp__inst_executed                    -- instruction count (quantize cost)

Usage:
    modal run modal_nsight.py
    modal run modal_nsight.py --batch 8
"""

import modal

app = modal.App("carrykernel-nsight")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cuda-nsight-compute-12-4")
    .pip_install("torch", "ninja", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.6"})
    .add_local_dir(".", remote_path="/root/repo", ignore=[".git", "__pycache__", "*.pyc"])
)

DRIVER = r'''
import sys
sys.path.insert(0, "/root/repo")
import torch
from statequant import kernel as tk
from cuda import binding

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HV, K, V = 32, 128, 128
dev = "cuda"
f32 = dict(device=dev, dtype=torch.float32)
torch.manual_seed(0)

q = torch.randn(B, HV, K, **f32)
k = torch.randn(B, HV, K, **f32); k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
v = torch.randn(B, HV, V, **f32)
alpha = torch.full((B, HV), 0.995, **f32)
beta = torch.rand(B, HV, **f32).sigmoid()
h0 = torch.randn(B, HV, K, V, **f32) * 0.1

binding.load_extension()

# FP32 baseline
h = h0.contiguous()
for _ in range(3):
    h, _ = binding.gdn_fp32_step_cuda(h, q, k, v, alpha, beta)

# INT8 + EF (int8 residual: the configuration we actually advocate)
h_i, sc, e, es = tk.init_quant_state(h0, 8, tk.RES_INT8)
for _ in range(3):
    _, h_i, sc, e, es = binding.gdn_quant_step_cuda(
        h_i, sc, e, es, q, k, v, alpha, beta, 8, tk.RES_INT8)
torch.cuda.synchronize()
print("driver done")
'''


@app.function(gpu="A10G", image=image, timeout=3600)
def profile(batch: int = 8):
    import subprocess, os, sys
    sys.path.insert(0, "/root/repo")
    os.makedirs("/root/work", exist_ok=True)
    with open("/root/work/driver.py", "w") as f:
        f.write(DRIVER)

    # Warm the JIT build first, outside the profiler, so ncu profiles the
    # kernels and not nvcc.
    subprocess.run([sys.executable, "/root/work/driver.py", str(batch)],
                   check=True, cwd="/root/work")

    metrics = ",".join([
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "smsp__inst_executed.sum",
        "launch__occupancy_limit_shared_mem",
    ])
    cmd = [
        "ncu", "--target-processes", "all",
        "--kernel-name", "regex:gdn_.*_kernel",
        "--launch-skip", "2", "--launch-count", "2",
        # Modal containers cannot lock GPU clocks (no privileged nvidia-smi), so
        # profile at whatever clocks the driver gives us. Absolute times then
        # carry some clock variance; the ratios and the byte counts do not.
        "--clock-control", "none",
        "--metrics", metrics,
        "--csv",
        sys.executable, "/root/work/driver.py", str(batch),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/work")
    print("=== ncu stdout ===")
    print(r.stdout[-20000:])
    if r.returncode != 0:
        print("=== ncu stderr ===")
        print(r.stderr[-6000:])
    return {"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-4000:]}


@app.local_entrypoint()
def main(batch: int = 8):
    profile.remote(batch=batch)
