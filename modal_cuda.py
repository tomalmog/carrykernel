"""Modal runner for the CUDA fused GDN-state kernel: correctness + benchmark.

Cheap (no model download): just torch + nvcc and a few MB of synthetic state.

Unlike `modal_kernel.py`, this needs a real CUDA toolkit to compile the `.cu`
(the pip torch wheel ships runtime libraries but no `nvcc`), so the image is
built from an `nvidia/cuda:*-devel` base.

Usage:
    modal run modal_cuda.py              # correctness, then benchmark
    modal run modal_cuda.py --mode test  # correctness only
    modal run modal_cuda.py --mode bench --steps 2000
"""

import modal

app = modal.App("carrykernel-cuda")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "ninja", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.6"})
    .add_local_dir(".", remote_path="/root/repo", ignore=[".git", "__pycache__", "*.pyc"])
)


@app.function(gpu="A10G", image=image, timeout=3600)
def run_cuda(mode: str = "all", steps: int = 1000, warmup: int = 200):
    import sys
    sys.path.insert(0, "/root/repo")

    import torch
    print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0),
          "| capability", torch.cuda.get_device_capability(0))

    out = {}
    if mode in ("all", "test"):
        print("\n" + "=" * 70)
        print("CORRECTNESS: CUDA kernel vs reference.py oracle / Triton")
        print("=" * 70)
        from experiments import test_cuda_kernel
        rc = test_cuda_kernel.main()
        out["test_rc"] = rc
        assert rc == 0, "CUDA correctness test FAILED"

    if mode in ("all", "bench"):
        print("\n" + "=" * 70)
        print("BENCHMARK: CUDA vs Triton vs FP32")
        print("=" * 70)
        from experiments import cuda_bench
        out["bench"] = cuda_bench.run(gpu_name="A10G", steps=steps, warmup=warmup)

    return out


@app.local_entrypoint()
def main(mode: str = "all", steps: int = 1000, warmup: int = 200):
    res = run_cuda.remote(mode=mode, steps=steps, warmup=warmup)
    bench = res.get("bench")
    if bench:
        print("\n=== summary (us/token) ===")
        for key, r in bench.items():
            if key.startswith("_"):
                continue
            tri = f"{r['triton_us']:.1f}" if r.get("triton_us") else "n/a"
            print(f"{key:<16} cuda={r['cuda_us']:.1f}  triton={tri}  "
                  f"MB={r['MB']:.2f}")
