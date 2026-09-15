"""Modal runner for the fused GDN-state kernel benchmark (Triton on GPU).

Cheap: no model download, just torch + triton and a few MB of synthetic state.
Runs the FP32-vs-INT8+EF kernel benchmark from `experiments/kernel_bench.py`.

The local repo (`statequant` + `experiments`) is mounted into the container at
`/root/repo` and added to `sys.path` so both packages are importable at runtime.

Usage:
    modal run modal_kernel.py --steps 500
"""

import modal

app = modal.App("statequant-kernel-bench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch")  # torch pulls a compatible triton on linux
    .add_local_dir(".", remote_path="/root/repo", ignore=[".git", "__pycache__"])
)


@app.function(gpu="A10G", image=image, timeout=1800)
def run_bench(BLOCK_V: int = 32, steps: int = 1000):
    import sys
    sys.path.insert(0, "/root/repo")

    import torch
    import triton
    from experiments import kernel_bench

    print("torch", torch.__version__, "triton", triton.__version__,
          "cuda", torch.cuda.get_device_name(0))

    results = kernel_bench.run(gpu_name="A10G", BLOCK_V=BLOCK_V, steps=steps)
    return results


@app.local_entrypoint()
def main(block_v: int = 32, steps: int = 1000):
    print(run_bench.remote(BLOCK_V=block_v, steps=steps))
