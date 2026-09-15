"""Modal runner for the vLLM-layout INT8+EF packed-decode kernel (Stage A).

Pure Triton (no nvcc needed) and no model download, so this is a cheap run.
Validates the kernel written against vLLM's vendored
``fused_recurrent_gated_delta_rule_packed_decode`` contract -- the [V, K]
layout, the paged ``ssm_state_indices`` pool, and NULL_BLOCK_ID padding --
against the same oracle the rest of the project uses.

Usage:
    modal run modal_vllm_stage_a.py
"""

import modal

app = modal.App("carrykernel-vllm-stage-a")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "numpy")  # torch pulls a matching triton on linux
    .add_local_dir(".", remote_path="/root/repo", ignore=[".git", "__pycache__", "*.pyc"])
)


@app.function(gpu="A10G", image=image, timeout=3600)
def run_test():
    import sys
    sys.path.insert(0, "/root/repo")

    import torch
    print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0))

    from vllm_integration import test_quantized_packed_decode as t
    rc = t.main()
    assert rc == 0, "vLLM Stage A correctness test FAILED"
    return rc


@app.local_entrypoint()
def main():
    print("rc =", run_test.remote())
