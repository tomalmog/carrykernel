"""Deploy the int4-residual benchmark and spawn it server-side, then exit.

Two attempts with `modal run --detach` were killed mid-run by a cancellation
signal (run 1 at ~12 min, run 2 at ~4.7 h). The local launcher process was still
alive both times, so a reaped parent does not explain it; what both shared was a
live client attached to an *ephemeral* app.

A **deployed** app is persistent and server-side: it does not belong to any
client session, so no client disconnect or cancellation can reach it.
`Function.spawn()` then queues the call and returns a handle immediately.

Usage:
    python spawn_int4resid.py           # 50 problems/arm (default, ~2-3 h)
    python spawn_int4resid.py 100       # 100 problems/arm (~4-6 h)

Then poll with:
    modal app logs carrykernel-int4resid
"""

import subprocess
import sys

import modal

n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

# Deploy (idempotent: re-deploying replaces the app's functions).
print(f"deploying carrykernel-int4resid ...", flush=True)
rc = subprocess.run(
    [sys.executable, "-m", "modal", "deploy", "modal_bench_int4resid.py"],
    capture_output=True, text=True)
print(rc.stdout[-2000:])
if rc.returncode != 0:
    print(rc.stderr[-2000:])
    raise SystemExit(f"deploy failed: rc={rc.returncode}")

fn = modal.Function.from_name("carrykernel-int4resid", "run")
handle = fn.spawn(n_gsm=n, n_mmlu=n)
print(f"\nSPAWNED  call id: {handle.object_id}   n_gsm={n} n_mmlu={n}")
print("Poll with:  modal app logs carrykernel-int4resid")
