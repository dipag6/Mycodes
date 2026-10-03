"""Run one backend at one size in this process and print one JSON record.

The campaign starts a fresh interpreter for every run so that no state
(compiled code in memory, allocator pools, kernel objects) leaks between
runs. Thread counts are pinned by the caller through the environment.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
import traceback


def rss_mb():
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def peak_rss_mb():
    # ru_maxrss is in KiB on Linux, bytes on macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2**20


def timed(fn, *args):
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn(*args)
    return out, time.perf_counter() - w0, time.process_time() - c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--mode", default="matched", choices=["matched", "native"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--save-spikes", default=None)
    a = ap.parse_args()

    from . import spec
    from .backends import load

    rec = {"backend": a.backend, "n_neurons": a.n, "mode": a.mode,
           "rep": a.rep, "is_warmup": a.rep < 0, "status": "ok",
           "python": platform.python_version()}
    rec["rss_start_mb"] = rss_mb()
    try:
        mod = load(a.backend)
        rec["library"], rec["paradigm"] = mod.LIBRARY, mod.PARADIGM
        vers, rec["import_s"], rec["import_cpu_s"] = timed(mod.versions)
        rec["versions"] = vers
        rec["notes"] = mod.notes(a.mode)
        # input/connectivity generation is shared work, kept out of "build"
        spec.input_spikes()
        spec.connectivity(a.n)
        rec["rss_after_import_mb"] = rss_mb()
        ctx, rec["build_s"], rec["build_cpu_s"] = timed(mod.build, a.n, a.mode)
        _, rec["prepare_s"], rec["prepare_cpu_s"] = timed(mod.prepare, ctx)
        _, rec["simulate_s"], rec["simulate_cpu_s"] = timed(mod.simulate, ctx)
        (steps, neurons), rec["collect_s"], _ = timed(mod.spikes, ctx)
        if hasattr(mod, "extra"):
            rec.update(mod.extra(ctx))
        rec["n_spikes"] = int(len(steps))
        rec["mean_rate_hz"] = len(steps) / a.n / (spec.T_MS / 1000.0)
        rec["n_synapses"] = a.n * spec.K_IN
        rec["n_input_spikes"] = int(len(spec.input_spikes()[0]))
        if a.save_spikes:
            import numpy as np
            np.savez_compressed(a.save_spikes, steps=steps, neurons=neurons)
    except Exception as e:  # recorded, never silently treated as valid
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    rec["peak_rss_mb"] = peak_rss_mb()
    rec["total_s"] = sum(rec.get(k, 0.0) or 0.0 for k in
                         ("build_s", "prepare_s", "simulate_s", "collect_s"))
    rec["cpu_s"] = sum(rec.get(k, 0.0) or 0.0 for k in
                       ("build_cpu_s", "prepare_cpu_s", "simulate_cpu_s"))
    print("@@RESULT@@" + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
