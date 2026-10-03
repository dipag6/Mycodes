"""Run the pilot campaign: timing (matched mode, warm-up + repeats) and a
single native-mode run per configuration for the fidelity comparison.

Every run is a fresh interpreter (see run_one.py). Runs are interleaved
across backends in a seeded random order inside each size so that slow
drift in the machine's state does not line up with one backend.

    python -m pilot.campaign --out data/pilot
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np

from . import fidelity, spec
from .backends import ALL

THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1"}


def machine_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "logical_cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        pass
    info["gpu"] = "none"
    info["power_telemetry"] = ("none (no RAPL powercap interface and no GPU "
                               "in this container)")
    return info


def run_once(backend, n, mode, rep, spikes_path, timeout):
    env = dict(os.environ, **THREAD_ENV)
    cmd = [sys.executable, "-m", "pilot.run_one", "--backend", backend,
           "--n", str(n), "--mode", mode, "--rep", str(rep)]
    if spikes_path:
        cmd += ["--save-spikes", spikes_path]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=timeout)
        for line in p.stdout.splitlines():
            if line.startswith("@@RESULT@@"):
                return json.loads(line[len("@@RESULT@@"):])
        return {"backend": backend, "n_neurons": n, "mode": mode, "rep": rep,
                "is_warmup": rep < 0, "status": "crash",
                "error": (p.stderr or p.stdout)[-1500:]}
    except subprocess.TimeoutExpired:
        return {"backend": backend, "n_neurons": n, "mode": mode, "rep": rep,
                "is_warmup": rep < 0, "status": "timeout",
                "error": f"exceeded {timeout} s",
                "total_s": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--backends", default=",".join(ALL))
    ap.add_argument("--sizes", default=",".join(map(str, spec.SIZES)))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--budget-s", type=float, default=120.0,
                    help="skip larger sizes once one run's total exceeds this")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    spike_dir = os.path.join(a.out, "spikes")
    os.makedirs(spike_dir, exist_ok=True)
    backends = a.backends.split(",")
    sizes = [int(s) for s in a.sizes.split(",")]

    meta_path = os.path.join(a.out, "meta.json")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if os.path.exists(meta_path):
        # resuming an interrupted campaign: keep the original start time
        with open(meta_path) as f:
            meta = json.load(f)
        meta.setdefault("resumed", []).append(now)
    else:
        meta = {"spec": spec.describe(), "machine": machine_info(),
                "started": now, "backends": backends, "sizes": sizes,
                "reps": a.reps, "warmup_runs": 1, "threads": 1,
                "seed": a.seed, "budget_s": a.budget_s, "timeout_s": a.timeout}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # shared inputs are generated once, outside every timed run
    spec.input_spikes()
    for n in sizes:
        spec.connectivity(n)

    runs_path = os.path.join(a.out, "runs.jsonl")
    fid_path = os.path.join(a.out, "fidelity.jsonl")
    rng = random.Random(a.seed)
    # backend -> (size, reason, hard). A hard drop (error/timeout) stops the
    # backend at once; a budget drop finishes the current size first.
    dropped = {}

    # resume: every (backend, size, mode, rep) already on disk is not re-run,
    # and backends dropped earlier stay dropped
    done = set()
    if os.path.exists(runs_path):
        with open(runs_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["backend"], r["n_neurons"], r["mode"], r["rep"]))
                if r["status"] == "skipped" or r["mode"] != "matched":
                    continue
                if r["status"] != "ok":
                    dropped.setdefault(r["backend"],
                                       (r["n_neurons"], r["status"], True))
                elif r.get("total_s", 0) > a.budget_s and not r["is_warmup"]:
                    dropped.setdefault(r["backend"], (
                        r["n_neurons"], f"run took {r['total_s']:.0f} s > budget",
                        False))
    fid_done = set()
    if os.path.exists(fid_path):
        with open(fid_path) as f:
            for line in f:
                r = json.loads(line)
                fid_done.add((r["backend"], r["n_neurons"], r["mode"]))

    def add_fidelity(b, n, mode, sp, ref):
        if (b, n, mode) in fid_done or not os.path.exists(sp):
            return
        z = np.load(sp)
        fr = fidelity.compare(ref["steps"], ref["neurons"],
                              z["steps"], z["neurons"], n)
        fr.update({"backend": b, "n_neurons": n, "mode": mode})
        with open(fid_path, "a") as f:
            f.write(json.dumps(fr) + "\n")
        fid_done.add((b, n, mode))

    def log(rec):
        with open(runs_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {rec['backend']:14s} "
              f"n={rec['n_neurons']:<6d} {rec['mode']:8s} rep={rec['rep']:<2d} "
              f"{rec['status']:7s} sim={rec.get('simulate_s', float('nan')):.3f}s "
              f"total={rec.get('total_s', float('nan')):.2f}s", flush=True)

    for n in sizes:
        ref_path = os.path.join(spike_dir, f"reference_{n}.npz")
        if not os.path.exists(ref_path):
            s, nn = spec.reference_spikes(n)
            np.savez_compressed(ref_path, steps=s, neurons=nn)
        ref = np.load(ref_path)
        active = [b for b in backends if b not in dropped or dropped[b][0] >= n]
        for b in [b for b in backends if b in dropped and dropped[b][0] < n]:
            if (b, n, "matched", 0) not in done:
                log({"backend": b, "n_neurons": n, "mode": "matched", "rep": 0,
                     "is_warmup": False, "status": "skipped",
                     "error": f"dropped after n={dropped[b][0]}: {dropped[b][1]}"})
        for rep in range(-1, a.reps):
            order = active[:]
            rng.shuffle(order)
            for b in order:
                if (b, n, "matched", rep) in done:
                    continue
                if b in dropped and (dropped[b][0] < n or dropped[b][2]):
                    continue
                sp = (os.path.join(spike_dir, f"{b}_matched_{n}.npz")
                      if rep == 0 else None)
                rec = run_once(b, n, "matched", rep, sp, a.timeout)
                rec["order_in_round"] = order.index(b)
                log(rec)
                if rec["status"] != "ok":
                    dropped[b] = (n, rec["status"], True)
                elif (rec.get("total_s", 0) > a.budget_s and not rec["is_warmup"]
                      and b not in dropped):
                    dropped[b] = (n, f"run took {rec['total_s']:.0f} s > budget",
                                  False)
        for b in active:
            add_fidelity(b, n, "matched",
                         os.path.join(spike_dir, f"{b}_matched_{n}.npz"), ref)
        # one native-mode run per backend and size, for fidelity
        for b in active:
            if b in dropped and (dropped[b][0] < n or dropped[b][2]):
                continue
            sp = os.path.join(spike_dir, f"{b}_native_{n}.npz")
            if (b, n, "native", 0) not in done:
                rec = run_once(b, n, "native", 0, sp, a.timeout)
                log(rec)
            add_fidelity(b, n, "native", sp, ref)

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("campaign finished", flush=True)


if __name__ == "__main__":
    main()
