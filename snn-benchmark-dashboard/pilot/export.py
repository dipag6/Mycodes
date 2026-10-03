"""Turn a campaign directory into the dashboard's dataset.

    python -m pilot.export --campaign <dir> [--host-a <dir>]

Writes data/pilot_runs.csv (one row per run), data/pilot_fidelity.csv,
data/pilot_dataset.json, and injects the dataset into
dashboard/template.html -> dashboard/index.html.
"""

import argparse
import csv
import json
import os
import statistics

import numpy as np

from . import spec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUN_FIELDS = ["backend", "library", "paradigm", "n_neurons", "mode", "rep",
              "is_warmup", "status", "import_s", "build_s", "prepare_s",
              "simulate_s", "collect_s", "total_s", "build_cpu_s",
              "prepare_cpu_s", "simulate_cpu_s", "cpu_s", "inner_run_s",
              "rss_after_import_mb", "peak_rss_mb", "n_spikes", "mean_rate_hz",
              "n_synapses", "host_fingerprint", "kernel", "started_at", "error"]

FID_FIELDS = ["backend", "n_neurons", "mode", "n_spikes_ref", "n_spikes",
              "count_rel_err", "rate_corr", "exact_match", "gamma_1ms",
              "first_divergence_ms"]

# Why a mode departs from the spec; established by reading each library's
# source and confirmed with single-input tests (see backends/*.py).
CAUSES = {
    ("brian2_cython", "native"): "Default schedule adds input after the threshold test: input decays one step before it can cause a spike, and input arriving in the step a neuron fires is overwritten by the reset.",
    ("brian2_cpp", "native"): "Same default schedule as the Cython runtime: input after the threshold test.",
    ("nengo", "native"): "Default initial membrane voltage is drawn from Uniform(0, 1) of threshold instead of starting at rest.",
    ("snntorch", "native"): "No synaptic delay: each input acts one step (0.1 ms) earlier, so spike counts match but almost no spike lands on the same step.",
    ("spikingjelly", "native"): "tau = tau_m/dt gives forward-Euler decay (1 - dt/tau_m) rather than exp(-dt/tau_m), and no synaptic delay.",
    ("snntorch", "matched"): "float32 state: a handful of near-threshold crossings round the other way.",
    ("spikingjelly", "matched"): "float32 state: a handful of near-threshold crossings round the other way.",
    ("nest", "native"): "NEST's defaults already implement the spec (exact integration, delay >= 1 step, >= threshold test).",
}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def raster_excerpt(spike_dir, n, backends, neurons=14, t_ms=200.0):
    """Spikes of the first `neurons` neurons in the first `t_ms`, per source."""
    max_step = int(t_ms / spec.DT_MS)

    def cut(path):
        if not os.path.exists(path):
            return None
        z = np.load(path)
        keep = (z["neurons"] < neurons) & (z["steps"] < max_step)
        pairs = sorted(zip(z["steps"][keep].tolist(), z["neurons"][keep].tolist()))
        return [[round(s * spec.DT_MS, 1), nn] for s, nn in pairs]

    out = {"n": n, "neurons": neurons, "t_ms": t_ms,
           "reference": cut(os.path.join(spike_dir, f"reference_{n}.npz")),
           "matched": {}, "native": {}}
    for b in backends:
        for mode in ("matched", "native"):
            r = cut(os.path.join(spike_dir, f"{b}_{mode}_{n}.npz"))
            if r is not None:
                out[mode][b] = r
    return out


def drift(host_a_runs, host_b_runs):
    """Median simulate time per backend at the sizes both hosts measured."""
    def med(runs):
        acc = {}
        for r in runs:
            if r.get("status") == "ok" and not r.get("is_warmup") \
                    and r.get("mode") == "matched":
                acc.setdefault((r["backend"], r["n_neurons"]), []).append(r["simulate_s"])
        return {k: statistics.median(v) for k, v in acc.items() if len(v) >= 3}
    a, b = med(host_a_runs), med(host_b_runs)
    rows = []
    for key in sorted(set(a) & set(b)):
        rows.append({"backend": key[0], "n_neurons": key[1],
                     "host_a_s": a[key], "host_b_s": b[key],
                     "ratio_a_over_b": a[key] / b[key]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--host-a", default=None,
                    help="earlier partial campaign on another machine")
    ap.add_argument("--raster-n", type=int, default=1000)
    a = ap.parse_args()

    with open(os.path.join(a.campaign, "meta.json")) as f:
        meta = json.load(f)
    runs = read_jsonl(os.path.join(a.campaign, "runs.jsonl"))
    fid = read_jsonl(os.path.join(a.campaign, "fidelity.jsonl"))

    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "pilot_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k) for k in RUN_FIELDS})
    with open(os.path.join(data_dir, "pilot_fidelity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FID_FIELDS + ["cause"], extrasaction="ignore")
        w.writeheader()
        for r in fid:
            row = {k: r.get(k) for k in FID_FIELDS}
            row["cause"] = CAUSES.get((r["backend"], r["mode"]), "")
            w.writerow(row)

    # versions and semantic notes, one entry per backend and mode
    backends = {}
    for r in runs:
        if r.get("status") != "ok":
            continue
        b = backends.setdefault(r["backend"], {"id": r["backend"],
                                               "library": r.get("library"),
                                               "paradigm": r.get("paradigm"),
                                               "versions": r.get("versions"),
                                               "notes": {}})
        b["notes"][r["mode"]] = r.get("notes")

    slim = []
    keep = ["backend", "n_neurons", "mode", "rep", "is_warmup", "status",
            "import_s", "build_s", "prepare_s", "simulate_s", "collect_s",
            "total_s", "simulate_cpu_s", "cpu_s", "peak_rss_mb",
            "rss_after_import_mb", "n_spikes", "mean_rate_hz", "inner_run_s",
            "error"]
    for r in runs:
        row = {k: r.get(k) for k in keep if r.get(k) is not None}
        for k, v in list(row.items()):
            if isinstance(v, float):
                row[k] = round(v, 6)
        if "error" in row:
            row["error"] = str(row["error"])[:300]
        slim.append(row)

    host_a = None
    if a.host_a:
        with open(os.path.join(a.host_a, "meta.json")) as f:
            meta_a = json.load(f)
        host_a = {"machine": meta_a.get("machine"),
                  "rows": drift(read_jsonl(os.path.join(a.host_a, "runs.jsonl")),
                                runs)}

    dataset = {
        "kind": "snn-benchmark-pilot",
        "title": "Reference pilot: feedforward LIF layer",
        "task": "Feedforward LIF inference",
        "hardware": "CPU, 1 thread",
        "meta": {k: meta.get(k) for k in ("spec", "machine", "started",
                                          "finished", "reps", "warmup_runs",
                                          "threads", "budget_s", "timeout_s",
                                          "resumed")},
        "backends": list(backends.values()),
        "runs": slim,
        "fidelity": [dict({k: r.get(k) for k in FID_FIELDS},
                          cause=CAUSES.get((r["backend"], r["mode"]), ""))
                     for r in fid],
        "raster": raster_excerpt(os.path.join(a.campaign, "spikes"), a.raster_n,
                                 list(backends)),
        "host_drift": host_a,
    }
    with open(os.path.join(data_dir, "pilot_dataset.json"), "w") as f:
        json.dump(dataset, f, separators=(",", ":"))

    tpl = os.path.join(ROOT, "dashboard", "template.html")
    if os.path.exists(tpl):
        with open(tpl) as f:
            html = f.read()
        blob = json.dumps(dataset, separators=(",", ":")).replace("</", "<\\/")
        html = html.replace("/*__PILOT_DATA__*/null", blob)
        with open(os.path.join(ROOT, "dashboard", "index.html"), "w") as f:
            f.write(html)
    print(f"{len(runs)} runs, {len(fid)} fidelity rows exported")


if __name__ == "__main__":
    main()
