"""Does the way input reaches NEST neurons change its cost?

    python -m pilot.nest_idiom_probe --n 4000 --relay direct|parrot

direct: 1,000 spike_generators connected straight to every target (the
        pilot's NEST adapter);
parrot: each generator feeds one parrot_neuron, and the parrots connect to the
        targets, so input travels the neuron-to-neuron path.

The parrot relay adds one step of delay, so its spikes land one step later
than the reference; this probe compares cost and spike counts only.
Prints one JSON line.
"""

import argparse
import json
import resource
import time

import numpy as np

from . import spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--relay", choices=["direct", "parrot"], required=True)
    a = ap.parse_args()

    import nest
    try:
        nest.verbosity = nest.VerbosityLevel.ERROR
    except AttributeError:
        nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": spec.DT_MS, "local_num_threads": 1})
    pre, post, w = spec.connectivity(a.n)
    in_steps, in_src = spec.input_spikes()

    t0 = time.perf_counter()
    neurons = nest.Create("iaf_psc_delta", a.n, params={
        "E_L": 0.0, "V_reset": spec.V_RESET_MV, "V_th": spec.THETA_MV,
        "t_ref": 0.0, "tau_m": spec.TAU_M_MS, "V_m": 0.0})
    gens = nest.Create("spike_generator", spec.N_IN)
    order = np.argsort(in_src, kind="stable")
    src_sorted, steps_sorted = in_src[order], in_steps[order]
    bounds = np.searchsorted(src_sorted, np.arange(spec.N_IN + 1))
    gens.set([{"spike_times": list(np.round(steps_sorted[bounds[j]:bounds[j + 1]] * spec.DT_MS, 6))}
              for j in range(spec.N_IN)])
    if a.relay == "parrot":
        parrots = nest.Create("parrot_neuron", spec.N_IN)
        nest.Connect(gens, parrots, "one_to_one", syn_spec={"delay": spec.DT_MS})
        sources = np.asarray(parrots.tolist())
    else:
        sources = np.asarray(gens.tolist())
    nest.Connect(sources[pre], np.asarray(neurons.tolist())[post], "one_to_one",
                 syn_spec={"weight": w, "delay": np.full(len(w), spec.DT_MS)})
    rec = nest.Create("spike_recorder")
    nest.Connect(neurons, rec)
    t1 = time.perf_counter()
    nest.Prepare()
    t2 = time.perf_counter()
    nest.Run(spec.T_MS)
    t3 = time.perf_counter()
    nest.Cleanup()
    print(json.dumps({"n": a.n, "relay": a.relay, "build_s": t1 - t0,
                      "prepare_s": t2 - t1, "simulate_s": t3 - t2,
                      "n_spikes": int(rec.get("n_events")),
                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))


if __name__ == "__main__":
    main()
