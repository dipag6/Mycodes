import numpy as np

from .. import spec

NAME = "nest"
LIBRARY = "NEST"
PARADIGM = "neuroscience"

_nest = None


def _imp():
    global _nest
    if _nest is None:
        import nest
        try:
            nest.verbosity = nest.VerbosityLevel.ERROR
        except AttributeError:
            nest.set_verbosity("M_ERROR")
        _nest = nest
    return _nest


def versions():
    nest = _imp()
    v = getattr(nest, "__version__", None) or nest.version()
    return {"nest-simulator": str(v)}


def notes(mode):
    return ("iaf_psc_delta with exact integration, delay = 1 step, V_th >= test. "
            "NEST's own update order already equals the spec, so native == matched.")


def build(n, mode):
    nest = _imp()
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": spec.DT_MS, "local_num_threads": 1,
                          "print_time": False})
    pre, post, w = spec.connectivity(n)
    in_steps, in_src = spec.input_spikes()
    neurons = nest.Create("iaf_psc_delta", n, params={
        "E_L": 0.0, "V_reset": spec.V_RESET_MV, "V_th": spec.THETA_MV,
        "t_ref": 0.0, "tau_m": spec.TAU_M_MS, "V_m": 0.0, "I_e": 0.0})
    gens = nest.Create("spike_generator", spec.N_IN)
    # spike emitted with time stamp k*dt reaches the neuron at (k+1)*dt,
    # i.e. canonical step k+1 (checked against a single-input test)
    order = np.argsort(in_src, kind="stable")
    src_sorted, steps_sorted = in_src[order], in_steps[order]
    bounds = np.searchsorted(src_sorted, np.arange(spec.N_IN + 1))
    times = [list(np.round(steps_sorted[bounds[j]:bounds[j + 1]] * spec.DT_MS, 6))
             for j in range(spec.N_IN)]
    gens.set([{"spike_times": t} for t in times])
    gids_pre = np.asarray(gens.tolist())[pre]
    gids_post = np.asarray(neurons.tolist())[post]
    nest.Connect(gids_pre, gids_post, conn_spec="one_to_one",
                 syn_spec={"synapse_model": "static_synapse",
                           "weight": w, "delay": np.full(len(w), spec.DT_MS)})
    rec = nest.Create("spike_recorder")
    nest.Connect(neurons, rec)
    first = neurons.tolist()[0]
    return {"nest": nest, "rec": rec, "first": first}


def prepare(ctx):
    nest = ctx["nest"]
    # Prepare() builds the connection infrastructure that Simulate() would
    # otherwise build inside the timed run
    nest.Prepare()


def simulate(ctx):
    nest = ctx["nest"]
    nest.Run(spec.T_MS)
    nest.Cleanup()


def spikes(ctx):
    ev = ctx["rec"].get("events")
    steps = np.round(np.asarray(ev["times"]) / spec.DT_MS).astype(np.int32)
    neurons = (np.asarray(ev["senders"]) - ctx["first"]).astype(np.int32)
    keep = steps < spec.N_STEPS
    return steps[keep], neurons[keep]
