import numpy as np

from .. import spec

NAME = "nengo"
LIBRARY = "Nengo (reference simulator)"
PARADIGM = "neural-engineering"

_ng = None


def _imp():
    global _ng
    if _ng is None:
        import nengo
        _ng = nengo
    return _ng


def versions():
    return {"nengo": _imp().__version__}


def notes(mode):
    base = ("LIF with tau_ref = 0, threshold normalised to 1 (strict >), "
            "gain 1 / bias 0, synapse=None, dense transform scaled by "
            "1 / (theta * (1 - beta)) so one step's input equals the spec's "
            "voltage jump; spikes probed densely (Nengo has no sparse spike "
            "monitor). ")
    if mode == "matched":
        return base + ("Initial voltage forced to 0 (Nengo's default draws it "
                       "from Uniform(0, 1) of threshold); input node presents "
                       "step k-1 at step k (1-step delay).")
    return base + ("Default random initial voltage, Uniform(0, 1) of threshold; "
                   "input node presents step k at step k (no delay).")


def build(n, mode):
    nengo = _imp()
    dt = spec.DT_MS * 1e-3
    x = spec.input_raster(np.float64)
    shift = 1 if mode == "matched" else 0
    W = spec.dense_weights(n, np.float64) / (spec.THETA_MV * (1.0 - spec.BETA))

    def present(t):
        k = int(round(t / dt)) - shift
        return x[k] if 0 <= k < spec.N_STEPS else np.zeros(spec.N_IN)

    model = nengo.Network(seed=0)
    with model:
        inp = nengo.Node(present, size_out=spec.N_IN)
        init = ({"voltage": nengo.dists.Choice([0.0])}
                if mode == "matched" else None)
        ens = nengo.Ensemble(
            n, 1,
            neuron_type=nengo.LIF(tau_rc=spec.TAU_M_MS * 1e-3, tau_ref=0.0,
                                  initial_state=init),
            gain=np.ones(n), bias=np.zeros(n))
        nengo.Connection(inp, ens.neurons, transform=W, synapse=None)
        probe = nengo.Probe(ens.neurons, "output", synapse=None)
    return {"nengo": nengo, "model": model, "probe": probe, "dt": dt}


def prepare(ctx):
    # the Nengo builder turns the model into signals and operators
    ctx["sim"] = ctx["nengo"].Simulator(ctx["model"], dt=ctx["dt"],
                                        progress_bar=False, optimize=True)


def simulate(ctx):
    ctx["sim"].run_steps(spec.N_STEPS - 1)


def spikes(ctx):
    data = ctx["sim"].data[ctx["probe"]]          # (steps, n), row i = step i+1
    rows, neurons = np.nonzero(data)
    ctx["sim"].close()
    return (rows + 1).astype(np.int32), neurons.astype(np.int32)
