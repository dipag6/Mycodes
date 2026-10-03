import numpy as np

from .. import spec

NAME = "numpy_ref"
LIBRARY = "NumPy (hand-written reference)"
PARADIGM = "reference"


def versions():
    return {"numpy": np.__version__}


def notes(mode):
    return "Canonical update in float64; defines the spec (identical in both modes)."


def build(n, mode):
    pre, post, w = spec.connectivity(n)
    order = np.argsort(pre, kind="stable")
    in_steps, in_src = spec.input_spikes()
    return {"n": n, "post": post[order], "w": w[order],
            "bounds": np.searchsorted(pre[order], np.arange(spec.N_IN + 1)),
            "in_steps": in_steps, "in_src": in_src}


def prepare(ctx):
    pass


def simulate(ctx):
    n, post, w, bounds = ctx["n"], ctx["post"], ctx["w"], ctx["bounds"]
    in_steps, in_src = ctx["in_steps"], ctx["in_src"]
    v = np.zeros(n)
    out_s, out_n = [], []
    ptr, n_in = 0, len(in_steps)
    for k in range(1, spec.N_STEPS):
        v *= spec.BETA
        while ptr < n_in and in_steps[ptr] == k - 1:
            j = in_src[ptr]
            a, b = bounds[j], bounds[j + 1]
            np.add.at(v, post[a:b], w[a:b])
            ptr += 1
        fired = np.flatnonzero(v >= spec.THETA_MV)
        if fired.size:
            out_s.append(np.full(fired.size, k, np.int32))
            out_n.append(fired.astype(np.int32))
            v[fired] = spec.V_RESET_MV
    ctx["out"] = (np.concatenate(out_s) if out_s else np.zeros(0, np.int32),
                  np.concatenate(out_n) if out_n else np.zeros(0, np.int32))


def spikes(ctx):
    return ctx["out"]
