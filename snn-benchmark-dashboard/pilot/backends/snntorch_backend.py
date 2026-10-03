import numpy as np

from .. import spec

NAME = "snntorch"
LIBRARY = "snnTorch (PyTorch, CPU)"
PARADIGM = "deep-learning"

_mods = None


def _imp():
    global _mods
    if _mods is None:
        import torch
        import snntorch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _mods = (torch, snntorch)
    return _mods


def versions():
    torch, snntorch = _imp()
    return {"snntorch": snntorch.__version__, "torch": torch.__version__}


def notes(mode):
    base = ("snn.Leaky(beta=exp(-dt/tau), threshold=theta, reset='zero') after "
            "nn.Linear(bias=False) holding the spec's weights; float32; "
            "step-by-step loop as in the snnTorch tutorials; no refractory "
            "period exists in snn.Leaky; spike test is strict (>). ")
    if mode == "matched":
        return base + "Input of step k-1 fed at step k (1-step delay)."
    return base + "Input of step k fed at step k (no delay)."


def build(n, mode):
    torch, snntorch = _imp()
    fc = torch.nn.Linear(spec.N_IN, n, bias=False)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(spec.dense_weights(n, np.float32)))
    lif = snntorch.Leaky(beta=spec.BETA, threshold=spec.THETA_MV,
                         reset_mechanism="zero")
    x = torch.from_numpy(spec.input_raster(np.float32))
    return {"torch": torch, "fc": fc, "lif": lif, "x": x, "n": n,
            "shift": 1 if mode == "matched" else 0}


def prepare(ctx):
    torch = ctx["torch"]
    # one throwaway step so lazy kernel/allocator set-up is not timed as
    # simulation
    with torch.no_grad():
        mem = ctx["lif"].init_leaky()
        ctx["lif"](ctx["fc"](ctx["x"][0:1]), mem)


def simulate(ctx):
    torch, fc, lif, x = ctx["torch"], ctx["fc"], ctx["lif"], ctx["x"]
    shift = ctx["shift"]
    out = torch.zeros((spec.N_STEPS, ctx["n"]), dtype=torch.uint8)
    with torch.no_grad():
        mem = lif.init_leaky()
        for k in range(1, spec.N_STEPS):
            cur = fc(x[k - shift])
            spk, mem = lif(cur, mem)
            out[k] = spk.to(torch.uint8)
    ctx["out"] = out


def spikes(ctx):
    steps, neurons = np.nonzero(ctx["out"].numpy())
    return steps.astype(np.int32), neurons.astype(np.int32)
