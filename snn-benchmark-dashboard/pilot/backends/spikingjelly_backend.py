import numpy as np

from .. import spec

NAME = "spikingjelly"
LIBRARY = "SpikingJelly (PyTorch, CPU)"
PARADIGM = "deep-learning"

_mods = None


def _imp():
    global _mods
    if _mods is None:
        import torch
        from spikingjelly.activation_based import functional, layer, neuron
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _mods = (torch, neuron, layer, functional)
    return _mods


def versions():
    import importlib.metadata as md
    torch = _imp()[0]
    return {"spikingjelly": md.version("spikingjelly"), "torch": torch.__version__}


def notes(mode):
    base = ("layer.Linear + neuron.LIFNode(decay_input=False, v_reset=0, hard "
            "reset, spike test >=) in multi-step mode (the whole 10,000-step "
            "sequence per call, torch backend); float32. ")
    if mode == "matched":
        return base + ("tau = 1/(1-beta) so the forward-Euler decay equals "
                       "exp(-dt/tau_m); input of step k-1 fed at step k.")
    return base + ("tau = tau_m/dt (forward-Euler decay 1 - dt/tau_m); input "
                   "of step k fed at step k.")


def build(n, mode):
    torch, neuron, layer, functional = _imp()
    if mode == "matched":
        tau = 1.0 / (1.0 - spec.BETA)
    else:
        tau = spec.TAU_M_MS / spec.DT_MS
    fc = layer.Linear(spec.N_IN, n, bias=False, step_mode="m")
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(spec.dense_weights(n, np.float32)))
    lif = neuron.LIFNode(tau=tau, decay_input=False, v_threshold=spec.THETA_MV,
                         v_reset=0.0, detach_reset=True, step_mode="m",
                         backend="torch")
    x = spec.input_raster(np.float32)
    if mode == "matched":
        x = np.vstack([np.zeros((1, spec.N_IN), np.float32), x[:-1]])
    # canonical step 0 holds the initial state, so present steps 1..T-1
    x = torch.from_numpy(x[1:]).unsqueeze(1)          # (T-1, batch=1, N_IN)
    return {"torch": torch, "fc": fc, "lif": lif, "x": x,
            "functional": functional}


def prepare(ctx):
    torch = ctx["torch"]
    # first call compiles the TorchScript loop used by the multi-step neuron
    with torch.no_grad():
        ctx["lif"](ctx["fc"](ctx["x"][:2]))
    ctx["functional"].reset_net(ctx["lif"])


def simulate(ctx):
    torch = ctx["torch"]
    with torch.no_grad():
        ctx["out"] = ctx["lif"](ctx["fc"](ctx["x"]))
    ctx["functional"].reset_net(ctx["lif"])


def spikes(ctx):
    rows, _, neurons = np.nonzero(ctx["out"].numpy())
    return (rows + 1).astype(np.int32), neurons.astype(np.int32)
