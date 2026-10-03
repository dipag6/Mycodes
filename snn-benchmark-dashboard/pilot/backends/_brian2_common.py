"""Shared Brian2 network construction for the runtime (Cython) and C++
standalone variants. Only the device differs between them."""

import numpy as np

from .. import spec


def construct(b2, n, mode):
    """Returns (net, monitor). b2 is the imported brian2 module."""
    ms, mV = b2.ms, b2.mV
    b2.defaultclock.dt = spec.DT_MS * ms
    pre, post, w = spec.connectivity(n)
    in_steps, in_src = spec.input_spikes()
    tau = spec.TAU_M_MS * ms
    theta = spec.THETA_MV * mV
    # native: the threshold test most Brian2 examples use; matched: the
    # spec's ">=" test
    thr = "v >= theta" if mode == "matched" else "v > theta"
    G = b2.NeuronGroup(n, "dv/dt = -v / tau : volt", threshold=thr,
                       reset="v = 0*mV", method="exact",
                       namespace={"tau": tau, "theta": theta})
    src = b2.SpikeGeneratorGroup(spec.N_IN, in_src, in_steps * spec.DT_MS * ms)
    S = b2.Synapses(src, G, "w : volt", on_pre="v_post += w")
    S.connect(i=pre, j=post)
    S.w = w * mV
    if mode == "matched":
        # Default schedule adds input after the threshold test, so input is
        # decayed once before it can trigger a spike and input arriving in the
        # step a neuron fires is overwritten by the reset. Delivering before
        # the threshold test reproduces the spec (verified with an
        # at-threshold single-input test).
        S.pre.when = "before_thresholds"
    M = b2.SpikeMonitor(G)
    net = b2.Network(G, src, S, M)
    return net, M


def to_steps(M):
    # t_ is the unitless value in seconds
    steps = np.round(np.asarray(M.t_) / (spec.DT_MS * 1e-3)).astype(np.int32)
    return steps, np.asarray(M.i[:]).astype(np.int32)
