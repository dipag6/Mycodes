"""Workload specification shared by every backend.

Feedforward LIF layer driven by a fixed bank of Poisson inputs. Every backend
receives byte-identical input spikes and connectivity, generated here once
from fixed seeds, so any difference in output spikes comes from the simulator
and not from the experiment.

Canonical update (what "matched" mode reproduces in every backend), for
step k >= 1 on a grid of width DT:

    v[k] = BETA * v[k-1] + sum_j W[i, j] * X[k-1, j]     (1-step input delay)
    if v[k] >= THETA: spike at step k, v[k] = V_RESET

with BETA = exp(-DT / TAU_M), no refractory period and v[0] = 0.
"""

import math
import os

import numpy as np

SPEC_VERSION = "1.0"

DT_MS = 0.1            # integration step
T_MS = 1000.0          # simulated (biological) time
N_STEPS = int(round(T_MS / DT_MS))
TAU_M_MS = 20.0
THETA_MV = 20.0
V_RESET_MV = 0.0
BETA = math.exp(-DT_MS / TAU_M_MS)

N_IN = 1000            # Poisson input sources
RATE_IN_HZ = 20.0
K_IN = 100             # fixed in-degree per neuron
W_MEAN_MV = 0.50       # weights ~ U(0.5, 1.5) * W_MEAN_MV

SEED_INPUT = 12345
SEED_CONN = 2024

SIZES = [500, 1000, 2000, 4000, 8000, 16000]

CACHE_DIR = os.environ.get(
    "SNNBENCH_CACHE", os.path.join(os.path.dirname(__file__), "_cache"))


def input_spikes():
    """(steps, sources) of every input spike, sorted by step."""
    path = os.path.join(CACHE_DIR, "input.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["steps"], z["sources"]
    rng = np.random.default_rng(SEED_INPUT)
    p = RATE_IN_HZ * DT_MS * 1e-3
    raster = rng.random((N_STEPS, N_IN)) < p
    # the last step's input would only arrive after the run ends
    raster[-1, :] = False
    # NEST spike generators need strictly positive spike times
    raster[0, :] = False
    steps, sources = np.nonzero(raster)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, steps=steps.astype(np.int32),
                        sources=sources.astype(np.int32))
    return steps.astype(np.int32), sources.astype(np.int32)


def input_raster(dtype=np.float32):
    steps, sources = input_spikes()
    x = np.zeros((N_STEPS, N_IN), dtype=dtype)
    x[steps, sources] = 1
    return x


def connectivity(n):
    """pre (input index), post (neuron index), weight in mV; sorted by post."""
    path = os.path.join(CACHE_DIR, f"conn_{n}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["pre"], z["post"], z["w"]
    rng = np.random.default_rng(SEED_CONN + n)
    pre = np.empty(n * K_IN, dtype=np.int32)
    for i in range(n):
        pre[i * K_IN:(i + 1) * K_IN] = rng.choice(N_IN, K_IN, replace=False)
    post = np.repeat(np.arange(n, dtype=np.int32), K_IN)
    w = rng.uniform(0.5, 1.5, n * K_IN) * W_MEAN_MV
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, pre=pre, post=post, w=w)
    return pre, post, w


def dense_weights(n, dtype=np.float32):
    """W[post, pre] as a dense matrix, the layout nn.Linear expects."""
    pre, post, w = connectivity(n)
    W = np.zeros((n, N_IN), dtype=dtype)
    W[post, pre] = w
    return W


def reference_spikes(n):
    """Canonical update in float64 NumPy. Returns (steps, neurons)."""
    pre, post, w = connectivity(n)
    in_steps, in_src = input_spikes()
    # inputs are delivered event by event, the way a clock-driven simulator
    # with a spike queue delivers them
    order = np.argsort(pre, kind="stable")
    pre_s, post_s, w_s = pre[order], post[order], w[order]
    bounds = np.searchsorted(pre_s, np.arange(N_IN + 1))
    v = np.zeros(n)
    out_steps, out_neurons = [], []
    ptr = 0
    n_in = len(in_steps)
    for k in range(1, N_STEPS):
        v *= BETA
        # inputs emitted at k-1 arrive now
        while ptr < n_in and in_steps[ptr] == k - 1:
            j = in_src[ptr]
            a, b = bounds[j], bounds[j + 1]
            np.add.at(v, post_s[a:b], w_s[a:b])
            ptr += 1
        fired = np.nonzero(v >= THETA_MV)[0]
        if fired.size:
            out_steps.append(np.full(fired.size, k, dtype=np.int32))
            out_neurons.append(fired.astype(np.int32))
            v[fired] = V_RESET_MV
    if out_steps:
        return np.concatenate(out_steps), np.concatenate(out_neurons)
    return np.zeros(0, np.int32), np.zeros(0, np.int32)


def describe():
    return {
        "spec_version": SPEC_VERSION, "dt_ms": DT_MS, "t_ms": T_MS,
        "n_steps": N_STEPS, "tau_m_ms": TAU_M_MS, "theta_mv": THETA_MV,
        "v_reset_mv": V_RESET_MV, "beta": BETA, "n_in": N_IN,
        "rate_in_hz": RATE_IN_HZ, "k_in": K_IN, "w_mean_mv": W_MEAN_MV,
        "seed_input": SEED_INPUT, "seed_conn": SEED_CONN,
        "input_delay_steps": 1, "refractory_ms": 0.0,
    }
