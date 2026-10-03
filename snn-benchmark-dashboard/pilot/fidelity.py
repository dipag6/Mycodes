"""Compare a backend's output spikes with the canonical reference.

Metrics
-------
count_rel_err   (spikes_backend - spikes_ref) / spikes_ref
rate_corr       Pearson r of per-neuron spike counts
exact_match     fraction of reference spikes reproduced at exactly the same
                (neuron, step)
gamma_1ms       coincidence factor of Kistler, Gerstner & van Hemmen (1997),
                precision +/-1 ms, averaged over neurons that fire in either
                train; 1 = identical trains, 0 = chance-level agreement
"""

import numpy as np

from . import spec


def _by_neuron(steps, neurons, n):
    order = np.lexsort((steps, neurons))
    s, nn = steps[order], neurons[order]
    bounds = np.searchsorted(nn, np.arange(n + 1))
    return [s[bounds[i]:bounds[i + 1]] for i in range(n)]


def coincidence_factor(ref, test, delta_steps, duration_steps):
    n_ref, n_test = len(ref), len(test)
    if n_ref == 0 and n_test == 0:
        return np.nan
    if n_ref == 0 or n_test == 0:
        return 0.0
    # count reference spikes with a test spike within +/- delta (each test
    # spike used at most once, greedy in time order)
    i = j = coinc = 0
    while i < n_ref and j < n_test:
        d = test[j] - ref[i]
        if abs(d) <= delta_steps:
            coinc += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    rate_test = n_test / duration_steps
    expected = 2 * rate_test * delta_steps * n_ref
    norm = 1 - 2 * rate_test * delta_steps
    if norm <= 0:
        return np.nan
    return (coinc - expected) / (0.5 * (n_ref + n_test)) / norm


def compare(ref_steps, ref_neurons, steps, neurons, n):
    out = {"n_spikes_ref": int(len(ref_steps)), "n_spikes": int(len(steps))}
    out["count_rel_err"] = ((len(steps) - len(ref_steps)) / len(ref_steps)
                            if len(ref_steps) else np.nan)
    c_ref = np.bincount(ref_neurons, minlength=n)
    c = np.bincount(neurons, minlength=n)
    out["rate_corr"] = (float(np.corrcoef(c_ref, c)[0, 1])
                        if c_ref.std() > 0 and c.std() > 0 else np.nan)
    key_ref = ref_neurons.astype(np.int64) * spec.N_STEPS + ref_steps
    key = neurons.astype(np.int64) * spec.N_STEPS + steps
    out["exact_match"] = (float(np.isin(key_ref, key).mean())
                          if len(key_ref) else np.nan)
    delta = int(round(1.0 / spec.DT_MS))
    r_tr = _by_neuron(ref_steps, ref_neurons, n)
    t_tr = _by_neuron(steps, neurons, n)
    g = [coincidence_factor(r_tr[i], t_tr[i], delta, spec.N_STEPS)
         for i in range(n)]
    g = np.asarray([x for x in g if not np.isnan(x)])
    out["gamma_1ms"] = float(g.mean()) if len(g) else np.nan
    # first step at which any neuron disagrees, a quick pointer for debugging
    diff = np.setxor1d(key_ref, key)
    out["first_divergence_ms"] = (float((diff % spec.N_STEPS).min() * spec.DT_MS)
                                  if len(diff) else None)
    return out
