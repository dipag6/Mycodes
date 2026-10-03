import os

from .. import spec
from ._brian2_common import construct, to_steps

NAME = "brian2_cython"
LIBRARY = "Brian2 (runtime, Cython)"
PARADIGM = "neuroscience"

_b2 = None


def _imp():
    global _b2
    if _b2 is None:
        import brian2
        brian2.prefs.codegen.target = "cython"
        cache = os.environ.get("SNNBENCH_BRIAN_CACHE")
        if cache:
            brian2.prefs.codegen.runtime.cython.cache_dir = cache
        brian2.prefs.logging.console_log_level = "ERROR"
        _b2 = brian2
    return _b2


def versions():
    b2 = _imp()
    import Cython
    return {"brian2": b2.__version__, "cython": Cython.__version__}


def notes(mode):
    if mode == "matched":
        return ("Exact integration; threshold v >= theta; synaptic pathway moved "
                "to 'before_thresholds' so input precedes the threshold test.")
    return ("Exact integration; threshold v > theta; default schedule (input "
            "added after the threshold test).")


def build(n, mode):
    b2 = _imp()
    b2.start_scope()
    net, M = construct(b2, n, mode)
    return {"b2": b2, "net": net, "M": M}


def prepare(ctx):
    # a zero-length run generates and compiles every code object (or loads
    # it from the on-disk Cython cache)
    ctx["net"].run(0 * ctx["b2"].ms)


def simulate(ctx):
    ctx["net"].run(spec.T_MS * ctx["b2"].ms)


def spikes(ctx):
    steps, neurons = to_steps(ctx["M"])
    keep = steps < spec.N_STEPS
    return steps[keep], neurons[keep]
