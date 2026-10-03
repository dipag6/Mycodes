import os
import tempfile

from .. import spec
from ._brian2_common import construct, to_steps

NAME = "brian2_cpp"
LIBRARY = "Brian2 (C++ standalone)"
PARADIGM = "neuroscience"

_b2 = None


def _imp():
    global _b2
    if _b2 is None:
        import brian2
        brian2.prefs.logging.console_log_level = "ERROR"
        _b2 = brian2
    return _b2


def versions():
    b2 = _imp()
    return {"brian2": b2.__version__}


def notes(mode):
    base = ("Whole network generated as C++ and compiled with Brian2's default "
            "flags (-O3 -ffast-math -march=native); prepare = code generation + "
            "compilation (make runs in parallel, so compile CPU time exceeds its "
            "wall time), simulate = running the binary on one thread. Peak memory "
            "covers the Python process only, not the binary. ")
    if mode == "matched":
        return base + "Threshold v >= theta; pathway before the threshold test."
    return base + "Threshold v > theta; default schedule."


def build(n, mode):
    b2 = _imp()
    root = os.environ.get("SNNBENCH_STANDALONE_DIR",
                          os.path.join(tempfile.gettempdir(), "snnbench_standalone"))
    # one directory per (size, mode): repeats reuse unchanged sources, which
    # is what a user re-running the same model experiences
    directory = os.path.join(root, f"{mode}_{n}")
    b2.prefs.devices.cpp_standalone.openmp_threads = 0
    b2.set_device("cpp_standalone", directory=directory, build_on_run=False)
    net, M = construct(b2, n, mode)
    net.run(spec.T_MS * b2.ms)
    return {"b2": b2, "M": M, "directory": directory}


def prepare(ctx):
    ctx["b2"].device.build(directory=ctx["directory"], compile=True, run=False,
                           with_output=False)


def simulate(ctx):
    dev = ctx["b2"].device
    dev.run(directory=ctx["directory"], with_output=False)
    ctx["inner_run_s"] = float(getattr(dev, "_last_run_time", float("nan")))


def spikes(ctx):
    steps, neurons = to_steps(ctx["M"])
    keep = steps < spec.N_STEPS
    return steps[keep], neurons[keep]


def extra(ctx):
    return {"inner_run_s": ctx.get("inner_run_s")}
