"""Backend adapters.

Each adapter module exposes:

    NAME, PARADIGM, LIBRARY            identification for the results table
    versions() -> dict                 library versions actually imported
    build(n, mode) -> ctx              construct the network (timed as "build")
    prepare(ctx)                       code generation / compilation / JIT
                                       (timed as "prepare")
    simulate(ctx)                      advance T_MS of model time ("simulate")
    spikes(ctx) -> (steps, neurons)    output spikes on the canonical step grid
    notes(mode) -> str                 what this mode does relative to the spec

mode is "matched" (configured to reproduce spec.py's canonical update) or
"native" (the most direct implementation using the library's own defaults).
"""

import importlib

MODULES = {
    "numpy_ref": "numpy_ref",
    "nest": "nest_backend",
    "brian2_cython": "brian2_cython",
    "brian2_cpp": "brian2_cpp",
    "nengo": "nengo_backend",
    "snntorch": "snntorch_backend",
    "spikingjelly": "spikingjelly_backend",
}
ALL = list(MODULES)


def load(name):
    return importlib.import_module(f".{MODULES[name]}", __name__)
