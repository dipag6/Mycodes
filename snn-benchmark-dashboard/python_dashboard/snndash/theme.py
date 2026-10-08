"""Colours, simulator identities and the shared Plotly styling.

Colour follows the simulator, never its rank: each configuration keeps one
colour whatever is filtered. Variants of one simulator (NEST thread counts, the
two NumPy controls) share its colour and differ by line dash. The palette is
the validated eight-slot categorical palette, with separate steps for the dark
theme rather than an automatic flip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SLOTS = {
    "light": {0: "#8a939d", 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100",
              5: "#e87ba4", 6: "#008300", 7: "#4a3aa7", 8: "#e34948"},
    "dark": {0: "#7d8691", 1: "#3987e5", 2: "#d95926", 3: "#199e70", 4: "#c98500",
             5: "#d55181", 6: "#008300", 7: "#9085e9", 8: "#e66767"},
}

CHROME = {
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", band="rgba(12,163,12,0.08)", band2="rgba(12,163,12,0.16)"),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", band="rgba(12,163,12,0.12)", band2="rgba(12,163,12,0.22)"),
}

# markers carry the same variant distinction as the line dash, for charts without lines
SYMBOLS = {"solid": "circle", "dash": "square", "dot": "diamond"}

# status colours are fixed across themes and always travel with a word
STATUS = {"good": "#0ca30c", "warn": "#fab219", "bad": "#d03b3b"}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


@dataclass(frozen=True)
class Identity:
    id: str
    label: str
    short: str
    slot: int
    paradigm: str
    order: float
    dash: str = "solid"      # plotly line dash: solid, dash or dot

    def color(self, theme: str = "light") -> str:
        return SLOTS[theme][self.slot]

    @property
    def symbol(self) -> str:
        return SYMBOLS.get(self.dash, "circle")

    @property
    def is_reference(self) -> bool:
        return self.paradigm == "Reference"


# snnbench names each configuration "framework/backend"
KNOWN = {
    "nest/threads1": Identity("nest_t1", "NEST · 1 thread", "NEST ×1", 1, "Neuroscience", 1.0),
    "nest/threads2": Identity("nest_t2", "NEST · 2 threads", "NEST ×2", 1, "Neuroscience", 1.1, "dash"),
    "nest/threads4": Identity("nest_t4", "NEST · 4 threads", "NEST ×4", 1, "Neuroscience", 1.2, "dot"),
    "brian2/numpy": Identity("brian2_numpy", "Brian2 · NumPy", "B2 NumPy", 7, "Neuroscience", 1.9),
    "brian2/cython": Identity("brian2_cython", "Brian2 · Cython", "B2 Cython", 2, "Neuroscience", 2.0),
    "brian2/cpp_standalone": Identity("brian2_cpp", "Brian2 · C++ standalone", "B2 C++", 3, "Neuroscience", 3.0),
    "nengo/reference": Identity("nengo", "Nengo", "Nengo", 4, "Neural engineering", 4.0),
    "snntorch/cpu": Identity("snntorch", "snnTorch", "snnTorch", 5, "Deep learning", 5.0),
    "spikingjelly/cpu": Identity("spikingjelly", "SpikingJelly", "SJelly", 6, "Deep learning", 6.0),
    "reference/vectorised": Identity("numpy_vec", "NumPy control · vectorised", "NumPy vec", 0, "Reference", 98.0),
    "reference/scalar": Identity("numpy_loop", "NumPy control · loop", "NumPy loop", 0, "Reference", 99.0, "dash"),
}

_extra: dict[str, Identity] = {}


def identity(key: str) -> Identity:
    """The identity for an snnbench "framework/backend" key. Unknown keys get
    slot 8, then grey, so a new configuration never repaints a known one."""
    if key in KNOWN:
        return KNOWN[key]
    if key not in _extra:
        slot = 8 if not any(i.slot == 8 for i in _extra.values()) else 0
        slug = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") or "unknown"
        _extra[key] = Identity(slug, key, key[:10], slot, "Other", 50.0 + len(_extra))
    return _extra[key]


def base_layout(theme: str = "light", **overrides) -> dict:
    """Shared figure styling: surface, ink, recessive grid, a hover label that
    matches the page."""
    c = CHROME[theme]
    axis = dict(gridcolor=c["grid"], linecolor=c["axis"], zerolinecolor=c["axis"], tickcolor=c["axis"],
                tickfont=dict(color=c["muted"], size=12), title=dict(font=dict(color=c["ink2"], size=13)),
                showline=True, ticks="outside", ticklen=4)
    layout = dict(
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        font=dict(family=FONT, color=c["ink2"], size=13),
        margin=dict(l=70, r=24, t=16, b=56),
        hoverlabel=dict(bgcolor=c["surface"], bordercolor=c["axis"], font=dict(color=c["ink"], family=FONT)),
        legend=dict(font=dict(color=c["ink2"], size=12), bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(axis), yaxis=dict(axis),
    )
    layout.update(overrides)
    return layout
