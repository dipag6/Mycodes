"""Plotly figures. Every builder takes the theme ("light" or "dark") so the
dark view uses its own colour steps rather than an inverted light one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .analysis import fmt_time, signed_pct
from .data import TOLERANCE, Campaign
from .theme import CHROME, STATUS, base_layout

DASH = {"solid": "solid", "dash": "dash", "dot": "dot"}


def _empty(theme: str, message: str) -> go.Figure:
    c = CHROME[theme]
    fig = go.Figure()
    fig.update_layout(**base_layout(theme, height=220, xaxis=dict(visible=False), yaxis=dict(visible=False)))
    fig.add_annotation(text=message, showarrow=False, font=dict(color=c["muted"], size=14), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def tick_time(s: float) -> str:
    """Short, exact labels for time ticks (no "1.67 min" on a 100 s tick)."""
    if s < 1e-3:
        return f"{s * 1e6:g} µs"
    if s < 1:
        return f"{s * 1e3:g} ms"
    return f"{s:,g} s"


def tick_mb(mb: float) -> str:
    return f"{mb:,g} MB"


def log_ticks(values, fmt):
    """Tick positions and labels for a log axis, labelled with ``fmt``: decades
    over a wide span, 1-2-5 steps over a narrower one. Plotly draws only the
    ticks inside the axis range, so a generous list is harmless."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x) and x > 0], dtype=float)
    if not len(v):
        return None
    lo, hi = v.min(), v.max()
    span = np.log10(hi / lo)
    steps = (1,) if span > 2.5 else (1, 2, 5) if span > 0.8 else (1, 2, 3, 4, 5, 6, 7, 8, 9)
    e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    ticks = [m * 10.0 ** e for e in range(e0, e1 + 1) for m in steps]
    if sum(lo <= t <= hi for t in ticks) < 2:
        return None  # too narrow to label well by hand; leave it to Plotly
    return dict(tickvals=ticks, ticktext=[fmt(t) for t in ticks])


def log_lines(c: Campaign, agg: pd.DataFrame, ex: pd.DataFrame, theme: str, ytitle: str, fmt, bands=True,
              height=520, hover_unit="", tick_fmt=tick_time) -> go.Figure:
    """Medians per size on log-log axes, one line per configuration, the
    fastest-to-slowest range as a faint band. Exponents ride in the legend."""
    if agg.empty:
        return _empty(theme, "No runs for this selection.")
    fig = go.Figure()
    k_of = ex.set_index("backend") if len(ex) else pd.DataFrame()
    for b in c.ordered(agg["backend"]):
        idt, g = c.identities[b], agg[agg["backend"] == b].sort_values("n")
        color = idt.color(theme)
        k = k_of.loc[b, "exponent"] if b in k_of.index else np.nan
        name = idt.label + (f"  ·  k = {k:.2f}" if np.isfinite(k) else "")
        if bands and len(g) > 1 and (g["high"] > g["low"]).any():
            fig.add_trace(go.Scatter(x=list(g["n"]) + list(g["n"][::-1]), y=list(g["high"]) + list(g["low"][::-1]),
                                     mode="lines", fill="toself", fillcolor=_rgba(color, 0.10), line=dict(width=0),
                                     hoverinfo="skip",
                                     showlegend=False, legendgroup=b))
        fig.add_trace(go.Scatter(
            x=g["n"], y=g["median"], name=name, legendgroup=b, mode="lines+markers",
            line=dict(color=color, width=2, dash=DASH[idt.dash]),
            marker=dict(size=8, color=color, symbol=idt.symbol, line=dict(color=CHROME[theme]["surface"], width=2)),
            customdata=np.stack([g["low"].map(fmt), g["high"].map(fmt), g["runs"], g["median"].map(fmt)], axis=-1),
            hovertemplate=f"<b>{idt.label}</b><br>N = %{{x:,}}<br>median %{{customdata[3]}}{hover_unit}"
                          f"<br>range %{{customdata[0]}} – %{{customdata[1]}} over %{{customdata[2]}} runs<extra></extra>",
        ))
    fig.update_layout(**base_layout(theme, height=height, hovermode="closest", margin=dict(l=10, r=16, t=16, b=56)))
    fig.update_xaxes(type="log", title_text="Network size N (neurons, log scale)", tickvals=sorted(agg["n"].unique()),
                     ticktext=[f"{n:,}" for n in sorted(agg["n"].unique())])
    shown = list(agg["median"]) + (list(agg["low"]) + list(agg["high"]) if bands else [])
    fig.update_yaxes(type="log", title_text=ytitle, automargin=True, **(log_ticks(shown, tick_fmt) or {}))
    return fig


def equivalence(c: Campaign, f: pd.DataFrame, theme: str) -> go.Figure:
    """Spike count of each configuration against the NumPy loop: one dot per
    size, the campaign's tolerance shaded, dots outside it ringed."""
    if f.empty:
        return _empty(theme, "No spike counts to compare.")
    ch = CHROME[theme]
    order = c.ordered(f["backend"])
    sizes = sorted(f["n"].unique())
    lo = min(-TOLERANCE * 100 - 10, np.floor(f["error"].min() * 10) * 10)
    hi = max(20, np.ceil(f["error"].max() * 10) * 10)
    fig = go.Figure()
    fig.add_vrect(x0=-TOLERANCE * 100, x1=TOLERANCE * 100, fillcolor=ch["band"], line_width=0, layer="below")
    fig.add_vrect(x0=-5, x1=5, fillcolor=ch["band2"], line_width=0, layer="below")
    fig.add_vline(x=0, line=dict(color=ch["axis"], width=1))
    for b in order:
        idt, g = c.identities[b], f[f["backend"] == b].sort_values("n")
        out = g["error"].abs() > TOLERANCE
        fig.add_trace(go.Scatter(
            x=g["error"] * 100, y=[idt.label] * len(g), mode="markers", name=idt.label, showlegend=False,
            marker=dict(size=[7 + 2.2 * sizes.index(n) for n in g["n"]], color=idt.color(theme), opacity=0.85,
                        line=dict(color=[STATUS["bad"] if o else ch["surface"] for o in out], width=[2.5 if o else 1 for o in out])),
            customdata=np.stack([g["n"], g["spikes"], g["ref_spikes"], g["error"].map(signed_pct)], axis=-1),
            hovertemplate=f"<b>{idt.label}</b> · N = %{{customdata[0]:,}}<br>%{{customdata[1]:,}} spikes against "
                          f"%{{customdata[2]:,}} for the NumPy loop<br>%{{customdata[3]}}<extra></extra>",
        ))
    fig.add_annotation(x=-TOLERANCE * 100, y=1.0, yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                       text=f"±{TOLERANCE:.0%}", font=dict(size=11, color=ch["ink2"]))
    fig.add_annotation(x=0, y=1.0, yref="paper", xanchor="center", yanchor="bottom", showarrow=False, text="±5%",
                       font=dict(size=11, color=ch["ink2"]))
    fig.update_layout(**base_layout(theme, height=max(320, 44 * len(order) + 90), margin=dict(l=10, r=16, t=30, b=56)))
    fig.update_xaxes(range=[lo, hi], ticksuffix="%", title_text="Spike count against the NumPy loop", zeroline=False,
                     dtick=10)
    fig.update_yaxes(categoryorder="array", categoryarray=[c.identities[b].label for b in order][::-1],
                     showgrid=True, ticks="", automargin=True)
    return fig


def strip(c: Campaign, runs: pd.DataFrame, column: str, theme: str, xtitle: str, fmt, log=True,
          ceilings: dict = None, sort_by_median=True, height=None, tick_fmt=tick_time) -> go.Figure:
    """Every run as a dot, one row per configuration, the median as a bar."""
    data = runs.dropna(subset=[column])
    if data.empty:
        return _empty(theme, "No runs for this selection.")
    ch = CHROME[theme]
    med = data.groupby("backend")[column].median()
    # the first row drawn at the top: fastest first when sorted by median
    order = list(med.sort_values(ascending=False, kind="stable").index) if sort_by_median else c.ordered(med.index)[::-1]
    labels = [c.identities[b].label for b in order]
    fig = go.Figure()
    rng = np.random.default_rng(7)
    for b in order:
        idt, g = c.identities[b], data[data["backend"] == b]
        y_pos = labels.index(idt.label)
        jitter = rng.uniform(-0.18, 0.18, len(g))
        fig.add_trace(go.Scatter(
            x=g[column], y=y_pos + jitter, mode="markers", showlegend=False,
            marker=dict(size=8, color=idt.color(theme), opacity=0.75, line=dict(color=ch["surface"], width=1)),
            customdata=np.stack([g["run_index"], g[column].map(fmt)], axis=-1),
            hovertemplate=f"<b>{idt.label}</b><br>run %{{customdata[0]}}: %{{customdata[1]}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[med[b], med[b]], y=[y_pos - 0.32, y_pos + 0.32], mode="lines", showlegend=False,
            line=dict(color=ch["ink"], width=2), hovertemplate=f"<b>{idt.label}</b><br>median {fmt(med[b])}<extra></extra>",
        ))
        if ceilings and b in ceilings:
            fig.add_trace(go.Scatter(
                x=[ceilings[b], ceilings[b]], y=[y_pos - 0.42, y_pos + 0.42], mode="lines", showlegend=False,
                line=dict(color=STATUS["bad"], width=2, dash="dot"),
                hovertemplate=f"<b>{idt.label}</b><br>ceiling for {ceilings[b] // 100} thread(s): {ceilings[b]}%<extra></extra>",
            ))
    fig.update_layout(**base_layout(theme, height=height or max(320, 40 * len(order) + 90),
                                    margin=dict(l=10, r=16, t=16, b=56), hovermode="closest"))
    fig.update_xaxes(type="log" if log else "linear", title_text=xtitle,
                     **((log_ticks(data[column], tick_fmt) or {}) if log else {}))
    fig.update_yaxes(tickvals=list(range(len(labels))), ticktext=labels, range=[-0.6, len(labels) - 0.4],
                     showgrid=False, ticks="", zeroline=False, automargin=True)
    return fig


def bars(c: Campaign, values: pd.Series, theme: str, xtitle: str, fmt, height=None) -> go.Figure:
    """One horizontal bar per configuration, smallest at the top."""
    values = values.dropna().sort_values(kind="stable")
    if values.empty:
        return _empty(theme, "No values for this selection.")
    ch = CHROME[theme]
    labels = [c.identities[b].label for b in values.index]
    fig = go.Figure(go.Bar(
        x=values.values, y=labels, orientation="h", marker=dict(color=[c.identities[b].color(theme) for b in values.index],
                                                                 line=dict(color=ch["surface"], width=2)),
        text=[fmt(v) for v in values.values], textposition="outside", textfont=dict(color=ch["ink2"], size=12),
        cliponaxis=False, hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(**base_layout(theme, height=height or max(300, 36 * len(values) + 90),
                                    margin=dict(l=10, r=70, t=16, b=56), bargap=0.35))
    fig.update_xaxes(title_text=xtitle)
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], ticks="", showgrid=False, automargin=True)
    return fig


def _overlap_groups(rows: pd.DataFrame, acc_gap=3.0, log_gap=0.12):
    """Points closer than ``acc_gap`` accuracy points and ``log_gap`` decades of
    time, which would hide one another, gathered into groups (largest accuracy
    first)."""
    groups = []
    for r in rows.sort_values(["accuracy", "total_s"], ascending=[False, True]).itertuples():
        for g in groups:
            if any(abs(r.accuracy - q.accuracy) < acc_gap and abs(np.log10(r.total_s / q.total_s)) < log_gap for q in g):
                g.append(r)
                break
        else:
            groups.append([r])
    return groups


def training(c: Campaign, rows: pd.DataFrame, theme: str, chance: float) -> go.Figure:
    """Test accuracy against total training time, the chance line dashed.
    Points that coincide are drawn as ring and dot and share one label; runs
    that ended at chance share one note instead of crowding labels."""
    rows = rows.dropna(subset=["total_s", "accuracy"])
    rows = rows[rows["total_s"] > 0]
    if rows.empty:
        return _empty(theme, "No training runs shown for this budget.")
    ch = CHROME[theme]
    x0, x1 = np.log10(rows["total_s"].min()) - 0.35, np.log10(rows["total_s"].max()) + 0.35
    if x1 - x0 < 2:
        mid = (x0 + x1) / 2
        x0, x1 = mid - 1, mid + 1
    fig = go.Figure()
    fig.add_hline(y=chance, line=dict(color=ch["ink2"], width=1, dash="dash"))
    fig.add_annotation(x=0.005, xref="paper", y=chance, yanchor="bottom", xanchor="left", showarrow=False,
                       text=f"Chance: {chance:g}% for {round(100 / chance)} classes", font=dict(size=11, color=ch["ink2"]))
    groups = _overlap_groups(rows)
    size, draw = {}, []
    for g in groups:
        for i, r in enumerate(g):
            size[r.Index] = 12 if len(g) == 1 else (18, 10, 6)[min(i, 2)]
            draw.append(r)
    rank = {r.Index: i for i, r in enumerate(rows.sort_values(["accuracy", "total_s"], ascending=[False, True]).itertuples())}
    for r in draw:  # biggest of each group first, so the smaller ones sit on top of it
        idt = c.identities[r.backend]
        fig.add_trace(go.Scatter(
            x=[r.total_s], y=[r.accuracy], mode="markers", name=idt.label, legendrank=rank[r.Index] + 1,
            marker=dict(size=size[r.Index], color=idt.color(theme), symbol=idt.symbol,
                        line=dict(color=ch["surface"], width=2)),
            customdata=[[fmt_time(r.total_s), fmt_time(r.time_s) if r.per_epoch else "–", r.rule[:90]]],
            hovertemplate=f"<b>{idt.label}</b><br>accuracy %{{y:.2f}}%<br>total %{{customdata[0]}} · per epoch "
                          f"%{{customdata[1]}}<br>%{{customdata[2]}}<extra></extra>",
        ))
    for g in groups:
        if max(r.accuracy for r in g) <= chance + 2:
            continue
        right = max(g, key=lambda r: r.total_s)
        at_right = (np.log10(right.total_s) - x0) / (x1 - x0) > 0.7
        text = "<br>".join(f"{c.identities[r.backend].label} {r.accuracy:.1f}%" for r in g)
        fig.add_annotation(x=np.log10(right.total_s), y=float(np.mean([r.accuracy for r in g])), text=text,
                           showarrow=False, xanchor="right" if at_right else "left", xshift=-14 if at_right else 14,
                           font=dict(size=12, color=ch["ink"]))
    low = rows[rows["accuracy"] <= chance + 2]
    if len(low):
        short = [c.identities[b].short for b in low.sort_values("total_s")["backend"]]
        names = "<br>".join(" · ".join(short[i:i + 3]) for i in range(0, len(short), 3))
        frac = (np.log10(low["total_s"].median()) - x0) / (x1 - x0)
        side = "right" if frac > 0.6 else "left" if frac < 0.4 else "center"
        x = {"right": low["total_s"].max(), "left": low["total_s"].min(), "center": low["total_s"].median()}[side]
        fig.add_annotation(x=np.log10(x), y=chance + 4, yanchor="bottom", xanchor=side, showarrow=False,
                           text=f"<b>At or below chance ({len(low)})</b><br>{names}", font=dict(size=11, color=ch["ink2"]),
                           align=side)
    layout = base_layout(theme, height=480, hovermode="closest", margin=dict(l=10, r=16, t=16, b=56))
    layout["legend"].update(itemsizing="constant", traceorder="normal")
    fig.update_layout(**layout)
    fig.update_xaxes(type="log", range=[x0, x1], title_text="Total training time (log scale)",
                     tickvals=[1, 10, 60, 600, 3600, 36000, 360000, 3600000],
                     ticktext=["1 s", "10 s", "1 min", "10 min", "1 h", "10 h", "100 h", "1,000 h"])
    fig.update_yaxes(range=[0, 104], title_text="Test accuracy", tickvals=[0, 20, 40, 60, 80, 100],
                     ticktext=[f"{v}%" for v in (0, 20, 40, 60, 80, 100)], automargin=True)
    return fig


def dumbbell(c: Campaign, d: pd.DataFrame, theme: str, a_label: str, b_label: str) -> go.Figure:
    """Each configuration's median in two sessions on the same machine: ring
    for the first, dot for the second, the ratio at the right."""
    if d is None or d.empty:
        return _empty(theme, "Only one session in this database.")
    ch = CHROME[theme]
    d = d.sort_values("ratio", kind="stable")
    labels = [c.identities[b].label for b in d["backend"]]
    fig = go.Figure()
    for i, r in enumerate(d.itertuples()):
        color = c.identities[r.backend].color(theme)
        fig.add_trace(go.Scatter(x=[r.a_s, r.b_s], y=[i, i], mode="lines", line=dict(color=color, width=2),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r.a_s], y=[i], mode="markers", showlegend=False,
                                 marker=dict(size=11, color=ch["surface"], line=dict(color=color, width=2.5)),
                                 hovertemplate=f"<b>{labels[i]}</b><br>{a_label}: {fmt_time(r.a_s)}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[r.b_s], y=[i], mode="markers", showlegend=False,
                                 marker=dict(size=11, color=color, line=dict(color=ch["surface"], width=2)),
                                 hovertemplate=f"<b>{labels[i]}</b><br>{b_label}: {fmt_time(r.b_s)}<extra></extra>"))
        fig.add_annotation(x=1.0, xref="paper", y=i, xanchor="left", showarrow=False, text=f"{r.ratio:.2f}×",
                           font=dict(size=12, color=ch["ink2"]))
    fig.update_layout(**base_layout(theme, height=max(300, 34 * len(d) + 90), margin=dict(l=10, r=60, t=16, b=56)))
    fig.update_xaxes(type="log", title_text=f"Median simulate time at N = {int(d['n'].iloc[0]):,} (log scale)",
                     **(log_ticks(list(d["a_s"]) + list(d["b_s"]), tick_time) or {}))
    fig.update_yaxes(tickvals=list(range(len(labels))), ticktext=labels, showgrid=True, ticks="", zeroline=False,
                     automargin=True)
    return fig
