"""The Dash application: page layout and callbacks.

Datasets live in a small in-process registry keyed by an id held in a
dcc.Store, so opening another database swaps every view at once. The app is a
local, single-user tool: it binds to 127.0.0.1 by default and only reads files.
"""

from __future__ import annotations

import base64
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from . import analysis as A
from . import figures as F
from .data import TOLERANCE, Campaign, load_campaign
from .theme import CHROME

try:  # Dash >= 2.4
    from dash import ctx as _ctx
except ImportError:  # pragma: no cover
    from dash import callback_context as _ctx

DATASETS: dict = {}
UPLOAD_DIR = Path(tempfile.gettempdir()) / "snn_dashboard_uploads"
ASSETS = Path(__file__).resolve().parent.parent / "assets"
GRAPH_CONFIG = {"displaylogo": False, "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
LEDE = ("Speed, memory and energy are compared only alongside the checks that say whether the simulators "
        "computed the same thing and what each timing really measures.")


def register(campaign: Campaign) -> str:
    key = uuid.uuid4().hex[:12]
    DATASETS[key] = campaign
    return key


def _get(key):
    return DATASETS.get(key) if key else None


# ---------------------------------------------------------------- small views
def panel(title, *children, sub=None, panel_id=None):
    head = [html.H2(title)]
    if sub:
        head.append(html.P(sub, className="sub"))
    extra = {"id": panel_id} if panel_id else {}
    return html.Section(className="panel", children=[html.Div(head, className="panel-head"), *children], **extra)


def seg(component_id, options, value=None):
    return dcc.RadioItems(id=component_id, options=options, value=value, inline=True, className="seg")


def caption(component_id=None, text=""):
    return html.P(text, id=component_id, className="caption") if component_id else html.P(text, className="caption")


def graph(component_id):
    return dcc.Loading(dcc.Graph(id=component_id, config=GRAPH_CONFIG), type="dot", color="#2a78d6")


def data_table(df: pd.DataFrame, columns, theme: str, numeric=(), page_size=None, export=False, table_id=None,
               conditional=None, quiet=()):
    """A sortable table in the page's colours; numbers right-aligned."""
    c = CHROME[theme]
    kwargs = dict(id=table_id) if table_id else {}
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=[{"name": n, "id": i} for i, n in columns],
        sort_action="native", page_action="native" if page_size else "none", page_size=page_size or 1000,
        export_format="csv" if export else "none", export_headers="display",
        style_as_list_view=True, style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": c["surface"], "color": c["ink"], "border": f"1px solid {c['grid']}",
                    "fontFamily": 'system-ui, -apple-system, "Segoe UI", sans-serif', "fontSize": "13px",
                    "fontWeight": "400", "padding": "6px 10px", "textAlign": "left", "whiteSpace": "normal",
                    "height": "auto", "maxWidth": "420px"},
        style_header={"backgroundColor": c["page"], "color": c["ink2"], "fontWeight": "700",
                      "border": f"1px solid {c['grid']}"},
        style_cell_conditional=[{"if": {"column_id": i}, "textAlign": "right",
                                 "fontVariantNumeric": "tabular-nums"} for i in numeric]
                               + [{"if": {"column_id": i}, "color": c["ink2"], "fontSize": "12px", "minWidth": "240px",
                                   "maxWidth": "360px"} for i in quiet],
        style_data_conditional=conditional or [],
        **kwargs,
    )


def chip(label, slot, dash_style):
    return html.Span([html.Span(className=f"swatch slot-{slot} {dash_style}"), label], className="chip-label")


# ---------------------------------------------------------------- layout
def build_layout(key, message, default_db):
    tabs = [
        dcc.Tab(label="Overview", value="overview", children=[
            html.Div(id="kpis", className="kpis"),
            panel("Before reading the rankings", html.Div(id="checks", className="checks-grid"),
                  sub="Checks computed from the data itself. Each limits what a chart can be used to claim."),
        ]),
        dcc.Tab(label="Equivalence", value="equivalence", children=[
            panel("Do the simulators fire the same spikes?", graph("g-equiv"), caption("equiv-cap"),
                  html.Div(id="t-equiv", className="table-wrap"), html.Div(id="equiv-notes", className="notes"),
                  sub="Spike counts against the plain-NumPy loop, one dot per network size. No spike trains were "
                      "stored, so only counts can be compared: agreement is necessary for equivalence, not sufficient."),
        ]),
        dcc.Tab(label="Scaling", value="scaling", children=[
            panel("How runtime grows with network size",
                  seg("scale-metric", [{"label": v[1], "value": k} for k, v in A.METRICS.items()], "simulate"),
                  graph("g-scaling"), caption("scale-cap"), html.Div(id="t-scaling", className="table-wrap"),
                  sub="Medians of the measured runs with the fastest-to-slowest range shaded; both axes logarithmic."),
        ]),
        dcc.Tab(label="Inference (N = 100)", value="inference", children=[
            panel("Inference at the reference size",
                  seg("inf-metric", [{"label": "Simulate", "value": "simulate_s"}, {"label": "Build (setup)", "value": "setup_s"}],
                      "simulate_s"),
                  graph("g-inf"), html.Div(id="inf-stats", className="stats"),
                  html.Div(id="t-inf", className="table-wrap"),
                  sub="Every measured run as a dot, the median as a bar; fastest at the top."),
        ]),
        dcc.Tab(label="Memory", value="memory", children=[
            panel("Peak memory",
                  seg("mem-phase", [{"label": "Higher of build and simulate", "value": "mem_peak_mb"},
                                    {"label": "Build", "value": "mem_setup_mb"}, {"label": "Simulate", "value": "mem_sim_mb"}],
                      "mem_peak_mb"),
                  graph("g-mem"), caption("mem-cap"), html.Div(id="t-mem", className="table-wrap"),
                  sub="Resident memory of the measuring process at its peak."),
        ]),
        dcc.Tab(label="CPU & energy", value="cpu", children=[
            panel("Processor utilisation", graph("g-cpu"), caption("cpu-cap"), panel_id="cpu-panel",
                  sub="Utilisation per run at the focus size, in per cent of one core. Red dotted bars mark what the "
                      "configuration's thread count allows."),
            panel("Energy proxy", graph("g-energy"), caption("energy-cap"),
                  sub="As recorded by the campaign: an estimate, not a measurement."),
        ]),
        dcc.Tab(label="Training", value="training", children=[
            panel("Training: what it cost and what it learned", seg("train-budget", []), graph("g-train"),
                  caption("train-cap"), html.Div(id="t-train", className="table-wrap"),
                  sub="Test accuracy against total training time for each simulator's own learning rule."),
        ]),
        dcc.Tab(label="Sessions", value="sessions", children=[
            panel("Same machine, a second session", graph("g-drift"), caption("drift-cap"),
                  html.Div(id="t-drift", className="table-wrap"),
                  sub="Each configuration's median simulate time in the first two sessions: ring = first, dot = second."),
        ]),
        dcc.Tab(label="Data", value="data", children=[
            panel("Aggregated data", html.Div([html.Button("Download as CSV", id="dl-agg-btn", n_clicks=0, className="btn ghost"),
                                               dcc.Download(id="dl-agg")], className="tools"),
                  html.Div(id="t-agg", className="table-wrap"),
                  sub="One row per configuration and size for the selected session and task."),
            panel("Every measured run", html.Div(id="t-runs", className="table-wrap"),
                  sub="Build and simulate phases of each run side by side. Export with the button above the table."),
        ]),
        dcc.Tab(label="Methods", value="methods", children=[html.Div(id="methods")]),
    ]
    for t in tabs:
        t.className, t.selected_className = "tab", "tab tab--selected"
    return html.Div(id="root", className="theme-light", children=[
        dcc.Store(id="ds-key", data=key),
        html.Header(className="masthead", children=[
            html.Div([
                html.P("Spiking neural network simulators · benchmark dashboard", className="eyebrow"),
                html.H1("SNN Benchmark Dashboard"),
                html.P(LEDE, className="lede"),
            ]),
            html.Div(className="theme-switch", children=[
                html.Span("Theme", className="flabel"),
                seg("theme-choice", [{"label": "Light", "value": "light"}, {"label": "Dark", "value": "dark"}], "light"),
            ]),
        ]),
        html.Section(className="panel loader", children=[
            html.Div(className="loader-row", children=[
                html.Label("Database", htmlFor="db-path", className="flabel"),
                dcc.Input(id="db-path", type="text", value=str(default_db or ""), className="path-input",
                          placeholder=r"D:\For_Lhakpa\...\results\benchmarks.db", spellCheck=False),
                html.Button("Open", id="open-db", n_clicks=0, className="btn"),
                dcc.Upload(id="upload-db", className="upload", multiple=False,
                           children=html.Span(["or drop a ", html.B("benchmarks.db"), " here"])),
            ]),
            html.P(message, id="db-status", className="status"),
            html.P(id="source", className="source"),
        ]),
        html.Div(className="panel filters", children=[
            html.Div([html.Span("Session", className="flabel"), seg("f-session", [])], className="fgroup"),
            html.Div([html.Span("Task", className="flabel"), seg("f-task", [])], className="fgroup"),
            html.Div([html.Span("Focus size", className="flabel"), seg("f-n", [])], className="fgroup"),
            html.Div([html.Span("Simulators", className="flabel"),
                      dcc.Checklist(id="f-backends", options=[], value=[], inline=True, className="chips")],
                     className="fgroup wide"),
        ]),
        dcc.Tabs(id="tabs", value="overview", className="tabs", parent_className="tabs-parent", children=tabs,
                 mobile_breakpoint=0),
        html.Footer(className="footer", children=[
            html.P("Read-only: the database is never modified. Built with the help of an AI assistant (Claude); "
                   "credit it if any part of this tool is used in a thesis or publication."),
        ]),
    ])


# ---------------------------------------------------------------- callbacks
def register_callbacks(app: Dash) -> None:

    @app.callback(Output("root", "className"), Input("theme-choice", "value"))
    def _theme(theme):
        return f"theme-{theme or 'light'}"

    @app.callback(Output("ds-key", "data"), Output("db-status", "children"),
                  Input("open-db", "n_clicks"), Input("upload-db", "contents"),
                  State("db-path", "value"), State("upload-db", "filename"), prevent_initial_call=True)
    def _load(_clicks, contents, path, filename):
        try:
            if _ctx.triggered_id == "upload-db" and contents:
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "upload.db")
                target = UPLOAD_DIR / uuid.uuid4().hex[:8] / safe  # a folder per upload keeps the file's own name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(contents.split(",", 1)[1]))
                c = load_campaign(target)
                return register(c), f"Opened {filename} (uploaded copy, read-only)."
            if not path or not str(path).strip():
                return no_update, "Type the path to a benchmarks.db first."
            clean = str(path).strip().strip('"').strip("'")
            c = load_campaign(clean)
            return register(c), f"Opened {Path(clean)}."
        except FileNotFoundError as e:
            return no_update, f"{e}. Check the path; on Windows you can paste it straight from Explorer's address bar."
        except Exception as e:  # shown to the user, never swallowed
            return no_update, f"Could not open it: {e}"

    @app.callback(
        Output("f-session", "options"), Output("f-session", "value"),
        Output("f-task", "options"), Output("f-task", "value"),
        Output("f-backends", "options"), Output("f-backends", "value"),
        Output("source", "children"), Input("ds-key", "data"))
    def _filters(key):
        c = _get(key)
        if c is None:
            return [], None, [], None, [], [], "No database open yet. Type a path above, or drop a file."
        sessions = [{"label": s, "value": s} for s in c.sessions]
        tasks = [{"label": t, "value": t} for t in sorted(c.runs["task_label"].unique(), key=lambda t: t != c.sweep_task)]
        ids = c.ordered(c.runs["backend"].unique())
        backends = [{"label": chip(c.identities[b].label, c.identities[b].slot, c.identities[b].dash), "value": b} for b in ids]
        m = c.machine
        source = [html.B(c.name), f" · {len(c.runs):,} measured runs in {len(c.sessions)} session(s)",
                  f" + {len(c.training)} training runs" if len(c.training) else "",
                  f" · {str(m.get('cpu', 'CPU')).replace('(R)', '').replace('(TM)', '')}",
                  f", {m.get('logical_cores')} logical cores" if m.get("logical_cores") else "",
                  f" · {c.started} to {c.finished}" if c.started else ""]
        return sessions, (c.sessions[0] if c.sessions else None), tasks, c.sweep_task, backends, ids, source

    @app.callback(Output("f-n", "options"), Output("f-n", "value"),
                  Input("ds-key", "data"), Input("f-task", "value"), State("f-n", "value"))
    def _sizes(key, task, current):
        c = _get(key)
        if c is None:
            return [], None
        ns = sorted(c.runs.loc[c.runs["task_label"] == task, "n"].unique()) if task else sorted(c.runs["n"].unique())
        value = current if current in ns else (ns[-1] if ns else None)
        return [{"label": f"{n:,}", "value": int(n)} for n in ns], (int(value) if value is not None else None)

    common = [Input("ds-key", "data"), Input("f-session", "value"), Input("f-task", "value"),
              Input("f-backends", "value"), Input("theme-choice", "value")]

    @app.callback(Output("kpis", "children"), Output("checks", "children"), *common)
    def _overview(key, session, task, backends, theme):
        c = _get(key)
        if c is None:
            return [], [html.P("Open a database to see its checks.", className="muted")]
        tiles = [html.Div(className="kpi", children=[html.Span(label, className="label"),
                                                      html.Span([value, html.Small(f" {unit}") if unit else None], className="value"),
                                                      html.Span(foot, className="foot")])
                 for label, value, unit, foot in A.kpis(c, session, backends or [])]
        icon = {"bad": "✕", "warn": "!", "good": "✓"}
        cards = [html.Article(className=f"check {k['level']}", children=[
            html.Span([html.Span(icon[k["level"]], className="icon"), k["tag"]], className="tag"),
            html.H3(k["title"]), html.P(k["body"])]) for k in A.checks(c, session)]
        return tiles, cards

    @app.callback(Output("g-equiv", "figure"), Output("equiv-cap", "children"), Output("t-equiv", "children"),
                  Output("equiv-notes", "children"), *common)
    def _equivalence(key, session, task, backends, theme):
        theme = theme or "light"
        c = _get(key)
        if c is None:
            return F._empty(theme, "Open a database."), "", None, None
        f = A.fidelity_rows(c, session)
        f = f[f["backend"].isin(backends or [])]
        fig = F.equivalence(c, f, theme)
        sizes = sorted(f["n"].unique())
        cap = (f"Dot size grows with network size ({', '.join(f'{n:,}' for n in sizes)}). A red ring marks a count "
               f"outside the ±{TOLERANCE:.0%} tolerance. Reference: the plain-NumPy loop, same session.")
        head = [html.Th("Simulator")] + [html.Th(f"{n:,}", className="n") for n in sizes]
        body = []
        for b in c.ordered(f["backend"]):
            cells = [html.Td(chip(c.identities[b].label, c.identities[b].slot, c.identities[b].dash))]
            for n in sizes:
                hit = f[(f["backend"] == b) & (f["n"] == n)]
                if len(hit):
                    e = float(hit["error"].iloc[0])
                    level, text = A.verdict(e)
                    cells.append(html.Td(html.Span(A.signed_pct(e), className=f"cell {level}", title=f"{text}: "
                                                   f"{int(hit['spikes'].iloc[0]):,} spikes against {int(hit['ref_spikes'].iloc[0]):,}"),
                                         className="n"))
                else:
                    cells.append(html.Td("–", className="n"))
            body.append(html.Tr(cells))
        table = html.Table([html.Thead(html.Tr(head)), html.Tbody(body)], className="matrix")
        notes = [html.P(s) for s in A.variant_mismatches(c, session)]
        return fig, cap, table, notes

    @app.callback(Output("g-scaling", "figure"), Output("scale-cap", "children"), Output("t-scaling", "children"),
                  *common, Input("scale-metric", "value"))
    def _scaling(key, session, task, backends, theme, metric):
        theme = theme or "light"
        c = _get(key)
        if c is None:
            return F._empty(theme, "Open a database."), "", None
        col, label, _ = A.METRICS[metric or "simulate"]
        runs = A.view(c, session, task, backends or [])
        agg = A.aggregate(runs, col)
        ex = A.exponents(agg)
        per_run = f" per run of {c.sim_seconds * 1000:g} ms simulated" if np.isfinite(c.sim_seconds) else ""
        fig = F.log_lines(c, agg, ex, theme, f"{label} (log scale)", A.fmt_time)
        cap = (f"{label}{per_run}. The exponent k in the legend is the slope of log time against log N over every size, "
               f"as the source analysis fits it; the table adds the slope between the two largest sizes, which shows "
               f"where one exponent hides a change of regime.")
        rows = []
        for b in c.ordered(agg["backend"]):
            e = ex[ex["backend"] == b]
            for r in agg[agg["backend"] == b].sort_values("n").itertuples():
                rows.append(dict(sim=c.label(b), n=f"{r.n:,}", runs=r.runs, med=A.fmt_time(r.median), lo=A.fmt_time(r.low),
                                 hi=A.fmt_time(r.high), cv=f"{r.cv:.1%}" if np.isfinite(r.cv) else "–",
                                 k=f"{e['exponent'].iloc[0]:.3f}" if len(e) else "–",
                                 last=f"{e['last_step'].iloc[0]:.2f}" if len(e) and np.isfinite(e['last_step'].iloc[0]) else "–"))
        table = data_table(pd.DataFrame(rows), [("sim", "Simulator"), ("n", "N"), ("runs", "Runs"), ("med", "Median"),
                                                ("lo", "Fastest"), ("hi", "Slowest"), ("cv", "CV"), ("k", "Exponent k"),
                                                ("last", "Last step")],
                           theme, numeric=("n", "runs", "med", "lo", "hi", "cv", "k", "last"), page_size=15) if rows else None
        return fig, cap, table

    @app.callback(Output("g-inf", "figure"), Output("inf-stats", "children"), Output("t-inf", "children"),
                  *common, Input("inf-metric", "value"))
    def _inference(key, session, task, backends, theme, metric):
        theme = theme or "light"
        c = _get(key)
        if c is None:
            return F._empty(theme, "Open a database."), None, None
        tasks = [t for t in c.runs["task_label"].unique() if t != c.sweep_task]
        inf_task = tasks[0] if tasks else c.sweep_task
        runs = A.view(c, session, inf_task, backends or [])
        if inf_task == c.sweep_task:
            runs = runs[runs["n"] == runs["n"].min()]
        metric = metric or "simulate_s"
        fig = F.strip(c, runs, metric, theme, ("Simulate" if metric == "simulate_s" else "Build (setup)") + " time per run (log scale)",
                      A.fmt_time)
        groups = {b: g[metric].dropna().values for b, g in runs.groupby("backend")}
        w_all = A.welch_anova(groups)
        w_ex = A.welch_anova({b: v for b, v in groups.items() if b not in c.compile_dominated})
        stats = []
        if w_all:
            stats.append(html.P([html.B("Welch's ANOVA, all configurations: "),
                                 f"F({w_all['df1']}, {w_all['df2']:.2f}) = {w_all['F']:.2f}, p = {w_all['p']:.3g}, "
                                 f"eta² = {w_all['eta2']:.4f}."]))
            if c.compile_dominated & set(groups) and w_ex:
                names = ", ".join(c.label(b) for b in c.compile_dominated & set(groups))
                stats.append(html.P([html.B(f"Without {names} "), "(flagged as timing compilation): ",
                                     f"F({w_ex['df1']}, {w_ex['df2']:.2f}) = {w_ex['F']:.2f}, p = {w_ex['p']:.3g}."]))
            fails = {b: p for b, p in w_all["shapiro"].items() if p < 0.05}
            stats.append(html.P([html.B("Shapiro-Wilk rejects normality for: "),
                                 ", ".join(f"{c.label(b)} (p = {p:.4f})" for b, p in fails.items()) if fails else "none",
                                 ". A difference this large is certain; the effect sizes in the table matter more than p."]))
        med = runs.groupby("backend")[metric].agg(["median", "min", "max", "count"]).sort_values("median")
        best = med["median"].min() if len(med) else np.nan
        table = data_table(pd.DataFrame([dict(sim=c.label(b), med=A.fmt_time(r["median"]), lo=A.fmt_time(r["min"]),
                                              hi=A.fmt_time(r["max"]), runs=int(r["count"]),
                                              x=f"{r['median'] / best:.1f}×" if best > 0 else "–")
                                         for b, r in med.iterrows()]),
                           [("sim", "Simulator"), ("med", "Median"), ("lo", "Fastest"), ("hi", "Slowest"), ("runs", "Runs"),
                            ("x", "× the fastest")], theme, numeric=("med", "lo", "hi", "runs", "x")) if len(med) else None
        return fig, stats, table

    @app.callback(Output("g-mem", "figure"), Output("mem-cap", "children"), Output("t-mem", "children"),
                  *common, Input("mem-phase", "value"))
    def _memory(key, session, task, backends, theme, phase):
        theme = theme or "light"
        c = _get(key)
        if c is None:
            return F._empty(theme, "Open a database."), "", None
        phase = phase or "mem_peak_mb"
        runs = A.view(c, session, task, backends or [])
        agg = A.aggregate(runs, phase)
        fig = F.log_lines(c, agg, pd.DataFrame(), theme, "Peak resident memory (log scale)", A.fmt_mb, bands=False,
                          height=460, tick_fmt=F.tick_mb)
        cap = ("The measuring process is counted whole: configurations that ran inside the shared benchmark process "
               "carry its libraries, while a backend run in its own worker process starts small. Compare growth "
               "within a configuration, not levels between processes.")
        rows = [dict(sim=c.label(r.backend), n=f"{r.n:,}", med=A.fmt_mb(r.median), runs=r.runs)
                for b in c.ordered(agg["backend"]) for r in agg[agg["backend"] == b].sort_values("n").itertuples()]
        table = data_table(pd.DataFrame(rows), [("sim", "Simulator"), ("n", "N"), ("med", "Peak (median)"), ("runs", "Runs")],
                           theme, numeric=("n", "med", "runs"), page_size=15) if rows else None
        return fig, cap, table

    @app.callback(Output("g-cpu", "figure"), Output("cpu-cap", "children"), Output("g-energy", "figure"),
                  Output("energy-cap", "children"), *common, Input("f-n", "value"))
    def _cpu_energy(key, session, task, backends, theme, n):
        theme = theme or "light"
        c = _get(key)
        if c is None:
            e = F._empty(theme, "Open a database.")
            return e, "", e, ""
        runs = A.view(c, session, task, backends or [])
        if n is not None:
            runs = runs[runs["n"] == n]
        ceilings = {b: 100 * k for b, k in c.threads.items()}
        fig_cpu = F.strip(c, runs, "cpu_pct", theme, "Utilisation while simulating (% of one core)",
                          lambda v: f"{v:.0f}%", log=False, ceilings=ceilings)
        def above(g, b):
            return g["cpu_pct"] > 100 * c.threads[b] * 1.05
        over = [f"{c.label(b)} {int(above(g, b).sum())} of {len(g)}"
                for b, g in runs.groupby("backend") if b in c.threads and above(g, b).any()]
        every = A.view(c, session, task, backends or [])
        elsewhere = sorted({int(size) for (b, size), g in every.groupby(["backend", "n"])
                            if b in c.threads and above(g, b).any()} - {n})
        if n is None:
            cpu_cap = ""
        elif over:
            cpu_cap = (f"N = {n:,}. Runs above their red ceiling are physically impossible readings: "
                       + "; ".join(over) + ".")
        else:
            cpu_cap = f"N = {n:,}. No reading exceeds its thread ceiling at this size."
        if elsewhere and n is not None:
            cpu_cap += (f" Readings above a ceiling also occur at N = {', '.join(f'{v:,}' for v in elsewhere)}"
                        " (choose a focus size above).")
        med = runs.groupby("backend")["energy_j"].median()
        fig_e = F.bars(c, med, theme, "Energy per run in joules, estimated (median)", A.fmt_energy)
        r = runs[(runs["energy_j"] > 0) & (runs["cpu_sim_s"] > 0)]
        if len(r):
            ratio = r["energy_j"] / r["cpu_sim_s"]
            e_cap = (f"Every value here equals CPU-seconds × {ratio.mean():.1f} W (spread {ratio.std() / ratio.mean():.2%}): the "
                     f"proxy is CPU time in other units and ranks configurations exactly as CPU time does. No power was measured.")
        else:
            e_cap = "This selection has no energy values."
        return fig_cpu, cpu_cap, fig_e, e_cap

    @app.callback(Output("train-budget", "options"), Output("train-budget", "value"), Input("ds-key", "data"),
                  State("train-budget", "value"))
    def _budgets(key, current):
        c = _get(key)
        if c is None or c.training.empty:
            return [], None
        t = c.training.copy()
        t["work"] = t["train_n"].fillna(0) * t["epochs"].fillna(0)
        budgets = list(t.sort_values(["work", "hidden"], ascending=[False, True])["budget"].drop_duplicates())
        return [{"label": b, "value": b} for b in budgets], (current if current in budgets else budgets[0])

    @app.callback(Output("g-train", "figure"), Output("train-cap", "children"), Output("t-train", "children"),
                  *common, Input("train-budget", "value"))
    def _training(key, session, task, backends, theme, budget):
        theme = theme or "light"
        c = _get(key)
        if c is None or c.training.empty:
            return F._empty(theme, "This database has no training runs."), "", None
        t = c.training[(c.training["budget"] == budget) & c.training["backend"].isin(backends or [])]
        classes = int(c.config.get("MNIST_CLASSES", 10) or 10)
        chance = 100 / classes
        fig = F.training(c, t, theme, chance)
        low = int((t["accuracy"] <= chance + 2).sum())
        whole = sorted({c.label(b) for b in t.loc[t["whole_run"], "backend"]})
        cap = ((f"{low} of {len(t)} configurations finish at or below chance: their time is the cost of a rule that did "
                f"not learn. " if low else "") + "Each simulator trains with its own rule, so these costs compare "
               "algorithms as much as simulators." + (f" {', '.join(whole)}: the recorded time is one closed-form solve, "
                                                     f"so compare total time, not time per epoch." if whole else ""))
        def flags(r):
            out = []
            if r.accuracy <= chance + 2:
                out.append("at chance")
            if r.whole_run:
                out.append("one solve, not per epoch")
            k = c.threads.get(r.backend)
            if k and r.cpu_pct > 100 * k * 1.05:
                out.append(f"CPU above {100 * k}%")
            return ", ".join(out)
        rows = [dict(sim=c.label(r.backend), rule=r.rule, acc=f"{r.accuracy:.2f}%", total=A.fmt_time(r.total_s),
                     epoch=A.fmt_time(r.time_s) if r.per_epoch else "–", cpu=f"{r.cpu_pct:.0f}%", mem=A.fmt_mb(r.mem_mb),
                     energy=A.fmt_energy(r.energy_j), flags=flags(r))
                for r in t.sort_values(["accuracy", "total_s"], ascending=[False, True]).itertuples()]
        table = data_table(pd.DataFrame(rows), [("sim", "Simulator"), ("rule", "Learning rule"), ("acc", "Accuracy"),
                                                ("total", "Total time"), ("epoch", "Per epoch"), ("cpu", "CPU"),
                                                ("mem", "Peak memory"), ("energy", "Energy proxy"), ("flags", "Flags")],
                           theme, numeric=("acc", "total", "epoch", "cpu", "mem", "energy"), quiet=("rule",),
                           conditional=[{"if": {"filter_query": '{flags} contains "chance"', "column_id": "flags"},
                                         "color": "#d03b3b", "fontWeight": "700"}]) if rows else None
        return fig, cap, table

    @app.callback(Output("g-drift", "figure"), Output("drift-cap", "children"), Output("t-drift", "children"), *common)
    def _sessions(key, session, task, backends, theme):
        theme = theme or "light"
        c = _get(key)
        if c is None or c.drift is None:
            return F._empty(theme, "Only one session in this database."), "", None
        d = c.drift[c.drift["backend"].isin(backends or [])]
        a, b = (s.split(" · ")[0] for s in c.sessions[:2])
        fig = F.dumbbell(c, d, theme, a, b)
        cap = (f"{b} repeated {a} on the same machine with the same code. Where a ratio moves away from 1, the ranking "
               f"moved without any change to code or hardware.")
        rows = [dict(sim=c.label(r.backend), a=A.fmt_time(r.a_s), b=A.fmt_time(r.b_s), ratio=f"{r.ratio:.2f}×")
                for r in d.sort_values("ratio", kind="stable").itertuples()]
        table = data_table(pd.DataFrame(rows), [("sim", "Simulator"), ("a", a), ("b", b), ("ratio", f"{a} ÷ {b}")], theme,
                           numeric=("a", "b", "ratio")) if rows else None
        return fig, cap, table

    def _aggregated(c, session, task, backends):
        runs = A.view(c, session, task, backends or [])
        sim = A.aggregate(runs, "simulate_s")
        extra = runs.groupby(["backend", "n"]).agg(peak_mb=("mem_peak_mb", "median"), cpu_s=("cpu_sim_s", "median"),
                                                    cpu_pct=("cpu_pct", "median"), energy_j=("energy_j", "median")).reset_index()
        out = sim.merge(extra, on=["backend", "n"], how="left")
        f = A.fidelity_rows(c, session)[["backend", "n", "error"]] if task == c.sweep_task else pd.DataFrame(columns=["backend", "n", "error"])
        out = out.merge(f, on=["backend", "n"], how="left")
        out = out.assign(_o=out["backend"].map(lambda b: c.identities[b].order)).sort_values(["_o", "n"]).drop(columns="_o")
        out.insert(0, "simulator", out["backend"].map(c.label))
        return out.rename(columns={"median": "simulate_median_s", "low": "simulate_min_s", "high": "simulate_max_s",
                                   "error": "spike_count_error"})

    @app.callback(Output("t-agg", "children"), Output("t-runs", "children"), *common)
    def _data(key, session, task, backends, theme):
        theme = theme or "light"
        c = _get(key)
        if c is None:
            return None, None
        agg = _aggregated(c, session, task, backends)
        show = pd.DataFrame(dict(sim=agg["simulator"], n=agg["n"].map(lambda v: f"{v:,}"), runs=agg["runs"],
                                 med=agg["simulate_median_s"].map(A.fmt_time), cv=agg["cv"].map(lambda v: f"{v:.1%}" if np.isfinite(v) else "–"),
                                 mem=agg["peak_mb"].map(A.fmt_mb), cpu=agg["cpu_pct"].map(lambda v: f"{v:.0f}%" if np.isfinite(v) else "–"),
                                 err=agg["spike_count_error"].map(A.signed_pct)))
        t_agg = data_table(show, [("sim", "Simulator"), ("n", "N"), ("runs", "Runs"), ("med", "Simulate (median)"), ("cv", "CV"),
                                  ("mem", "Peak memory"), ("cpu", "CPU"), ("err", "Spike-count error")], theme,
                           numeric=("n", "runs", "med", "cv", "mem", "cpu", "err"), page_size=20)
        runs = A.view(c, session, task, backends or [])
        cols = ["session", "task_label", "backend", "n", "run_index", "setup_s", "simulate_s", "cpu_pct",
                "mem_setup_mb", "mem_sim_mb", "spikes", "energy_j"]
        raw = runs[cols].assign(_o=runs["backend"].map(lambda b: c.identities[b].order))
        raw = raw.sort_values(["session", "_o", "n", "run_index"]).drop(columns="_o")
        raw["backend"] = raw["backend"].map(c.label)
        for col in ["setup_s", "simulate_s"]:
            raw[col] = raw[col].map(lambda v: float(f"{v:.6g}") if np.isfinite(v) else None)
        for col in ["cpu_pct", "mem_setup_mb", "mem_sim_mb", "energy_j"]:
            raw[col] = raw[col].map(lambda v: float(f"{v:.4g}") if np.isfinite(v) else None)
        t_runs = data_table(raw, [(k, k) for k in cols], theme, numeric=cols[3:], page_size=25, export=True)
        return t_agg, t_runs

    @app.callback(Output("dl-agg", "data"), Input("dl-agg-btn", "n_clicks"), State("ds-key", "data"),
                  State("f-session", "value"), State("f-task", "value"), State("f-backends", "value"),
                  prevent_initial_call=True)
    def _download(_n, key, session, task, backends):
        c = _get(key)
        if c is None:
            return no_update
        agg = _aggregated(c, session, task, backends)
        return dcc.send_data_frame(agg.to_csv, "snn_benchmark_aggregated.csv", index=False)

    @app.callback(Output("methods", "children"), Input("ds-key", "data"))
    def _methods(key):
        c = _get(key)
        if c is None:
            return html.P("Open a database to see how it was measured.", className="muted")
        cfg, m = c.config, c.machine
        def dl(pairs):
            items = []
            for k, v in pairs:
                if v not in (None, "", "None"):
                    items += [html.Dt(k), html.Dd(str(v))]
            return html.Dl(items, className="spec")
        steps = cfg.get("SIM_MS") and cfg.get("DT_MS") and round(cfg["SIM_MS"] / cfg["DT_MS"])
        workload = dl([
            ("Model", f"Leaky integrate-and-fire; N Poisson sources drive N neurons, {cfg.get('FANIN')} inputs per neuron"),
            ("Membrane", f"τ = {cfg.get('TAU_MS')} ms, threshold {cfg.get('V_THRESH')}, reset {cfg.get('V_RESET')} (normalised units)"),
            ("Input", f"{cfg.get('INPUT_RATE_HZ')} Hz Poisson per source, weight {cfg.get('SYN_WEIGHT')}, seed {cfg.get('SEED')}"),
            ("Time", f"dt = {cfg.get('DT_MS')} ms, {cfg.get('SIM_MS')} ms simulated per run ({steps} steps)" if steps else None),
            ("Sizes", ", ".join(f"{n:,}" for n in cfg.get("SIZES", [])) or None),
            ("Repetitions", f"{cfg.get('WARMUP_RUNS')} warm-up runs discarded and {cfg.get('MEASURED_RUNS')} measured per condition"
             if cfg.get("MEASURED_RUNS") else None),
            ("Sessions", "; ".join(c.sessions)),
            ("Training", f"MNIST, {cfg.get('MNIST_INPUTS')} inputs, {cfg.get('MNIST_CLASSES')} classes, {cfg.get('TRAIN_SIM_MS')} ms "
                         f"per sample at dt = {cfg.get('TRAIN_DT_MS')} ms" if len(c.training) else None),
            ("Energy proxy", f"runtime × utilisation × {cfg.get('TDP_CPU_W')} W ({cfg.get('TDP_CPU_SOURCE')})" if cfg.get("TDP_CPU_W") else None),
        ])
        env = dl([("Machine", f"{m.get('hostname', '')} · {str(m.get('cpu', '')).replace('(R)', '').replace('(TM)', '')}"),
                  ("Cores / memory", f"{m.get('logical_cores')} logical cores, {m.get('ram_gb')} GB" if m.get("logical_cores") else None),
                  ("OS", m.get("os")), ("Python", m.get("python")), ("Accelerator", m.get("accelerator")),
                  ("Versions", ", ".join(f"{k} {v}" for k, v in c.packages.items()) or None),
                  ("File", str(c.path))])
        notes = [html.Div([html.Div(chip(c.identities[b].label, c.identities[b].slot, c.identities[b].dash), className="who"),
                           html.P(text)]) for b, text in sorted(c.notes.items(), key=lambda kv: c.identities[kv[0]].order)]
        how = html.Ul([
            html.Li("Medians, ranges and exponents are computed here from the stored runs; nothing is taken from the "
                    "campaign's own summaries."),
            html.Li("Exponent k: least-squares slope of log10(median time) on log10(N) over every size. Last step: the "
                    "slope between the two largest sizes."),
            html.Li("Spike-count error: median count of a configuration minus the median count of the NumPy loop in "
                    "the same session and size, divided by the latter."),
            html.Li("Welch's ANOVA on per-run times at the inference size; Shapiro-Wilk per configuration."),
        ])
        return html.Div(className="methods-grid", children=[
            panel("Workload", workload), panel("Environment", env),
            panel("How the numbers here are computed", how),
            panel("What each simulator was asked to do", html.Div(notes, className="notes")),
        ])


def create_app(default_db=None) -> Dash:
    """Build the Dash app, opening ``default_db`` when it exists."""
    key, message = None, ""
    if default_db:
        try:
            key = register(load_campaign(default_db))
            message = f"Opened {Path(default_db).expanduser().resolve()}."
        except FileNotFoundError:
            message = f"No database at {default_db} yet. Type a path above, or drop a benchmarks.db on the box."
        except Exception as e:
            message = f"Could not open {default_db}: {e}"
    app = Dash(__name__, title="SNN Benchmark Dashboard", assets_folder=str(ASSETS),
               suppress_callback_exceptions=True)
    app.layout = build_layout(key, message, default_db)
    register_callbacks(app)
    return app
