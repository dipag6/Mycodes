"""Statistics and the checks a reader needs before trusting a ranking.

Every check is computed from the loaded data; nothing here is specific to one
campaign's numbers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .data import TOLERANCE, Campaign

METRICS = {
    # key: (column, label, unit)
    "simulate": ("simulate_s", "Simulate time", "s"),
    "setup": ("setup_s", "Build (setup) time", "s"),
    "total": ("total_s", "Build + simulate time", "s"),
    "cpu": ("cpu_sim_s", "CPU time while simulating", "s"),
}


def fmt_time(s) -> str:
    if s is None or not np.isfinite(s):
        return "–"
    if s < 1e-3:
        return f"{s * 1e6:.2g} µs"
    if s < 1:
        return f"{s * 1e3:.3g} ms"
    if s < 60:
        return f"{s:.3g} s"
    if s < 3600:
        return f"{s / 60:.3g} min"
    return f"{s / 3600:.3g} h"


def fmt_mb(mb) -> str:
    if mb is None or not np.isfinite(mb):
        return "–"
    return f"{mb / 1024:.3g} GB" if mb >= 1024 else f"{mb:,.0f} MB"


def fmt_energy(j) -> str:
    if j is None or not np.isfinite(j):
        return "–"
    return f"{j / 1000:.3g} kJ" if j >= 1000 else f"{j:.3g} J"


def signed_pct(x, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{'+' if x >= 0 else '−'}{abs(x) * 100:.{digits}f}%"


def view(c: Campaign, session: Optional[str], task: Optional[str], backends=None) -> pd.DataFrame:
    """The runs a chart draws: one session (or all), one task (or all), the
    selected configurations."""
    r = c.runs
    if session:
        r = r[r["session"] == session]
    if task:
        r = r[r["task_label"] == task]
    if backends is not None:
        r = r[r["backend"].isin(backends)]
    return r


def aggregate(runs: pd.DataFrame, column: str) -> pd.DataFrame:
    """Median, range and spread of one measure per configuration and size."""
    g = runs.dropna(subset=[column]).groupby(["backend", "n"])[column]
    out = g.agg(median="median", low="min", high="max", runs="count", mean="mean", sd="std").reset_index()
    out["cv"] = out["sd"] / out["mean"]
    return out


def fit_exponent(points: pd.DataFrame) -> float:
    """Slope of log(median) against log(N) over every size, as the source
    analysis fits it."""
    p = points[(points["median"] > 0) & (points["n"] > 0)]
    if p["n"].nunique() < 2:
        return float("nan")
    return float(np.polyfit(np.log10(p["n"]), np.log10(p["median"]), 1)[0])


def last_step(points: pd.DataFrame) -> float:
    """Slope between the two largest sizes: where one exponent hides a change
    of regime, this shows it."""
    p = points[(points["median"] > 0)].sort_values("n")
    if len(p) < 3:
        return float("nan")
    a, b = p.iloc[-2], p.iloc[-1]
    return float(np.log(b["median"] / a["median"]) / np.log(b["n"] / a["n"]))


def exponents(agg: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(backend=b, exponent=fit_exponent(g), last_step=last_step(g)) for b, g in agg.groupby("backend")]
    return pd.DataFrame(rows, columns=["backend", "exponent", "last_step"])


def welch_anova(groups: dict) -> dict:
    """Welch's ANOVA for unequal variances, with eta squared, plus the groups
    Shapiro-Wilk rejects. Groups need at least two values each."""
    from scipy import stats

    groups = {k: np.asarray(v, float) for k, v in groups.items() if len(v) >= 2}
    k = len(groups)
    if k < 2:
        return {}
    ns = np.array([len(v) for v in groups.values()])
    means = np.array([v.mean() for v in groups.values()])
    var = np.array([v.var(ddof=1) for v in groups.values()])
    var[var == 0] = np.finfo(float).tiny
    w = ns / var
    mw = (w * means).sum() / w.sum()
    tmp = (((1 - w / w.sum()) ** 2) / (ns - 1)).sum()
    f = ((w * (means - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
    df2 = (k ** 2 - 1) / (3 * tmp)
    allv = np.concatenate(list(groups.values()))
    ssb = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in groups.values())
    sst = ((allv - allv.mean()) ** 2).sum()
    shapiro = {}
    for name, v in groups.items():
        if len(v) >= 3 and np.ptp(v) > 0:
            shapiro[name] = float(stats.shapiro(v).pvalue)
    return dict(F=float(f), df1=k - 1, df2=float(df2), p=float(stats.f.sf(f, k - 1, df2)),
                eta2=float(ssb / sst) if sst else float("nan"), k=k, shapiro=shapiro)


def fidelity_rows(c: Campaign, session: Optional[str]) -> pd.DataFrame:
    s = session or (c.sessions[0] if c.sessions else None)
    f = c.fidelity[c.fidelity["session"] == s]
    return f[~f["backend"].map(lambda b: c.identities[b].is_reference)]


def verdict(error: float, tol: float = TOLERANCE) -> tuple:
    e = abs(error)
    if e <= 0.05:
        return "good", "Count within 5%"
    if e <= tol:
        return "warn", f"Within {tol:.0%}"
    return "bad", f"Outside {tol:.0%}"


def variant_mismatches(c: Campaign, session: Optional[str]) -> list:
    """Variants of one simulator, given the same configuration, should emit
    the same spikes. Returns one sentence per family that does not."""
    f = fidelity_rows(c, session)
    out = []
    for family, prefix in (("NEST", "nest"), ("Brian2", "brian2")):
        fam = f[f["backend"].str.startswith(prefix)]
        for n in sorted(fam["n"].unique()):
            at = fam[fam["n"] == n]
            if at["spikes"].nunique() > 1:
                ordered = at.set_index("backend").loc[c.ordered(at["backend"])]
                listed = ", ".join(f"{int(r.spikes):,} ({c.identities[b].short})" for b, r in ordered.iterrows())
                out.append(f"{family}'s variants emit different spike counts on the same configuration: at N = {n:,}, "
                           f"{listed}, against {int(ordered['ref_spikes'].iloc[0]):,} for the NumPy loop. Their runtimes "
                           f"are not measured on identical work.")
                break
    return out


def checks(c: Campaign, session: Optional[str]) -> list:
    """What the data itself says about the claims it can support. Each item:
    level (bad, warn, good), tag, title, body."""
    out = []
    add = lambda level, tag, title, body: out.append(dict(level=level, tag=tag, title=title, body=body))
    session = session or (c.sessions[0] if c.sessions else None)
    runs = view(c, session, None)
    sweep = runs[runs["task_label"] == c.sweep_task]
    f = fidelity_rows(c, session)
    sizes = sorted(f["n"].unique())

    # equivalence
    outside = f[f["error"].abs() > TOLERANCE]
    if len(outside):
        worst = outside.loc[outside.groupby("backend")["error"].apply(lambda e: e.abs().idxmax())]
        fail_n = sorted(outside["n"].unique())
        ok_n = [n for n in sizes if n not in fail_n]
        other_n = sorted(set(runs.loc[runs["task_label"] != c.sweep_task, "n"]) & set(fail_n))
        listed = "; ".join(f"{c.label(r.backend)} {signed_pct(r.error, 0)} at N = {r.n:,}" for r in worst.itertuples())
        where = (f"Every configuration passes at N = {', '.join(f'{n:,}' for n in ok_n)}, but not at "
                 f"N = {', '.join(f'{n:,}' for n in fail_n)}" if ok_n else "No size passes for every configuration")
        extra = ", the size the separate inference task runs at" if other_n else ""
        add("bad", "Equivalence", f"Outside the {TOLERANCE:.0%} count tolerance",
            f"{listed}. {where}{extra}. A check run at one size does not certify the others.")
    elif len(f):
        add("good", "Equivalence", f"Within {TOLERANCE:.0%} everywhere",
            f"Every configuration's spike count stays within {TOLERANCE:.0%} of the reference at every size.")
    big = sizes[-2:]
    one_sided = []
    for b, g in f.groupby("backend"):
        e = g.set_index("n")["error"].reindex(big)
        if e.notna().all() and len(big) == 2 and ((e < -0.1).all() or (e > 0.1).all()):
            one_sided.append(f"{c.label(b)} {signed_pct(e.iloc[0], 0)} and {signed_pct(e.iloc[1], 0)}")
    if one_sided:
        add("warn", "Equivalence", "A one-sided offset the tolerance absorbs",
            f"{'; '.join(one_sided)} at N = {big[0]:,} and {big[1]:,}. A count that stays on one side of the "
            f"reference points to a modelling difference (threshold rule, delay, input) rather than noise.")
    for sentence in variant_mismatches(c, session):
        add("bad", "Equivalence", "Same configuration, different spikes", sentence)

    # processor utilisation against what the thread count allows
    over = []
    for b, k in sorted(c.threads.items()):
        r = runs[(runs["backend"] == b) & runs["cpu_pct"].notna()]
        hi = r[r["cpu_pct"] > 100 * k * 1.05]
        if len(hi):
            over.append(f"{c.label(b)}: {len(hi)} of {len(r)} runs above {100 * k}%, up to {hi['cpu_pct'].max():.0f}%")
    if over:
        add("bad", "Instrument", "CPU readings above the thread ceiling",
            f"{'; '.join(over)}. A process limited to k threads cannot exceed k × 100% of one core, so these readings "
            f"count something other than the simulator's own threads, and the energy proxy built on them inherits the error.")

    # scaling regimes
    agg = aggregate(sweep, "simulate_s")
    ex = exponents(agg)
    flat = ex[ex["exponent"] < 0.1]
    curve = ex[(ex["exponent"] >= 0.1) & (ex["last_step"] - ex["exponent"] > 0.5)]
    span = f"{sizes[-1] // sizes[0]}-fold" if len(sizes) > 1 else ""
    if len(flat):
        names = ", ".join(f"{c.label(r.backend)} ({r.exponent:.3f})" for r in flat.itertuples())
        add("warn", "Scaling", "Runtime that does not respond to size",
            f"{names}: a {span} larger network costs about the same, so the sweep never leaves their fixed per-run "
            f"cost. A near-zero exponent here means overhead dominates, not that they scale best.")
    if len(curve):
        names = "; ".join(f"{c.label(r.backend)} {r.exponent:.2f} overall, {r.last_step:.2f} at the top" for r in curve.itertuples())
        add("warn", "Scaling", "One exponent hides a change of regime",
            f"{names} (between N = {big[0]:,} and {big[-1]:,}). Growth steepens at the top of the sweep, so the "
            f"overall exponent underestimates the cost of larger networks.")

    # memory
    if sizes:
        at0 = sweep[sweep["n"] == sizes[0]].groupby("backend")["mem_peak_mb"].median()
        refs = at0[[c.identities[b].is_reference for b in at0.index]]
        sims = at0[[not c.identities[b].is_reference for b in at0.index]]
        if len(refs) and len(sims) and refs.max() > 2 * sims.min():
            add("warn", "Memory", "Peak memory measures the process, not the simulator",
                f"At N = {sizes[0]:,} the plain-NumPy control, which allocates almost nothing, peaks at "
                f"{fmt_mb(refs.max())}, while {c.label(sims.idxmin())} peaks at {fmt_mb(sims.min())}. The figure "
                f"includes whatever else the measuring process had loaded, so gaps between simulators measured in "
                f"different processes are not theirs.")
        top = sweep[sweep["n"] == sizes[-1]].groupby("backend")[["mem_setup_mb", "mem_sim_mb"]].median()
        gaps = top[top["mem_setup_mb"] > 1.3 * top["mem_sim_mb"]]
        if len(gaps):
            listed = "; ".join(f"{c.label(b)} {fmt_mb(r.mem_setup_mb)} while building against {fmt_mb(r.mem_sim_mb)} "
                               f"while simulating" for b, r in gaps.iterrows())
            add("warn", "Memory", "Building the network needs more memory than simulating it",
                f"At N = {sizes[-1]:,}: {listed}. A peak quoted from the simulate phase alone understates the memory "
                f"needed to run the model; the memory chart here uses the higher of the two.")

    # energy
    r = runs[(runs["energy_j"] > 0) & (runs["cpu_sim_s"] > 0)]
    if len(r) > 10:
        ratio = r["energy_j"] / r["cpu_sim_s"]
        if ratio.std() / ratio.mean() < 0.01:
            add("warn", "Energy", "The energy column is CPU time in other units",
                f"Every recorded energy value equals CPU-seconds × {ratio.mean():.1f} W (spread "
                f"{ratio.std() / ratio.mean():.2%}). It cannot rank configurations differently from CPU time, and its "
                f"joules charge a whole processor's rated power to each busy core.")

    # workload scale
    sim_ms = c.config.get("SIM_MS")
    if sim_ms and len(agg):
        quick = int((agg["median"] < 0.2).sum())
        add("warn" if quick > len(agg) / 2 else "good", "Workload", f"{sim_ms:g} ms simulated per run",
            f"{quick} of {len(agg)} sweep conditions finish in under 200 ms, where fixed per-call costs weigh as much "
            f"as the simulation itself. Rankings at this scale do not extrapolate to longer simulations.")

    # training
    t = c.training
    if len(t):
        classes = int(c.config.get("MNIST_CLASSES", 10) or 10)
        chance = 100 / classes
        low = t[t["accuracy"] <= chance + 2]
        if len(low):
            fams = sorted({c.label(b).split(" · ")[0] for b in low["backend"]})
            add("bad", "Training", "Training runs that did not learn",
                f"{len(low)} of {len(t)} training runs ({', '.join(fams)}) finish at or below chance: "
                f"{low['accuracy'].min():.1f}–{low['accuracy'].max():.1f}% for {classes} classes. Their time per epoch "
                f"is the cost of a rule that did not learn, and each simulator trains with a different rule, so "
                f"training cost compares algorithms as much as simulators.")
        whole = sorted({c.label(b) for b in t.loc[t["whole_run"], "backend"]})
        if whole:
            add("warn", "Training", "A time per epoch that is not per epoch",
                f"{', '.join(whole)}: the recorded time equals the whole training run, one closed-form solve, while "
                f"the other rows are per epoch. Compare total training time instead.")

    rank = {"bad": 0, "warn": 1, "good": 2}
    return sorted(out, key=lambda o: rank[o["level"]])


def kpis(c: Campaign, session: Optional[str], backends) -> list:
    """Four headline tiles: label, value, unit, foot."""
    tiles = []
    f = fidelity_rows(c, session)
    f = f[f["backend"].isin(backends)]
    if len(f):
        per = f.groupby("backend")["error"].apply(lambda e: e.abs().max())
        at500 = f[f["n"] == 500].set_index("backend")["error"].abs()
        foot = f"Within 5% at every size: {(per <= 0.05).sum()} of {len(per)}"
        if len(at500):
            foot += f" · at N = 500 alone: {(at500 <= TOLERANCE).sum()} of {len(at500)}"
        tiles.append((f"Spike counts within {TOLERANCE:.0%} of the reference at every size",
                      f"{(per <= TOLERANCE).sum()} of {len(per)}", "", foot))
    runs = view(c, session, c.sweep_task, backends)
    if len(runs):
        n_max = runs["n"].max()
        med = runs[runs["n"] == n_max].groupby("backend")["simulate_s"].median().sort_values()
        sims = med[[not c.identities[b].is_reference for b in med.index]]
        refs = med[[c.identities[b].is_reference for b in med.index]]
        if len(sims):
            foot = f"{fmt_time(sims.iloc[0])} per run"
            if len(sims) > 1 and sims.iloc[1] / sims.iloc[0] < 1.05:
                foot += f", level with {c.label(sims.index[1])} (within 5%)"
            if len(refs):
                foot += f" · NumPy control {fmt_time(refs.min())}"
            tiles.append((f"Fastest simulator at N = {n_max:,}", c.label(sims.index[0]), "", foot))
        mem = runs.groupby(["backend", "n"])["mem_peak_mb"].median()
        if mem.notna().any():
            (b, n), v = mem.idxmax(), mem.max()
            tiles.append(("Largest peak memory", fmt_mb(v), "", f"{c.label(b)} at N = {n:,}, build phase included"))
    d = c.drift
    if d is not None and len(d[d["backend"].isin(backends)]):
        d = d[d["backend"].isin(backends)]
        dev = np.maximum(d["ratio"], 1 / d["ratio"])
        w = d.loc[dev.idxmax()]
        tiles.append((f"{c.sessions[0].split(' · ')[0]} against {c.sessions[1].split(' · ')[0]}, same machine",
                      f"{dev.max():.2f}×", "apart",
                      f"Largest gap: {c.label(w['backend'])} at N = {int(w['n']):,}; median gap {np.median(dev):.2f}× "
                      f"over {len(d)} configurations"))
    return tiles
