"""Read an snnbench campaign database into tidy pandas tables.

The database stores one row per phase of every measured run (``runs``), one
summary row per condition (``conditions``) and the context of each run group
(``meta``). The file is opened read-only and nothing is written back.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .theme import identity

TOLERANCE = 0.35          # snnbench's own equivalence tolerance on spike counts
TASK_NAMES = {"inference": "Inference (N = 100)", "scalability": "Scaling sweep"}
RUN_COLUMNS = {"run_group", "framework", "backend", "task", "network_size", "phase", "run_index", "runtime_s"}


@dataclass
class Campaign:
    path: Path
    runs: pd.DataFrame            # one row per measured run, build and simulate side by side
    fidelity: pd.DataFrame        # spike counts against the NumPy loop, per session and size
    training: pd.DataFrame        # one row per training condition
    drift: Optional[pd.DataFrame] # the first two sessions compared at the largest size
    sessions: list                # session labels in the order they started
    sweep_task: str               # the task the scaling analysis uses
    identities: dict              # backend id -> Identity
    config: dict
    machine: dict
    packages: dict
    notes: dict                   # backend id -> the note recorded with its runs
    compile_dominated: set        # backend ids the source flags as timing compilation
    threads: dict                 # backend id -> thread count, where the name states one
    started: str
    finished: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def sim_seconds(self) -> float:
        """Simulated time per run, from the campaign's configuration."""
        ms = self.config.get("SIM_MS")
        return float(ms) / 1000 if ms else float("nan")

    def ordered(self, ids) -> list:
        """Backend ids in display order: simulators by family, controls last."""
        return sorted(set(ids), key=lambda b: (self.identities[b].order, b))

    def label(self, backend_id: str) -> str:
        return self.identities[backend_id].label


def open_readonly(path) -> sqlite3.Connection:
    """Open a database read-only. A file URI keeps Windows drive letters,
    backslashes and spaces intact."""
    db = Path(path).expanduser().resolve()
    if not db.is_file():
        raise FileNotFoundError(f"No such file: {db}")
    return sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)


def _num(frame: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in frame:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame


def _json(text) -> dict:
    try:
        value = json.loads(text) if isinstance(text, str) else {}
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def load_campaign(path) -> Campaign:
    """Load an snnbench ``benchmarks.db``. Raises ``ValueError`` with a plain
    reason when the file is not one."""
    con = open_readonly(path)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"runs", "conditions"} <= tables:
            raise ValueError("This is not an snnbench database: it needs 'runs' and 'conditions' tables.")
        raw = pd.read_sql_query("SELECT * FROM runs", con)
        cond = pd.read_sql_query("SELECT * FROM conditions", con)
        meta = pd.read_sql_query("SELECT * FROM meta", con) if "meta" in tables else pd.DataFrame()
    finally:
        con.close()
    missing = RUN_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"The runs table lacks columns: {', '.join(sorted(missing))}")

    raw = _num(raw, ["network_size", "run_index", "runtime_s", "peak_rss_mb", "cpu_pct", "spikes", "energy_j"])
    cond = _num(cond, ["network_size", "runtime_median_s", "cpu_pct", "peak_rss_mb", "energy_median_j",
                       "accuracy_pct", "compile_dominated"])
    for frame in (raw, cond):
        frame["key"] = frame["framework"].astype(str) + "/" + frame["backend"].astype(str)
    identities = {identity(k).id: identity(k) for k in pd.concat([raw["key"], cond["key"]]).unique()}
    id_of = {k: identity(k).id for k in pd.concat([raw["key"], cond["key"]]).unique()}

    # sessions: the run groups that hold the simulation tasks, in the order they started
    sim_raw = raw[raw["task"] != "training"].copy()
    if "created_at" not in sim_raw:
        sim_raw["created_at"] = ""
    starts = sim_raw.groupby("run_group")["created_at"].min().sort_values()
    session_of = {g: f"Session {chr(65 + i)} · {str(t)[:16].replace('T', ' ')}" for i, (g, t) in enumerate(starts.items())}
    sessions = list(session_of.values())

    # one row per measured run: build (setup) and simulate side by side
    keys = ["run_group", "key", "task", "network_size", "run_index"]
    want = lambda cols: [c for c in cols if c in sim_raw]
    sim = sim_raw[sim_raw["phase"] == "simulate"][want(keys + ["runtime_s", "peak_rss_mb", "cpu_pct", "spikes", "energy_j"])]
    sim = sim.rename(columns={"runtime_s": "simulate_s", "peak_rss_mb": "mem_sim_mb", "cpu_pct": "cpu_pct"})
    setup = sim_raw[sim_raw["phase"] == "setup"][want(keys + ["runtime_s", "peak_rss_mb", "cpu_pct"])]
    setup = setup.rename(columns={"runtime_s": "setup_s", "peak_rss_mb": "mem_setup_mb", "cpu_pct": "cpu_setup_pct"})
    runs = sim.merge(setup, on=keys, how="left")
    for col in ["mem_sim_mb", "cpu_pct", "spikes", "energy_j", "setup_s", "mem_setup_mb", "cpu_setup_pct"]:
        if col not in runs:
            runs[col] = np.nan
    runs["n"] = runs["network_size"].astype(int)
    runs["backend"] = runs["key"].map(id_of)
    runs["session"] = runs["run_group"].map(session_of)
    runs["task_label"] = runs["task"].map(TASK_NAMES).fillna(runs["task"])
    runs["total_s"] = runs["setup_s"].fillna(0) + runs["simulate_s"]
    runs["cpu_sim_s"] = runs["simulate_s"] * runs["cpu_pct"] / 100
    runs["cpu_total_s"] = runs["cpu_sim_s"] + (runs["setup_s"] * runs["cpu_setup_pct"] / 100).fillna(0)
    runs["mem_peak_mb"] = runs[["mem_setup_mb", "mem_sim_mb"]].max(axis=1)
    runs = runs.drop(columns=["network_size"]).sort_values(["session", "task", "backend", "n", "run_index"]).reset_index(drop=True)

    sweep = TASK_NAMES["scalability"] if (runs["task"] == "scalability").any() else (runs["task_label"].iloc[0] if len(runs) else "")
    fidelity = _fidelity(runs, sweep)
    drift = _drift(runs, sessions, sweep)
    training = _training(cond, id_of)

    first_group = starts.index[0] if len(starts) else None
    row = {}
    if len(meta) and "run_group" in meta:
        hits = meta[meta["run_group"] == first_group] if first_group is not None else meta
        row = (hits if len(hits) else meta).iloc[0].to_dict()
    notes = {}
    for _, c in cond[cond["task"] != "training"].iterrows():
        if isinstance(c.get("notes"), str) and c["notes"].strip():
            notes.setdefault(id_of[c["key"]], c["notes"].strip())
    compile_dominated = {id_of[k] for k in cond.loc[cond["compile_dominated"] == 1, "key"]} if "compile_dominated" in cond else set()
    threads = {}
    for k, b in id_of.items():
        m = re.search(r"threads(\d+)", k)
        if m:
            threads[b] = int(m.group(1))
    stamps = raw["created_at"].dropna().astype(str) if "created_at" in raw else pd.Series(dtype=str)
    return Campaign(
        path=Path(path).expanduser().resolve(), runs=runs, fidelity=fidelity, training=training, drift=drift,
        sessions=sessions, sweep_task=sweep, identities=identities,
        config=_json(row.get("config_json")), machine=_json(row.get("machine_json")),
        packages=_json(row.get("packages_json")), notes=notes, compile_dominated=compile_dominated,
        threads=threads, started=stamps.min()[:16].replace("T", " ") if len(stamps) else "",
        finished=stamps.max()[:16].replace("T", " ") if len(stamps) else "",
    )


def _fidelity(runs: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """Median spike count of each configuration against the plain-NumPy loop,
    per session and size. Counts that agree are necessary for two simulators
    to compute the same thing, not sufficient."""
    rs = runs[runs["task_label"] == sweep]
    ref_id = "numpy_loop" if (rs["backend"] == "numpy_loop").any() else "numpy_vec"
    med = rs.groupby(["session", "backend", "n"])["spikes"].median().rename("spikes").reset_index()
    ref = med[med["backend"] == ref_id][["session", "n", "spikes"]].rename(columns={"spikes": "ref_spikes"})
    out = med.merge(ref, on=["session", "n"], how="inner")
    out = out[out["ref_spikes"] > 0].copy()
    out["error"] = (out["spikes"] - out["ref_spikes"]) / out["ref_spikes"]
    out["reference"] = ref_id
    return out.reset_index(drop=True)


def _drift(runs: pd.DataFrame, sessions: list, sweep: str) -> Optional[pd.DataFrame]:
    """The first two sessions compared at the largest size both reached."""
    if len(sessions) < 2:
        return None
    rs = runs[runs["task_label"] == sweep]
    n_max = rs["n"].max()
    med = rs[rs["n"] == n_max].groupby(["session", "backend"])["simulate_s"].median().unstack(0)
    a, b = sessions[0], sessions[1]
    if a not in med or b not in med:
        return None
    out = med[[a, b]].dropna().rename(columns={a: "a_s", b: "b_s"}).reset_index()
    out["ratio"] = out["a_s"] / out["b_s"]
    out["n"] = int(n_max)
    return out


def _training(cond: pd.DataFrame, id_of: dict) -> pd.DataFrame:
    """Training conditions, with the budget and learning rule parsed from the
    notes snnbench records with each one."""
    rows = []
    for _, c in cond[cond["task"] == "training"].iterrows():
        note = str(c.get("notes") or "")
        kv = dict((k, v.strip()) for k, v in re.findall(r"(\w+)=([^;]*);", note))
        rule = re.sub(r"^(\s*\w+=[^;]*;)+\s*", "", note).strip()
        num = lambda v: pd.to_numeric(v, errors="coerce")
        epochs, wall, t = num(kv.get("epochs")), num(kv.get("util_wall_s")), c.get("runtime_median_s")
        per_epoch = bool(np.isfinite(wall) and epochs and epochs > 0 and t and abs(t - wall / epochs) / t < 0.05)
        whole = bool(not per_epoch and np.isfinite(wall) and t and abs(t - wall) / t < 0.05 and epochs and epochs > 1)
        rows.append(dict(
            backend=id_of[c["key"]], hidden=int(c["network_size"]), preset=kv.get("preset", ""),
            train_n=num(kv.get("train_n")), test_n=num(kv.get("test_n")), epochs=epochs,
            time_s=t, total_s=wall if np.isfinite(wall) else t, per_epoch=per_epoch, whole_run=whole,
            cpu_pct=c.get("cpu_pct"), mem_mb=c.get("peak_rss_mb"), energy_j=c.get("energy_median_j"),
            accuracy=c.get("accuracy_pct"), rule=rule,
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df["budget"] = df.apply(lambda r: f"{int(r.train_n):,} images × {int(r.epochs)} epochs · hidden {r.hidden}"
                                if np.isfinite(r.train_n) and np.isfinite(r.epochs) else f"hidden {r.hidden}", axis=1)
    return df
