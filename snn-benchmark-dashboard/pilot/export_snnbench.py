"""Build a dashboard page for one snnbench campaign database.

    python -m pilot.export_snnbench --db results/benchmarks.db --out campaign.html
        [--title "..."] [--lede "..."]

The page embeds the database's runs, conditions and meta tables as they are
stored, and reads them with the same code that opens a benchmarks.db through
"Load your results", so both paths show the same numbers. Nothing is
recomputed or filtered here.
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "dashboard", "template.html")
# the columns the page reads; row ids and the long machine strings stay out
RUN_COLS = ["run_group", "framework", "backend", "task", "network_size", "phase",
            "run_index", "runtime_s", "peak_rss_mb", "cpu_pct", "spikes", "energy_j",
            "created_at"]
COND_COLS = ["run_group", "framework", "backend", "task", "network_size", "phase",
             "runtime_median_s", "cpu_pct", "peak_rss_mb", "energy_median_j",
             "accuracy_pct", "compile_dominated", "notes"]
META_COLS = ["run_group", "machine_json", "packages_json", "config_json"]


def table(con, name, cols, tail=""):
    have = {r[1] for r in con.execute(f'PRAGMA table_info("{name}")')}
    use = [c for c in cols if c in have]
    rows = [list(r) for r in con.execute(f'SELECT {", ".join(use)} FROM "{name}" {tail}')]
    return {"columns": use, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--lede", default=None)
    a = ap.parse_args()

    db = Path(a.db).resolve()
    if not db.is_file():
        raise SystemExit(f"no such file: {db}")
    # a proper file URI keeps Windows drive letters, backslashes and spaces intact;
    # mode=ro guarantees the campaign database is never written to
    con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = {"runs", "conditions"} - names
    if missing:
        raise SystemExit(f"{a.db} is not an snnbench database (no {', '.join(sorted(missing))} table)")
    tables = {"runs": table(con, "runs", RUN_COLS, "ORDER BY id"),
              "conditions": table(con, "conditions", COND_COLS, "ORDER BY id")}
    if "meta" in names:
        # machine, packages and configuration repeat within a run group; keep its first row
        tables["meta"] = table(con, "meta", META_COLS,
                               "WHERE id IN (SELECT MIN(id) FROM meta GROUP BY run_group)")
    dataset = {"kind": "snnbench-db", "title": a.title or os.path.basename(a.db), "tables": tables}
    if a.lede:
        dataset["lede"] = a.lede

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if "/*__PILOT_DATA__*/null" not in html:
        raise SystemExit("dashboard/template.html has no data placeholder")
    blob = json.dumps(dataset, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__PILOT_DATA__*/null", blob)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}: {len(tables['runs']['rows'])} run rows, "
          f"{len(tables['conditions']['rows'])} conditions, "
          f"{os.path.getsize(a.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
