"""Print the key numbers of an snnbench campaign without starting the dashboard.

    python verify.py                    uses data/benchmarks.db
    python verify.py --db PATH [--session B]

Useful for checking a thesis's figures against its database.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from snndash import analysis as A  # noqa: E402
from snndash.data import load_campaign  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(HERE / "data" / "benchmarks.db"))
    ap.add_argument("--session", default="A", help="session letter (default A)")
    a = ap.parse_args()
    c = load_campaign(a.db)
    session = next((s for s in c.sessions if s.startswith(f"Session {a.session.upper()}")), c.sessions[0])
    print(f"{c.name}: {len(c.runs):,} measured runs, sessions {', '.join(c.sessions)}; using {session}\n")

    sweep = A.view(c, session, c.sweep_task)
    ex = A.exponents(A.aggregate(sweep, "simulate_s")).set_index("backend")
    print("Scaling exponents (simulate time, every size) and the slope between the two largest sizes")
    for b in c.ordered(ex.index):
        print(f"  {c.label(b):<28} k = {ex.loc[b, 'exponent']:6.3f}   last step {ex.loc[b, 'last_step']:5.2f}")

    tasks = [t for t in c.runs["task_label"].unique() if t != c.sweep_task]
    if tasks:
        inf = A.view(c, session, tasks[0])
        groups = {b: g["simulate_s"].values for b, g in inf.groupby("backend")}
        w = A.welch_anova(groups)
        print(f"\nWelch's ANOVA, {tasks[0]}, all configurations: F({w['df1']}, {w['df2']:.2f}) = {w['F']:.2f}, "
              f"p = {w['p']:.3e}, eta^2 = {w['eta2']:.4f}")
        w2 = A.welch_anova({b: v for b, v in groups.items() if b not in c.compile_dominated})
        if c.compile_dominated and w2:
            print(f"  without compile-dominated configurations: F({w2['df1']}, {w2['df2']:.2f}) = {w2['F']:.2f}")
        fails = {c.label(b): round(p, 4) for b, p in w["shapiro"].items() if p < 0.05}
        print(f"  Shapiro-Wilk rejects normality for: {fails or 'none'}")

    print("\nChecks")
    for k in A.checks(c, session):
        print(f"  [{k['level'].upper():4}] {k['tag']}: {k['title']}\n         {k['body']}")


if __name__ == "__main__":
    main()
