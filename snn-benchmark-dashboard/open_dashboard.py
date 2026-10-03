"""Open the dashboard in your default browser.

    python open_dashboard.py                          the pilot data (dashboard/index.html)
    python open_dashboard.py --db path/to/benchmarks.db
                                                      build a page from an snnbench campaign
                                                      database, then open it

Nothing needs installing: the page is plain HTML and JavaScript, and building
a campaign page uses only the Python standard library.
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="an snnbench benchmarks.db to show instead of the pilot data")
    ap.add_argument("--out", default=str(HERE / "campaign.html"),
                    help="where to write the campaign page (default: campaign.html here)")
    a = ap.parse_args()
    page = HERE / "dashboard" / "index.html"
    if a.db:
        subprocess.run([sys.executable, "-m", "pilot.export_snnbench", "--db", a.db, "--out", a.out],
                       cwd=HERE, check=True)
        page = Path(a.out).resolve()
    print(f"opening {page}")
    if not webbrowser.open(page.as_uri()):
        print("no browser found; open the file above by hand")


if __name__ == "__main__":
    main()
