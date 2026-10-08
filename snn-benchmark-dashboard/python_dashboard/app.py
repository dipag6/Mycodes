"""SNN Benchmark Dashboard: run this file.

    python app.py                       opens data/benchmarks.db if it is there
    python app.py --db "D:\\path\\to\\benchmarks.db"
    python app.py --port 8060 --no-browser

In PyCharm: right-click app.py and choose Run. The dashboard opens in your
browser at http://127.0.0.1:8050/; stop it with PyCharm's Stop button.
"""

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from snndash.layout import create_app  # noqa: E402

DEFAULT_DB = HERE / "data" / "benchmarks.db"


def main():
    ap = argparse.ArgumentParser(description="SNN Benchmark Dashboard")
    ap.add_argument("--db", default=os.environ.get("SNN_DB", str(DEFAULT_DB)),
                    help="an snnbench benchmarks.db (default: data/benchmarks.db, or the SNN_DB variable)")
    ap.add_argument("--host", default="127.0.0.1", help="address to serve on (default: this computer only)")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = ap.parse_args()

    app = create_app(a.db)
    url = f"http://{a.host}:{a.port}/"
    print(f"SNN Benchmark Dashboard: {url}")
    print("Stop it with Ctrl+C (or PyCharm's red Stop button).")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    run = getattr(app, "run", None) or app.run_server
    run(host=a.host, port=a.port, debug=False)


if __name__ == "__main__":
    main()
