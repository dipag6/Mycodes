# SNN Benchmark Dashboard (Python)

A local dashboard for the `benchmarks.db` files that snnbench writes: speed, scaling, memory, CPU, energy
proxy and training results for NEST, Brian2, Nengo, snnTorch, SpikingJelly and the NumPy controls, shown
next to the checks that limit what each ranking can claim. It is written in Python (Dash and Plotly) and
runs on your own computer. It does not upload anything and never changes the database.

## Run it in PyCharm

1. Unzip the folder anywhere, for example `D:\snn_dashboard_python`.
2. In PyCharm, choose **File → Open…**, pick that folder and choose **Trust Project**.
3. Give the project a Python interpreter (Python 3.10 or newer). Click the interpreter name in the
   bottom-right corner, then **Add New Interpreter → Add Local Interpreter… → Virtualenv → New**, and click **OK**.
4. Install the packages. Either click **Install requirements** in the yellow banner PyCharm shows over
   `requirements.txt`, or open PyCharm's **Terminal** tab and run:

   ```
   pip install -r requirements.txt
   ```

5. Open `app.py`. Right-click inside it and choose **Run 'app'**, or use the green ▶ in the gutter.
   The Run window prints `SNN Benchmark Dashboard: http://127.0.0.1:8050/` and your browser opens that page.
6. Stop the dashboard with the red ■ **Stop** button.

The folder includes `data/benchmarks.db` (the final campaign), which opens automatically.

## Open a different database

- Paste its path into the **Database** box at the top and press **Open**. Quotes and spaces in the path are fine,
  so a path copied from Explorer's address bar works as it is.
- Or drag the file onto the **drop a benchmarks.db here** box.
- To open the same file every time: **Run → Edit Configurations… → app**. In **Script parameters**, enter
  `--db "D:\For_Lhakpa\SNN Simulator Benchmark - Final\SNN Simulator Benchmark - Final\results\benchmarks.db"`.
  Setting the `SNN_DB` environment variable to the path does the same.

## From a terminal instead

```
python -m venv .venv
.venv\Scripts\activate            (Windows)
source .venv/bin/activate         (macOS, Linux)
pip install -r requirements.txt
python app.py
python app.py --db "D:\path\to\benchmarks.db" --port 8060 --no-browser
```

## What each tab shows

| Tab | Contents |
|---|---|
| Overview | Four headline figures, and the checks computed from the data that limit what the rankings can claim |
| Equivalence | Spike counts against the plain-NumPy loop at every network size, with the ±35% tolerance and a matrix |
| Scaling | Runtime against network size on log–log axes, with the exponent k and the slope between the two largest sizes. Switch between simulate, build, build + simulate and CPU time |
| Inference (N = 100) | Every measured run as a dot. Welch's ANOVA with and without the compile-dominated configuration, and Shapiro–Wilk tests |
| Memory | Peak resident memory against network size: build, simulate, or the higher of the two |
| CPU & energy | Utilisation per run against each configuration's thread ceiling, and the energy proxy with its exact relation to CPU time |
| Training | Test accuracy against total training time for each training budget, with the chance line |
| Sessions | Each configuration's median in Session A and Session B on the same machine |
| Data | The aggregated table (Download as CSV) and every measured run (Export) |
| Methods | Workload, machine and versions, how each number is computed, and what each simulator was asked to do |

The filters (session, task, focus size and simulators) apply to every tab. The Light/Dark switch is at the
top right.

## Check the numbers without the browser

```
python verify.py                  Session A of data/benchmarks.db
python verify.py --session B
python verify.py --db "D:\path\to\benchmarks.db"
```

This prints the scaling exponents, Welch's ANOVA and every check, so figures quoted in a thesis can be compared
with the database directly.

## If something goes wrong

- **`No module named dash`.** PyCharm is running a different interpreter from the one you installed into. Pick the
  project's virtualenv in the bottom-right corner, then run `pip install -r requirements.txt` again.
- **Port already in use.** An earlier run is still going, so stop it. Or add `--port 8060` to the script parameters.
- **No browser tab opened.** Go to http://127.0.0.1:8050/ yourself.
- **A Windows Firewall prompt appears.** The dashboard listens only on 127.0.0.1 (this computer), so you can cancel it.
- **The console shows "This is a development server".** This is normal for a tool that runs only on your own machine.
- **A large database is slow to drag in.** Open it by path instead; a dragged file is copied first.

## Files

```
app.py              start here
verify.py           prints the key numbers in the console
requirements.txt    dash, plotly, pandas, numpy, scipy
snndash/
  data.py           reads benchmarks.db (read-only) into tables
  analysis.py       medians, exponents, Welch's ANOVA and the checks
  figures.py        the Plotly charts
  layout.py         page layout and callbacks
  theme.py          colours and simulator identities
assets/style.css    page styling (Dash loads it automatically)
data/benchmarks.db  the campaign database (kept out of git)
```

## Tested with

- Python 3.10 with the minimum versions in `requirements.txt` (Dash 2.16, Plotly 5.18, pandas 2.0).
- Python 3.11, 3.13 and a 3.14 release candidate with Dash 4.4, Plotly 7.1 and pandas 3.0.
- `verify.py` printed identical numbers in every environment.
- Every tab was checked in Chromium under both Dash 2.16 and Dash 4.4, at desktop and phone widths, in both themes.

## Credit

This tool was built with the help of an AI assistant (Claude). If any part of it, or a figure from it, is used in a
thesis or publication, acknowledge that assistance as the institution's AI-use policy requires.
