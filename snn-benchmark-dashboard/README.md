# SNN Benchmark Dashboard

An interactive dashboard for comparing spiking neural network simulators, and a
small pilot benchmark that shows why a speed comparison needs a fidelity check
first.

```
python_dashboard/        the dashboard as a Python (Dash) application for snnbench campaigns
open_dashboard.py        opens the HTML dashboard in a browser (optionally from a benchmarks.db)
dashboard/index.html     the HTML dashboard (open it in any browser; no install)
dashboard/template.html  the same page before the pilot data is embedded
pilot/                   the benchmark harness that produced the pilot data,
                         and export_snnbench.py for snnbench campaign pages
data/                    the pilot's per-run table, fidelity table and dataset
```

## The Python application

`python_dashboard/` is a Dash and Plotly application for snnbench `benchmarks.db`
files. It has the same checks and session comparison as the HTML page, plus
Welch's ANOVA and Shapiro–Wilk tests at the inference size. Install its `requirements.txt` and
run `app.py` (from PyCharm or a terminal). The dashboard is then served on
http://127.0.0.1:8050/. Its README gives the PyCharm steps. The application
reads the database read-only, and the database itself is kept out of git.

## Open the dashboard

Double-click `dashboard/index.html`. It opens with the pilot's measured data.

The dashboard is HTML and JavaScript in one file, so viewing it needs no Python.
From PyCharm (or any Python 3.8+), `open_dashboard.py` opens it in your browser;
with `--db path/to/benchmarks.db` it first builds a page from an snnbench
campaign (`campaign.html`, kept out of git) and opens that instead. Neither
step needs any package installed.

To look at another benchmark's results, choose **Load your results** and pick a
CSV, JSON or SQLite file (for example `results/benchmarks.db` or `runs.csv`
from an `snnbench` campaign). The file is read inside the browser and never
uploaded. You map your columns to roles; only a simulator name, a network size
and one timing column are required. Optional roles cover build and compile
time, CPU time, memory, energy, task, hardware, condition and parity
(share of reference spikes matched, spike-count error). An energy column is
charted only after you say whether it was measured or estimated.

Reading a `.db` file loads the SQLite reader from jsDelivr, so that one path
needs an internet connection; CSV and JSON work offline. Fonts come from
Google Fonts and fall back to system fonts offline.

## Read an snnbench campaign

A `benchmarks.db` written by `snnbench` (tables `runs`, `conditions` and
`meta`) is recognised when loaded and needs no column mapping. To give
someone a page that opens straight on a campaign, embed it:

```
python -m pilot.export_snnbench --db path/to/results/benchmarks.db --out campaign.html
```

Both paths read the stored rows with the same code and recompute nothing
upstream. Every configuration keeps its own line (NEST thread counts and the
NumPy controls are dashed variants), sessions become a filter, and two panels
appear that the pilot does not need:

- **Before reading the rankings** runs checks on the data itself. These are:
  - spike counts outside the campaign's 35% tolerance at any size;
  - one-sided count offsets;
  - variants of one simulator emitting different spikes on the same configuration;
  - CPU readings above what the thread count allows;
  - flat or curving scaling;
  - memory that reflects the measuring process, or a build phase that peaks above the simulation;
  - an energy column that is CPU time times a constant;
  - very short simulated runs;
  - training runs at chance accuracy, and times labelled per epoch that cover a whole run.

  Each check reports the numbers it found.
- **Training** plots test accuracy against total training time, with the
  chance line, and flags each row.

The decision helper is switched off when runs simulate under 0.1 s, because
fixed per-call costs make such timings impossible to extrapolate.

## What the pilot measures

One feedforward layer of leaky integrate-and-fire neurons, driven by 1,000
Poisson inputs at 20 Hz, 100 inputs per neuron, membrane time constant 20 ms,
step 0.1 ms, 1 s simulated, from 500 to 32,000 neurons. Every simulator gets
byte-identical input spikes and connectivity (`pilot/spec.py`).

| Configuration | Library |
| --- | --- |
| `numpy_ref` | hand-written NumPy reference that defines the canonical update |
| `nest` | NEST 3.10 |
| `brian2_cython` | Brian2 2.9 runtime mode (Cython) |
| `brian2_cpp` | Brian2 2.9 C++ standalone mode |
| `nengo` | Nengo 4.1 reference simulator |
| `snntorch` | snnTorch 1.0 on PyTorch 2.14 (CPU) |
| `spikingjelly` | SpikingJelly 0.0.0.0.14 on PyTorch 2.14 (CPU) |

Each configuration runs in two modes:

- **matched**: configured to reproduce the canonical update (one-step synaptic
  delay, input before the threshold test, `>=` threshold, zero initial voltage,
  exact exponential decay);
- **native**: the most direct implementation using the library's defaults.

Every run is a fresh Python process on one thread. Build, prepare
(code generation, compilation, JIT) and simulate are timed separately, with
peak memory and CPU time. Each size gets one warm-up and five measured runs in
matched mode, interleaved across simulators in a seeded random order, plus one
native run for the fidelity table. Output spikes are compared with the
reference (`pilot/fidelity.py`): spike-count error, per-neuron rate
correlation, exact-step match and the coincidence factor of Kistler et al.
(1997) at ±1 ms.

No energy was measured: the container has no RAPL or GPU power counters. The
dashboard's energy panel is an estimate (CPU core-seconds × an adjustable
watts-per-core figure) and says so.

## Rerun it

```
python -m venv .venv && . .venv/bin/activate
pip install -r pilot/requirements.txt
python -m pilot.campaign --out runs/my_machine --sizes 500,1000,2000,4000,8000,16000,32000 --reps 5
python -m pilot.export --campaign runs/my_machine
```

The campaign resumes after an interruption, records a host fingerprint with
every run, and stops if the machine changes underneath it (a cloud session
moved hosts during the first attempt; those partial runs are kept separately
and shown in the dashboard's host comparison).

## Provenance and known limits

The pilot data in `data/` was produced by this harness in a cloud container
(Intel Xeon @ 2.10 GHz, 4 vCPU, 16 GB). It demonstrates the method; it is not
a substitute for results from the thesis's own benchmark campaign.

- CPU time counts child processes as well as the Python process. That matters
  only for Brian2's C++ standalone mode, which compiles with `make` (in
  parallel: about 8.7 s of CPU in 2.8 s of wall time) and runs the model as a
  separate binary. Its runs were repeated after the main campaign once the
  accounting was fixed, so they are not interleaved with the others; their
  first pass is kept in `data/archive_brian2_cpp_first_pass.jsonl`. The two
  passes' medians differ by -10% to +15% with overlapping ranges, which is the
  session-to-session noise on this machine. The compiler's CPU time in the very
  first Brian2 Cython run (cold cache) was recorded before the fix and is
  undercounted.
- Peak memory is the Python process only. For Brian2 C++ it excludes the model
  binary.
- NEST's input arrives from spike generators wired directly to every neuron.
  Relaying it through parrot neurons was 8 to 11% slower
  (`data/nest_idiom_probe.jsonl`, `pilot/nest_idiom_probe.py`), so the direct
  wiring was kept.
