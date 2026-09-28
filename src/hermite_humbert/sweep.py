"""Batch computation of gamma_{K,2} over many fields, in parallel.

    hermite-humbert-sweep results/gamma_K2.tsv 2-200 --jobs 4 --timeout 3600
    hermite-humbert-sweep out.tsv 46 91 --timeout 0          # no time limit

One line per field is appended to the TSV file as soon as the field finishes
(so an interrupted sweep loses nothing), and a JSON file with the full result
(all extreme classes, normal forms, minimal vectors, certificate statistics) is
written to <details>/D<D>.json.  Fields whose line already says "certified" are
skipped, so re-running the same command resumes the sweep; fields that timed
out are retried (with the new time limit).
"""
import argparse
import csv
import json
import os
import signal
import sys
import time

import mpmath
import numpy as np

from .field import RealQuadraticField, _is_squarefree

COLUMNS = ["D", "d_K", "h_K", "eps0", "gamma", "closed_form", "#M", "#classes",
           "status", "seconds", "notes"]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def jsonable(x):
    """Convert a result dict (mpmath numbers, numpy arrays, Fractions, tuples)
    into plain JSON types; high-precision reals become decimal strings."""
    from fractions import Fraction
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, mpmath.mpf):
        return mpmath.nstr(x, 50)
    if isinstance(x, mpmath.matrix):
        return [[jsonable(x[i, j]) for j in range(x.cols)] for i in range(x.rows)]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Fraction):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def run_field(D, timeout=0, verify=True, details=None, field_jobs=1):
    """Compute one field; returns (D, TSV row as a dict).  Never raises."""
    from .core import compute
    F = RealQuadraticField(D)
    try:
        eps = F.fmt(F.fundamental_unit())
    except Exception:           # noqa: unit beyond reach
        eps = "?"
    row = dict(D=D, d_K=F.disc, h_K="-", eps0=eps, gamma="-", closed_form="-",
               **{"#M": "-", "#classes": "-"}, status="", seconds=0, notes="")
    t0 = time.time()
    old = signal.signal(signal.SIGALRM, _alarm)
    if timeout:
        signal.alarm(int(timeout))
    try:
        r = compute(D, verbose=False, exact=True, verify=verify, jobs=field_jobs)
        signal.alarm(0)
        ex = r["exact"]
        if ex and ex["proved"]:
            st = "certified" + ("+verified" if verify else "")
        else:
            st = "bounds:%s<=gamma<=%.15f" % (mpmath.nstr(r["gamma"], 15), r["certificate"]["upper"])
        c0 = r["classes"][0]
        row.update({"h_K": r["h"], "gamma": mpmath.nstr(r["gamma"], 25),
                    "closed_form": c0["closed_form"][1] or "?", "#M": c0["n_min"],
                    "#classes": len(r["classes"]), "status": st, "notes": "; ".join(r["notes"])})
        if details:
            os.makedirs(details, exist_ok=True)
            with open(os.path.join(details, "D%d.json" % D), "w") as fh:
                json.dump(jsonable(r), fh, indent=1)
    except _Timeout:
        row["status"] = "timeout(%ds)" % timeout
    except Exception as e:      # noqa
        row["status"] = "error:%s" % repr(e).replace("\t", " ")[:200]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    row["seconds"] = round(time.time() - t0)
    return D, row


def parse_fields(specs):
    """'2-200', '46', '91,95' -> sorted squarefree D > 1."""
    out = set()
    for s in specs:
        for part in s.split(","):
            if "-" in part:
                a, b = part.split("-")
                out.update(range(int(a), int(b) + 1))
            elif part:
                out.add(int(part))
    return sorted(D for D in out if D > 1 and _is_squarefree(D))


def read_table(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {int(r["D"]): r for r in csv.DictReader(fh, delimiter="\t")}


def write_table(path, rows):
    """Rewrite the table sorted by D (one row per field, the latest wins)."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for D in sorted(rows):
            w.writerow(rows[D])
    os.replace(tmp, path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--child":
        return _child(argv[1])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="TSV results table (created or updated)")
    ap.add_argument("fields", nargs="+", help="D values or ranges, e.g. 2-200 46 91,95")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--field-jobs", type=int, default=1,
                    help="processes for the branch-and-bound of each field; use with --jobs 1 "
                         "for fields with a large fundamental unit")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per field (0 = none)")
    ap.add_argument("--details", default=None,
                    help="directory for per-field JSON (default: <out dir>/fields)")
    ap.add_argument("--no-verify", action="store_true", help="skip the outward-rounded re-check")
    ap.add_argument("--redo", action="store_true", help="recompute fields already certified")
    a = ap.parse_args(argv)
    details = a.details or os.path.join(os.path.dirname(os.path.abspath(a.out)), "fields")
    rows = read_table(a.out)

    def pending(D):
        st = rows.get(D, {}).get("status", "")
        if a.redo or not st:
            return True
        if st.startswith("certified"):
            return False
        if st.startswith("timeout(") and a.timeout:
            # already timed out with at least this time limit: retrying cannot help
            return int(st[len("timeout("):].rstrip("s)")) < a.timeout
        return True
    todo = [D for D in parse_fields(a.fields) if pending(D)]
    print("%d fields to compute with %d worker(s)" % (len(todo), a.jobs), flush=True)
    jobs = [(D, a.timeout, not a.no_verify, details, a.field_jobs) for D in todo]

    def record(D, row):
        rows[D] = row
        write_table(a.out, rows)
        print("%5d  h=%-2s %-28s %-40s %s  %ss" % (D, row["h_K"], row["gamma"], row["closed_form"][:40],
                                                  row["status"], row["seconds"]), flush=True)

    run_isolated(jobs, a.jobs, a.timeout, record)


def run_isolated(jobs, n_parallel, timeout, record, grace=120, poll=1.0):
    """Run each field in its own Python subprocess, at most n_parallel at a time.
    The child enforces the time limit itself (SIGALRM, recorded as a timeout row);
    the parent kills a child that overruns it by more than `grace` seconds (e.g.
    stuck in C code), and records a child that dies without a result as an
    error.  Unlike multiprocessing.Pool, a crashed or killed worker can never
    make the sweep wait forever."""
    import subprocess
    import tempfile
    queue = list(jobs)
    running = []                     # (proc, D, start, outfile)
    while queue or running:
        while queue and len(running) < n_parallel:
            args = queue.pop(0)
            out = tempfile.TemporaryFile(mode="w+")
            cmd = [sys.executable, "-m", "hermite_humbert.sweep", "--child", json.dumps(args)]
            running.append((subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL),
                            args[0], time.time(), out))
        still = []
        for proc, D, t0, out in running:
            rc = proc.poll()
            elapsed = time.time() - t0
            if rc is None and timeout and elapsed > timeout + grace:
                proc.kill()
                proc.wait()
                rc = "killed"
            if rc is None:
                still.append((proc, D, t0, out))
                continue
            row = None
            out.seek(0)
            lines = [ln for ln in out.read().splitlines() if ln.startswith("{")]
            out.close()
            if rc == 0 and lines:
                row = json.loads(lines[-1])
            else:
                F = RealQuadraticField(D)
                row = dict(D=D, d_K=F.disc, h_K="-", eps0="?", gamma="-", closed_form="-",
                           **{"#M": "-", "#classes": "-"}, seconds=round(elapsed), notes="")
                row["status"] = ("timeout(%ds)" % timeout if rc == "killed"
                                 else "error:worker exited with code %s" % rc)
            record(D, row)
        running = still
        if running:
            time.sleep(poll)


def _child(argjson):
    D, timeout, verify, details, field_jobs = json.loads(argjson)
    _, row = run_field(D, timeout, verify, details, field_jobs)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
