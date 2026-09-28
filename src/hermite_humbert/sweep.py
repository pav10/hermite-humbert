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
import multiprocessing as mp
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


def run_field(D, timeout=0, verify=True, details=None):
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
        r = compute(D, verbose=False, exact=True, verify=verify)
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


def _worker(args):
    return run_field(*args)


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="TSV results table (created or updated)")
    ap.add_argument("fields", nargs="+", help="D values or ranges, e.g. 2-200 46 91,95")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per field (0 = none)")
    ap.add_argument("--details", default=None,
                    help="directory for per-field JSON (default: <out dir>/fields)")
    ap.add_argument("--no-verify", action="store_true", help="skip the outward-rounded re-check")
    ap.add_argument("--redo", action="store_true", help="recompute fields already certified")
    a = ap.parse_args(argv)
    details = a.details or os.path.join(os.path.dirname(os.path.abspath(a.out)), "fields")
    rows = read_table(a.out)
    todo = [D for D in parse_fields(a.fields)
            if a.redo or not rows.get(D, {}).get("status", "").startswith("certified")]
    print("%d fields to compute with %d worker(s)" % (len(todo), a.jobs), flush=True)
    jobs = [(D, a.timeout, not a.no_verify, details) for D in todo]
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(a.jobs, maxtasksperchild=1) as pool:
        for D, row in pool.imap_unordered(_worker, jobs):
            rows[D] = row
            write_table(a.out, rows)
            print("%5d  h=%-2s %-28s %-40s %s  %ss" % (D, row["h_K"], row["gamma"], row["closed_form"][:40],
                                                      row["status"], row["seconds"]), flush=True)


if __name__ == "__main__":
    main()
