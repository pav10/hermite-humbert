"""All perfect binary Humbert forms of K (multiplicative normalisation) by box
enumeration over the fundamental region.

Setting (as for the certificate of gamma_{K,2}, see certify.py): in each chart the
region R consists of the points p = (x1, x2, s, t) where e1 is a minimal vector, so
G(p) = f_{e1}(p) = e^{-2t}.  For a vector v = (a, b) put N_v = f_v / f_{e1} =
Q1 Q2 with Q_i = (a_i + b_i x_i)^2 + b_i^2 y_i^2; then v is minimal at p in R iff
N_v(p) = 1, and R = {N_v >= 1 for all v}.  A perfect form (Coulangeon) has at
least 5 minimal classes whose gradients differ by a rank-4 set; every perfect
form has a representative in some chart's region (h_K classes, Steinitz charts),
and a second minimal class forces y1 y2 <= 1, i.e. t <= 0.

Algorithm.  Subdivide the compact box [x-parallelogram] x [s_min, s_max] x
[t_lo, 0]; on a box B use the rigorous bounds of certify.py:
  * rule 3: B outside the translation parallelogram -> discard;
  * rule 1: some v with sup_B N_v < 1 -> B lies outside R -> discard;
  * rule 5: fewer than 5 classes v (e1 included) with inf_B N_v <= 1 -> no point of B
    has 5 minimal classes -> discard.
Boxes of hyperbolic width <= w_min that survive are solved: for every 4 candidate
classes (besides e1) Newton is run for N_v = 1 from several starting points of the
box; solutions in the box are polished at dps digits, their full minimal set is
taken from the exact short-vector enumeration (independent of the candidate set),
and perfection and eutaxy are decided (VertexFinder.analyze).  Classes up to
GL_2(O_K) come from analysis.classify_global (Pohst-Wagner, Alg. 11).

Rigour: the discarding rules are certified bounds (floating point with a relative
safety margin, as in certify.py); solving the 4 x 4 systems in the remaining small
boxes is numerical (Newton from several starts), so a missed vertex would need a
Newton failure inside a box of hyperbolic width <= w_min.
"""
import itertools
import math
import time

import mpmath
import numpy as np

from .certify import BranchAndBound
from .vertex import VertexFinder


class PerfectSearch:
    def __init__(self, P, dps=50, w_min=0.05, safety=1e-9, chunk=4096, log=None):
        self.P = P
        self.dps = dps
        self.w_min = w_min
        self.safety = safety
        self.chunk = chunk
        self.log = log or (lambda *a: None)
        # all classes with f_v <= d/2 somewhere on the region (G <= d/2 there)
        self.bb = BranchAndBound(P, P.d / 2.0, delta=0.0)
        self.VF = VertexFinder.__new__(VertexFinder)
        self.VF.P = P
        self.stats = {"boxes": 0, "leaves": 0, "newton": 0, "vertices": 0,
                      "discarded": [0, 0, 0], "vectors": len(self.bb.vecs)}
        self.found = []

    # ------------------------------------------------------------ the search
    def run(self):
        bb, P = self.bb, self.P
        t0 = time.time()
        lo, hi = bb.initial_boxes(0.0, n=(8, 8, 4, 4))
        stack = [(lo, hi, np.arange(len(bb.A1)))]
        while stack:
            lo, hi, vidx = stack.pop()
            if len(lo) > max(8, min(self.chunk, int(2e6 // max(1, len(vidx))))):
                h = len(lo) // 2
                stack.append((lo[h:], hi[h:], vidx))
                stack.append((lo[:h], hi[:h], vidx))
                continue
            self.stats["boxes"] += len(lo)
            out = bb.outside(lo, hi)
            self.stats["discarded"][0] += int(out.sum())
            lo, hi = lo[~out], hi[~out]
            if not len(lo):
                continue
            A1, B1, A2, B2 = bb.A1[vidx], bb.B1[vidx], bb.A2[vidx], bb.B2[vidx]
            y1l, y1h, y2l, y2h = bb.yr(lo, hi)
            x1l, x1h, x2l, x2h = lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]
            NQ = bb._qsup(A1, B1, x1l, x1h, y1h) * bb._qsup(A2, B2, x2l, x2h, y2h)
            r1 = NQ.min(axis=1) * (1 + self.safety) < 1.0
            NI = bb._qinf(A1, B1, x1l, x1h, y1l) * bb._qinf(A2, B2, x2l, x2h, y2l)
            cand = NI <= 1.0 + self.safety
            r5 = (cand.sum(axis=1) < 5) & ~r1
            self.stats["discarded"][1] += int(r1.sum())
            self.stats["discarded"][2] += int(r5.sum())
            keep = ~(r1 | r5)
            if not keep.any():
                continue
            lo, hi, cand = lo[keep], hi[keep], cand[keep]
            y1l, y2l = y1l[keep], y2l[keep]
            # hyperbolic widths (|d phi| <= dx_i / y_i, |du_i|): x_i / y_i, s, t
            w = np.stack([(hi[:, 0] - lo[:, 0]) / y1l, (hi[:, 1] - lo[:, 1]) / y2l,
                          hi[:, 2] - lo[:, 2], hi[:, 3] - lo[:, 3]], axis=1)
            leaf = w.max(axis=1) <= self.w_min
            for i in np.where(leaf)[0]:
                self.solve_leaf(lo[i], hi[i], vidx[cand[i]])
            lo, hi, w, cand = lo[~leaf], hi[~leaf], w[~leaf], cand[~leaf]
            if not len(lo):
                continue
            vidx = vidx[cand.any(axis=0)]
            dim = np.argmax(w, axis=1)
            r = np.arange(len(lo))
            mid = 0.5 * (lo[r, dim] + hi[r, dim])
            lo2, hi1 = lo.copy(), hi.copy()
            hi1[r, dim] = mid
            lo2[r, dim] = mid
            stack.append((np.vstack([lo, lo2]), np.vstack([hi1, hi]), vidx))
        self.stats["time"] = time.time() - t0
        return self.found

    # ------------------------------------------------------------ leaves
    @staticmethod
    def _q(c):
        """box coordinates (x1, x2, s, t) -> q = (x1, x2, u1, u2), u = t -+ s"""
        return np.array([c[0], c[1], c[3] - c[2], c[3] + c[2]])

    def _logN(self, q, idx):
        bb = self.bb
        A1, B1, A2, B2 = bb.A1[idx], bb.B1[idx], bb.A2[idx], bb.B2[idx]
        y1, y2 = math.exp(q[2]), math.exp(q[3])
        L1, L2 = A1 + B1 * q[0], A2 + B2 * q[1]
        T1, T2 = (B1 * y1) ** 2, (B2 * y2) ** 2
        Q1, Q2 = L1 * L1 + T1, L2 * L2 + T2
        val = np.log(Q1) + np.log(Q2)
        g = np.stack([2 * B1 * L1 / Q1, 2 * B2 * L2 / Q2, 2 * T1 / Q1, 2 * T2 / Q2], axis=1)
        return val, g

    def solve_leaf(self, lo, hi, cidx):
        self.stats["leaves"] += 1
        e1 = [i for i in cidx if self.bb.B1[i] == 0 and self.bb.B2[i] == 0]
        others = [i for i in cidx if i not in e1]
        if len(others) < 4:
            return
        qlo, qhi = self._q(lo), self._q(hi)
        # the q-box containing the (x, s, t)-box
        qbox_lo = np.array([lo[0], lo[1], lo[3] - hi[2], lo[3] + lo[2]])
        qbox_hi = np.array([hi[0], hi[1], hi[3] - lo[2], hi[3] + hi[2]])
        starts = [self._q(0.5 * (lo + hi))]
        rng = np.random.default_rng(len(self.found) + self.stats["leaves"])
        for _ in range(3):
            starts.append(self._q(lo + rng.random(4) * (hi - lo)))
        for four in itertools.combinations(others, 4):
            idx = np.array(four)
            for q0 in starts:
                q = self._newton(q0, idx)
                if q is None:
                    continue
                # inside the box (with a small margin, neighbours share boundaries)
                span = qbox_hi - qbox_lo
                if np.any(q < qbox_lo - 0.01 * span) or np.any(q > qbox_hi + 0.01 * span):
                    continue
                self.vertex(q, list(four))
                break

    def _newton(self, q, idx):
        self.stats["newton"] += 1
        q = q.copy()
        for _ in range(40):
            try:
                val, g = self._logN(q, idx)
                dq = np.linalg.solve(g, -val)
            except (np.linalg.LinAlgError, OverflowError, FloatingPointError):
                return None
            n = math.sqrt((dq[0] / math.exp(q[2])) ** 2 + (dq[1] / math.exp(q[3])) ** 2 + dq[2] ** 2 + dq[3] ** 2)
            if n > 0.5:
                dq *= 0.5 / n
            q = q + dq
            if np.max(np.abs(val)) < 1e-13:
                return q
        return None

    def vertex(self, q, four):
        """Polish at dps digits, full minimal set, perfection and eutaxy."""
        for v in self.found:
            if np.max(np.abs(v["qf"] - q) / (1 + np.abs(q))) < 1e-8:
                return
        P, VF = self.P, self.VF
        vecs = [self.bb.vecs[i] for i in four]
        old = mpmath.mp.dps
        mpmath.mp.dps = self.dps + 10
        try:
            qm = mpmath.matrix([mpmath.mpf(float(c)) for c in q])
            for _ in range(60):
                rows = [self._logN_mp(qm, v) for v in vecs]
                Fv = mpmath.matrix([r[0] for r in rows])
                J = mpmath.matrix([r[1] for r in rows])
                try:
                    dq = mpmath.lu_solve(J, -Fv)
                except ZeroDivisionError:
                    # these 4 classes are degenerate at the point (rank < 4): another
                    # 4-subset of the leaf finds the vertex if there is one
                    return
                qm = qm + dq
                if mpmath.norm(dq, mpmath.inf) < mpmath.mpf(10) ** (-(self.dps + 3)):
                    break
            else:
                return
            qf = np.array([float(c) for c in qm])
            p = np.array([qf[0], qf[1], math.exp(qf[2]), math.exp(qf[3])])
            m, _ = P.minimum(p)
            fe1 = 1.0 / (p[2] * p[3])
            if m < fe1 * (1 - 1e-9):
                return                                   # not in the region: e1 not minimal
            cands = [v for v, fv in P.short_vectors(p, fe1 * (1 + 1e-6))]
            vals = [(v, VF.phi_grad_mp(qm, v)) for v in cands]
            mmin = -qm[2] - qm[3]                        # phi_{e1} = -u1 - u2
            tol = mpmath.mpf(10) ** (-(self.dps - 10))
            if any(x[1][0] < mmin - tol for x in vals):
                return
            act = [x for x in vals if abs(x[1][0] - mmin) < tol]
            vert = {"q": qm, "qf": qf, "p": p, "dps": self.dps, "gamma": mpmath.exp(mmin),
                    "vectors": [v for v, _ in act], "grads": [g for _, (phi, g) in act]}
            VF.analyze(vert)
            if not vert["perfect"]:
                return
            self.stats["vertices"] += 1
            self.found.append(vert)
        finally:
            mpmath.mp.dps = old

    def _logN_mp(self, q, v):
        """log N_v and its q-gradient at mp precision."""
        a1, b1, a2, b2 = self.VF._emb_mp(v)
        y1, y2 = mpmath.exp(q[2]), mpmath.exp(q[3])
        L1, L2 = a1 + b1 * q[0], a2 + b2 * q[1]
        T1, T2 = (b1 * y1) ** 2, (b2 * y2) ** 2
        Q1, Q2 = L1 ** 2 + T1, L2 ** 2 + T2
        return mpmath.log(Q1) + mpmath.log(Q2), [2 * b1 * L1 / Q1, 2 * b2 * L2 / Q2, 2 * T1 / Q1, 2 * T2 / Q2]


def perfect_forms(D, dps=50, w_min=0.05, log=None):
    """Classes of perfect binary Humbert forms over Q(sqrt D) up to GL_2(O_K) and
    scaling, with eutaxy, automorphism group order and the search statistics."""
    from .core import _problems, _slabs
    from .analysis import classify_global
    from .voronoi import automorphisms
    charts, h = _problems(D)
    problems = _slabs(charts)
    verts, stats = [], []
    for k, P in enumerate(problems):
        S = PerfectSearch(P, dps=dps, w_min=w_min, log=log)
        for v in S.run():
            v["chart"] = k
            verts.append(v)
        stats.append(S.stats)
        P.release_V()
        if log:
            log("  part %d: %s" % (k, S.stats))
    classes = classify_global(problems, verts) if verts else []
    F = problems[0].F
    for c in classes:
        c["aut"] = len(automorphisms(F, c["rep"]["vectors_OK"]))
    return {"D": D, "h": h, "classes": classes, "stats": stats}


# ---------------------------------------------------------------- records, CLI
def record(r, certified_gamma=None, dps=50):
    """JSON-able record of perfect_forms(D)."""
    from fractions import Fraction
    from .analysis import closed_form_gamma

    def vec(v):
        return [[str(Fraction(c)) for c in x] for x in v]

    classes = []
    for i, c in enumerate(r["classes"]):
        v = c["rep"]
        status = ("extreme" if v["extreme"] else
                  "weakly eutactic" if abs(v["eutaxy_margin"]) <= 1e-9 else "not eutactic")
        cf = closed_form_gamma(v["gamma"], dps=dps)
        classes.append({
            "index": i, "gamma": mpmath.nstr(v["gamma"], dps - 10),
            "closed_form": cf[1], "n_min": len(v["vectors"]), "perf_rank": v["perf_rank"],
            "eutaxy": status, "eutaxy_margin": v["eutaxy_margin"],
            "eutactic_coefficients": [mpmath.nstr(x, 20) for x in v["rho"]] if v.get("rho") else None,
            "aut": c["aut"], "self_conjugate": c["self_conjugate"], "conjugate_of": c["conjugate_of"],
            "copies_found": len(c["members"]), "chart": v["chart"], "point": [float(x) for x in v["qf"]],
            "vectors_OK": [vec(u) for u in v["vectors_OK"]]})
    classes.sort(key=lambda c: -float(c["gamma"]))
    gmax = max((float(c["gamma"]) for c in classes), default=None)
    return {
        "D": r["D"], "h_K": r["h"], "n_perfect": len(classes),
        "n_extreme": sum(c["eutaxy"] == "extreme" for c in classes),
        "n_weakly_eutactic": sum(c["eutaxy"] == "weakly eutactic" for c in classes),
        "n_not_eutactic": sum(c["eutaxy"] == "not eutactic" for c in classes),
        "gamma_max": classes[0]["gamma"][:28] if classes else None,
        "gamma_max_matches_certified": (None if certified_gamma is None or gmax is None
                                        else abs(gmax - certified_gamma) <= 1e-12 * certified_gamma),
        "stats": r["stats"], "status": "COMPLETE", "classes": classes}


def _child(D, w_min, certified_gamma):
    import json
    print(json.dumps(record(perfect_forms(D, w_min=w_min), certified_gamma)))


def main(argv=None):
    """hermite-humbert-perfect [FIELDS] [--table TSV] [--out DIR] [-j N] [--timeout S]:
    all perfect binary Humbert forms (multiplicative normalisation) of each field
    up to GL_2(O_K), with eutaxy (extreme = perfect + eutactic), in parallel (one
    subprocess per field).  Writes OUT/D<D>.json and OUT/summary.tsv."""
    import argparse
    import csv
    import json
    import os
    import subprocess
    import sys
    from .sweep import parse_fields
    ap = argparse.ArgumentParser(prog="hermite-humbert-perfect", description=main.__doc__)
    ap.add_argument("fields", nargs="*", help="D values or ranges; default: the certified fields of --table")
    ap.add_argument("--table", default="results/gamma_K2.tsv")
    ap.add_argument("--out", default="results/perfect")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--w-min", type=float, default=0.05)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        _child(int(a.child[0]), float(a.child[1]), None if a.child[2] == "-" else float(a.child[2]))
        return
    cert = {}
    if os.path.exists(a.table):
        for r in csv.DictReader(open(a.table), delimiter="\t"):
            if r["status"].startswith("certified"):
                cert[int(r["D"])] = float(r["gamma"])
    Ds = parse_fields(a.fields) if a.fields else sorted(cert)
    os.makedirs(a.out, exist_ok=True)
    todo = [D for D in Ds if a.redo or not os.path.exists(os.path.join(a.out, "D%d.json" % D))]
    print("%d fields, %d to compute with %d processes" % (len(Ds), len(todo), a.jobs), flush=True)
    running = []
    while todo or running:
        while todo and len(running) < a.jobs:
            D = todo.pop(0)
            cmd = [sys.executable, "-m", "hermite_humbert.perfect", "--child", str(D), str(a.w_min),
                   str(cert[D]) if D in cert else "-"]
            running.append((subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True),
                            D, time.time()))
        still = []
        for p, D, t0 in running:
            if p.poll() is None and time.time() - t0 < a.timeout:
                still.append((p, D, t0))
                continue
            if p.poll() is None:
                p.kill()
            out, err = p.communicate()
            lines = [ln for ln in out.splitlines() if ln.startswith("{")]
            if p.returncode == 0 and lines:
                rec = json.loads(lines[-1])
            else:
                msg = ("timeout(%ds)" % a.timeout if time.time() - t0 >= a.timeout
                       else "error:" + ((err.strip().splitlines() or ["?"])[-1])[:200])
                rec = {"D": D, "status": msg}
            rec["seconds"] = round(time.time() - t0)
            with open(os.path.join(a.out, "D%d.json" % D), "w") as fh:
                json.dump(rec, fh, indent=1)
            print("  D = %3d  %-10s perfect %-4s extreme %-4s gamma_max ok %s  %ds"
                  % (D, rec["status"][:10], rec.get("n_perfect", "-"), rec.get("n_extreme", "-"),
                     rec.get("gamma_max_matches_certified", "-"), rec["seconds"]), flush=True)
        running = still
        if running:
            time.sleep(0.5)
    write_summary(a.out)


def write_summary(out):
    import glob
    import json
    import os
    rows = sorted((json.load(open(f)) for f in glob.glob(os.path.join(out, "D*.json"))), key=lambda r: r["D"])
    cols = ["D", "h_K", "n_perfect", "n_extreme", "n_weakly_eutactic", "n_not_eutactic", "gamma_max",
            "gamma_max_matches_certified", "status", "seconds"]
    with open(os.path.join(out, "summary.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")


if __name__ == "__main__":
    main()
