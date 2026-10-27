"""Perfect binary Humbert forms in the multiplicative normalisation, by a Voronoi walk.

Forms S = (S1, S2), det S1 = det S2 = 1, are the points q = (x1, x2, u1, u2) of
H x H (S_i = [[1, x_i], [x_i, x_i^2 + y_i^2]] / y_i, y_i = e^{u_i}); a vector v in
O_K^2 has f_v(q) = S1[v1] S2[v2] and phi_v = log f_v.  G(q) = min_v f_v is the
multiplicative Hermite-Humbert invariant (Icaza; the gamma_{K,2} of this package).
A form is *perfect* (Coulangeon) iff its minimal vectors determine it, i.e. the
gradients of phi_v, v in M(S), differ by a rank-4 set: q is a vertex of the cell
complex of H x H by minimal classes.  Its edges are the curves on which 4 minimal
classes (differences of rank 3) stay tied and minimal.  Starting from one perfect
form, every edge is followed until a new vector becomes minimal (the neighbour);
neighbours are classified up to GL_2(O_K) by their minimal vectors (Pohst-Wagner,
Alg. 11) until no new class appears.  Eutaxy (hence extremality = perfect +
eutactic) is decided for each class as in VertexFinder.analyze.

This is the multiplicative theory; it is never mixed with the additive (trace,
Koecher/Leibak) Voronoi theory of pav10/perfect_forms, whose perfect forms differ.

Rigour.  Every vertex is polished at `dps` digits and its full minimal set is
obtained from the exact short-vector enumeration (independent of any candidate set);
classes are compared exactly.  Following an edge is numerical: vectors more than a
factor `far` above the minimum cannot become minimal within a step (|grad phi_v| <= 2
in the hyperbolic metric for every v), and nearer vectors are watched at every
step and at the midpoint of every step (default step 0.05 in the hyperbolic
metric).  The certificate `edge_balance` (N_ij / |Aut_i| = N_ji / |Aut_j|) checks the
whole graph; completeness rests on the connectedness of the Voronoi graph.
"""
import itertools
import math
import time
from fractions import Fraction

import mpmath
import numpy as np

from .analysis import cols, m_det, m_inv, m_mul, m_vec, associated_vec
from .humbert import HumbertProblem
from .vertex import VertexFinder


class WalkError(RuntimeError):
    pass


def _key(v):
    return tuple(tuple(Fraction(c) for c in x) for x in v)


def automorphisms(F, M):
    """Elements U of GL_2(O_K) with U M = M (as unit classes), modulo scalar units
    (which fix every form).  Variant of analysis.equivalence collecting all U."""
    M = [_key(v) for v in M]
    a1 = M[0]
    a2 = next(v for v in M[1:] if m_det(F, cols(a1, v)) != (0, 0))
    a3 = next(v for v in M if v is not a1 and v is not a2 and
              all(c != (0, 0) for c in m_vec(F, m_inv(F, cols(a1, a2)), v)))
    A = cols(a1, a2)
    lam = m_vec(F, m_inv(F, A), a3)
    found = []
    for b1, b2 in itertools.permutations(M, 2):
        B = cols(b1, b2)
        if m_det(F, B) == (0, 0):
            continue
        Binv = m_inv(F, B)
        for b3 in M:
            if b3 is b1 or b3 is b2:
                continue
            mu = m_vec(F, Binv, b3)
            if mu[0] == (0, 0) or mu[1] == (0, 0):
                continue
            e1, e2 = F.div(lam[0], mu[0]), F.div(lam[1], mu[1])
            if not (F.is_unit(e1) and F.is_unit(e2)):
                continue
            Dg = ((e1, (0, 0)), ((0, 0), e2))
            U = m_mul(F, m_mul(F, A, Dg), Binv)          # U b_i ~ a_i
            if not all(F.is_integral(U[i][j]) for i in range(2) for j in range(2)):
                continue
            if not F.is_unit(m_det(F, U)):
                continue
            # U maps the class of b_i to that of a_i; we need U^{-1} M = M as well
            if not all(any(associated_vec(F, s, m_vec(F, U, w)) for s in M) for w in M):
                continue
            if not any(_scalar_multiple(F, U, W) for W in found):
                found.append(U)
    return found


def _scalar_multiple(F, U, W):
    """U = e W for a unit e?"""
    for i in range(2):
        for j in range(2):
            if W[i][j] != (0, 0):
                e = F.div(U[i][j], W[i][j])
                if not F.is_unit(e):
                    return False
                return all(F.mul(e, W[k][l]) == tuple(Fraction(c) for c in U[k][l])
                           for k in range(2) for l in range(2))
    return False


class MultiplicativeVoronoi:
    """Voronoi walk over the perfect forms of O_K^2 (any class number)."""

    def __init__(self, D, dps=50, far=1.5, step=0.05, max_classes=2000, max_steps=20000, log=None):
        self.P = P = HumbertProblem(D, require_h1=False)
        self.F = P.F
        self.D = D
        self.dps = dps
        self.far = far
        self.step = step
        self.max_classes = max_classes
        self.max_steps = max_steps
        self.log = log or (lambda *a: None)
        self.VF = VertexFinder.__new__(VertexFinder)   # only the mp/analysis routines (no V)
        self.VF.P = P
        self.p_ref = np.array([0.0, 0.0, 1.0, 1.0])
        self.classes = []
        self.stats = {"edges": 0, "steps": 0, "short_calls": 0, "unbounded": 0, "tangencies": 0}

    # ------------------------------------------------------------- float kernel
    def same(self, v, w):
        """v, w in the same unit class (exact)."""
        return self.P.unit_associated(_key(v), _key(w))

    def member(self, v, reps):
        return any(self.same(v, w) for w in reps)

    @staticmethod
    def _p(q):
        return np.array([q[0], q[1], math.exp(q[2]), math.exp(q[3])])

    def phig(self, q, vecs):
        A1, B1, A2, B2 = self.P.vec_emb(vecs)
        x1, x2, u1, u2 = [float(c) for c in q]
        y1, y2 = math.exp(u1), math.exp(u2)
        L1, L2 = A1 + B1 * x1, A2 + B2 * x2
        T1, T2 = (B1 * y1) ** 2, (B2 * y2) ** 2
        Q1, Q2 = L1 * L1 + T1, L2 * L2 + T2
        phi = np.log(Q1) + np.log(Q2) - u1 - u2
        g = np.stack([2 * B1 * L1 / Q1, 2 * B2 * L2 / Q2, 2 * T1 / Q1 - 1, 2 * T2 / Q2 - 1], axis=1)
        return phi, g

    def short(self, q, bound_factor, m=None):
        """Unit classes with f_v <= bound_factor * m at q (exact enumeration)."""
        p = self._p(q)
        if m is None:
            m, _ = self.P.minimum(p)
        self.stats["short_calls"] += 1
        return [v for v, fv in self.P.short_vectors(p, m * bound_factor)]

    def others(self, q, reps, m):
        """Classes not among `reps` with f_v <= far * m (more if there are none)."""
        fac = self.far
        while True:
            c = [v for v in self.short(q, fac, m) if not self.member(v, reps)]
            if c:
                return c
            fac *= 2

    @staticmethod
    def hnorm(q, t):
        return math.sqrt((t[0] / math.exp(q[2])) ** 2 + (t[1] / math.exp(q[3])) ** 2 + t[2] ** 2 + t[3] ** 2)

    # ------------------------------------------------------------- vertices
    def polish(self, q0, five):
        """Newton at dps digits on 5 vectors (4 equations); returns a vertex dict
        with the full minimal set, or None if the 5 are not minimal there."""
        VF = self.VF
        old = mpmath.mp.dps
        mpmath.mp.dps = self.dps + 10
        try:
            q = mpmath.matrix([mpmath.mpf(float(c)) for c in q0])
            for _ in range(80):
                vals = [VF.phi_grad_mp(q, v) for v in five]
                Fv = mpmath.matrix([vals[i][0] - vals[0][0] for i in range(1, 5)])
                J = mpmath.matrix([[vals[i][1][k] - vals[0][1][k] for k in range(4)] for i in range(1, 5)])
                dq = mpmath.lu_solve(J, -Fv)
                q = q + dq
                if mpmath.norm(dq, mpmath.inf) < mpmath.mpf(10) ** (-(self.dps + 3)):
                    break
            else:
                return None
            qf = np.array([float(c) for c in q])
            cand = self.short(qf, 1 + 1e-6)
            allv = list(cand) + [v for v in five if not self.member(v, cand)]
            vals = [(v, VF.phi_grad_mp(q, v)) for v in allv]
            mmin = min(x[1][0] for x in vals)
            tol = mpmath.mpf(10) ** (-(self.dps - 10))
            act = [x for x in vals if x[1][0] - mmin < tol]
            if any(not self.member(v, [w for w, _ in act]) for v in five):
                return None
            vert = {"q": q, "qf": qf, "dps": self.dps, "gamma": mpmath.exp(mmin),
                    "vectors": [v for v, _ in act], "grads": [g for _, (phi, g) in act]}
            VF.analyze(vert)
            return vert
        finally:
            mpmath.mp.dps = old

    def rays(self, vert):
        """Edges at a perfect vertex: (T, d) with T a set of 4 minimal classes whose
        gradient differences have rank 3, d the kernel direction along which T stays
        minimal among all minimal classes; deduplicated by the tied set."""
        M = vert["vectors"]
        G = np.array([[float(c) for c in g] for g in vert["grads"]])
        n = len(M)
        out, seen = [], set()
        for T in itertools.combinations(range(n), 4):
            Dm = G[list(T[1:])] - G[T[0]]
            if np.linalg.matrix_rank(Dm, tol=1e-9) != 3:
                continue
            d = np.linalg.svd(Dm)[2][-1]
            for sgn in (1.0, -1.0):
                dd = sgn * d
                val = G @ dd
                lo = val.min()
                tied = tuple(i for i in range(n) if val[i] - lo < 1e-9 * (1 + abs(lo)))
                if not set(T) <= set(tied):
                    continue
                if np.linalg.matrix_rank(G[list(tied[1:])] - G[tied[0]], tol=1e-9) != 3:
                    continue
                if tied in seen:
                    continue
                seen.add(tied)
                out.append(([M[i] for i in tied], dd))
        return out

    # ------------------------------------------------------------- edges
    def _corrector(self, q, T4, t):
        """Newton back onto the curve {phi_t = phi_t0, t in T4} in the hyperplane
        orthogonal to t through q."""
        q0 = q.copy()
        try:
            for _ in range(60):
                phi, g = self.phig(q, T4)
                Fv = np.append(phi[1:] - phi[0], t @ (q - q0))
                if np.max(np.abs(Fv)) < 1e-13:
                    return q
                J = np.vstack([g[1:] - g[0], t])
                dq = np.linalg.solve(J, -Fv)
                n = self.hnorm(q, dq)
                if n > 0.1:                      # damped Newton (stay near the curve)
                    dq *= 0.1 / n
                q = q + dq
        except (OverflowError, FloatingPointError, np.linalg.LinAlgError):
            pass
        raise WalkError("corrector did not converge")

    def _tangent(self, q, T4, prev):
        _, g = self.phig(q, T4)
        t = np.linalg.svd(g[1:] - g[0])[2][-1]
        if t @ prev < 0:
            t = -t
        return t / self.hnorm(q, t)

    def follow(self, vert, T4, d):
        """Follow the edge from vertex `vert` with tied classes T4 (>= 4, rank 3) in
        direction d to the next vertex; returns the vertex dict (polished) or None
        if the edge runs off to the cusp (counted as unbounded)."""
        T4 = list(T4)
        base = T4[:4]
        q = np.array([float(c) for c in vert["qf"]])
        t = d / self.hnorm(q, d)
        travelled = 0.0

        def gap(qq, cand):
            ph, _ = self.phig(qq, base)
            pc, _ = self.phig(qq, cand)
            return float(np.min(pc - ph[0]))

        def point(s):
            return q if s == 0 else self._corrector(q + s * t, base, t)

        def first_crossing(sa, qa, ga, sb, qb, gb, cand, depth=0):
            """None if no candidate crosses on [sa, sb], else (s_safe, s_cross).
            Rigorous rule: |d gap / ds| <= 4 in the hyperbolic metric, so a segment
            of length L whose end gaps both exceed 2 L (with a margin for the
            variation of y along it) contains no crossing."""
            if gb < 0:
                return sa, sb
            L = self.hnorm(qa, qb - qa) * math.exp(abs(qb[2] - qa[2]) + abs(qb[3] - qa[3]))
            if min(ga, gb) > 2.2 * L:
                return None
            if depth > 40:
                # a class touching the minimum without crossing (gap ~ 0 on a
                # segment of length ~ 1e-10, positive at both ends): not a vertex
                # (the differences have rank 3 there), the edge goes on
                if min(ga, gb) > 0:
                    self.stats["tangencies"] += 1
                    return None
                raise WalkError("cannot separate a near-tangency: s=[%.3g, %.3g]" % (sa, sb))
            sm = 0.5 * (sa + sb)
            qm = point(sm)
            gm = gap(qm, cand)
            r = first_crossing(sa, qa, ga, sm, qm, gm, cand, depth + 1)
            if r is not None:
                return r
            return first_crossing(sm, qm, gm, sb, qb, gb, cand, depth + 1)

        for _ in range(self.max_steps):
            self.stats["steps"] += 1
            phi, _ = self.phig(q, base)
            cand = self.others(q, T4, math.exp(phi[0]))
            # vectors outside cand have gap >= log(far): a step of hyperbolic length
            # < log(far) / 4.4 cannot bring them down (same Lipschitz rule)
            h = min(self.step, 0.9 * math.log(self.far) / 4.4)
            while True:
                try:
                    q1 = point(h)
                    break
                except (WalkError, np.linalg.LinAlgError):
                    h *= 0.5
                    if h < 1e-9:
                        raise WalkError("corrector did not converge near %s" % q)
            if travelled == 0:
                # at the vertex its other minimal classes are tied (gap 0) but leave
                # at a positive rate along the ray (rays()); they are checked from
                # s0 = 2^-20 h on, all other classes from s = 0
                near = [v for v in cand if self.member(v, vert["vectors"])]
                far_ = [v for v in cand if not self.member(v, vert["vectors"])]
                r = first_crossing(0.0, q, gap(q, far_), h, q1, gap(q1, far_), far_) if far_ else None
                if near:
                    s0 = h * 2.0 ** -20
                    qa = point(s0)
                    if gap(qa, near) <= 0:
                        raise WalkError("a tied class does not leave along the ray")
                    r2 = first_crossing(s0, qa, gap(qa, near), h, q1, gap(q1, near), near)
                    if r2 is not None and (r is None or r2[1] < r[1]):
                        r = r2
            else:
                r = first_crossing(0.0, q, gap(q, cand), h, q1, gap(q1, cand), cand)
            if r is not None:
                lo, hi = r
                while hi - lo > 1e-10:
                    mid = 0.5 * (lo + hi)
                    if gap(point(mid), cand) < 0:
                        hi = mid
                    else:
                        lo = mid
                qq = point(hi)
                ph, _ = self.phig(qq, base)
                pc, _ = self.phig(qq, cand)
                for j in np.argsort(pc - ph[0])[:4]:
                    five = self._five(qq, base, cand[int(j)])
                    if five is None:
                        continue
                    v = self.polish(qq, five)
                    if v is not None and v["perf_rank"] == 5 and \
                            self.hnorm(qq, v["qf"] - qq) < 1e-6 and \
                            all(self.member(u, v["vectors"]) for u in T4):
                        return v
                raise WalkError("could not polish the neighbour vertex")
            travelled += h
            t = self._tangent(q1, base, t)
            q = q1
            if travelled > 50 or q[2] + q[3] > 50:
                self.stats["unbounded"] += 1
                return None
        raise WalkError("edge too long")

    def _five(self, q, base, w):
        """w plus 4 tied classes whose 5 gradients have rank-4 differences."""
        _, g = self.phig(q, base + [w])
        for four in itertools.combinations(range(len(base)), 4):
            idx = list(four) + [len(base)]
            if np.linalg.matrix_rank(g[idx[1:]] - g[idx[0]], tol=1e-7) == 4:
                return [base[i] for i in four] + [w]
        return None

    # ------------------------------------------------------------- the walk
    def seed(self):
        """A first perfect form: ascend G from (0, 0, 0, 0) through strata of growing
        rank (gradient ascent of the tied value projected onto the stratum)."""
        q = np.array([0.0, 0.0, 0.0, 0.0])
        m, M = self.P.minimum(self._p(q))
        T = list(M)
        for _ in range(5000):
            phi, g = self.phig(q, T)
            if len(T) >= 2:
                Dm = g[1:] - g[0]
                r = np.linalg.matrix_rank(Dm, tol=1e-9)
            else:
                Dm, r = np.zeros((0, 4)), 0
            if r >= 4:
                five = self._rank4_subset(T, g)
                v = self.polish(q, five)
                if v is None or v["perf_rank"] != 5:
                    raise WalkError("seed polish failed")
                return v
            # ascent direction of phi_T0 within the stratum
            if r:
                Nn = np.linalg.svd(Dm)[2][r:]
                d = Nn.T @ (Nn @ g[0])
            else:
                d = g[0].copy()
            if np.linalg.norm(d) < 1e-12:
                d = (np.linalg.svd(Dm)[2][r:]).T[:, 0] if r < 4 else d
            t = d / self.hnorm(q, d)
            cand = self.others(q, T, math.exp(phi[0]))
            h = 0.05
            while True:
                qn = q + h * t
                qn = self._stratum_correct(qn, T)
                ph, _ = self.phig(qn, T)
                pc, _ = self.phig(qn, cand)
                gap = pc - ph[0]
                if np.min(gap) >= 0 or h < 1e-12:
                    break
                h *= 0.5
            q = qn
            if np.min(gap) < 1e-6 * h or h < 1e-12:
                j = int(np.argmin(gap))
                T = T + [cand[j]]
                q = self._stratum_correct(q, T)
        raise WalkError("no seed vertex found")

    def _stratum_correct(self, q, T):
        if len(T) < 2:
            return q
        for _ in range(30):
            phi, g = self.phig(q, T)
            Fv = phi[1:] - phi[0]
            J = g[1:] - g[0]
            dq = np.linalg.lstsq(J, -Fv, rcond=None)[0]
            q = q + dq
            if np.max(np.abs(dq)) < 1e-14:
                break
        return q

    @staticmethod
    def _rank4_subset(T, g):
        for five in itertools.combinations(range(len(T)), 5):
            if np.linalg.matrix_rank(g[list(five[1:])] - g[five[0]], tol=1e-9) == 4:
                return [T[i] for i in five]
        raise WalkError("no rank-4 subset")

    def identify(self, vert):
        """Index of the class of vert (adding a new class if needed)."""
        from .analysis import equivalence
        n = len(vert["vectors"])
        for i, c in enumerate(self.classes):
            if len(c["vectors"]) != n or abs(c["gamma"] - vert["gamma"]) > mpmath.mpf(10) ** (-30) * c["gamma"]:
                continue
            if equivalence(self.F, c["vectors"], vert["vectors"]) is not None:
                return i, False
        vert["index"] = len(self.classes)
        self.classes.append(vert)
        return vert["index"], True

    def run(self, seed=None):
        t0 = time.time()
        v0 = seed or self.seed()
        self.identify(v0)
        todo = [0]
        while todo:
            i = todo.pop(0)
            c = self.classes[i]
            c["aut"] = len(automorphisms(self.F, c["vectors"]))
            c["neighbours"] = []
            for T, d in self.rays(c):
                self.stats["edges"] += 1
                nb = self.follow(c, T, d)
                if nb is None:
                    c["neighbours"].append(None)
                    continue
                j, new = self.identify(nb)
                c["neighbours"].append(j)
                if new:
                    todo.append(j)
                    if len(self.classes) > self.max_classes:
                        raise WalkError("more than %d classes" % self.max_classes)
            self.log("  class %d: |M| = %d, gamma = %s, %s, %d edges -> %s"
                     % (i, len(c["vectors"]), mpmath.nstr(c["gamma"], 15),
                        "extreme" if c["extreme"] else ("eutactic" if c["eutactic"] else "perfect"),
                        len(c["neighbours"]), c["neighbours"]))
        self.stats["time"] = time.time() - t0
        return self


def edge_balance(V):
    """N_ij / |Aut_i| = N_ji / |Aut_j| for all classes i, j (list of violations)."""
    from collections import Counter
    N = [Counter(j for j in c["neighbours"] if j is not None) for c in V.classes]
    bad = []
    for i, c in enumerate(V.classes):
        for j, d in enumerate(V.classes):
            if Fraction(N[i][j], c["aut"]) != Fraction(N[j][i], d["aut"]):
                bad.append((i, j, N[i][j], c["aut"], N[j][i], d["aut"]))
    return bad


# ---------------------------------------------------------------- records, CLI
def record(V, certified_gamma=None):
    """JSON-able result of a finished walk."""
    from .analysis import closed_form_gamma
    F = V.F

    def vec(v):
        return [[str(Fraction(c)) for c in x] for x in v]

    classes = []
    for c in V.classes:
        status = ("extreme" if c["extreme"] else
                  "weakly eutactic" if abs(c["eutaxy_margin"]) <= 1e-9 else "not eutactic")
        cf = closed_form_gamma(c["gamma"], dps=V.dps)
        classes.append({
            "index": c["index"], "gamma": mpmath.nstr(c["gamma"], V.dps - 10),
            "closed_form": cf[0] if cf and cf[0] else None,
            "n_min": len(c["vectors"]), "perf_rank": c["perf_rank"], "eutaxy": status,
            "eutaxy_margin": c["eutaxy_margin"], "aut": c["aut"],
            "neighbours": c["neighbours"], "point": [float(x) for x in c["qf"]],
            "vectors": [vec(v) for v in c["vectors"]]})
    gmax = max(float(c["gamma"]) for c in V.classes)
    bal = edge_balance(V)
    return {
        "D": V.D, "d_K": F.disc, "h_K": _class_number(F),
        "n_perfect": len(classes),
        "n_extreme": sum(c["eutaxy"] == "extreme" for c in classes),
        "n_weakly_eutactic": sum(c["eutaxy"] == "weakly eutactic" for c in classes),
        "gamma_max": mpmath.nstr(max(c["gamma"] for c in V.classes), 25),
        "gamma_max_matches_certified": (None if certified_gamma is None
                                        else abs(gmax - certified_gamma) <= 1e-12 * certified_gamma),
        "edge_balance_violations": len(bal), "stats": V.stats,
        "status": "COMPLETE" if not bal and V.stats["unbounded"] == 0 else "CHECK",
        "classes": classes}


def _class_number(F):
    if F.class_number_is_one():
        return 1
    from .ideals import class_group_charts
    return class_group_charts(F)[1]


def _child(D, step, certified_gamma):
    import json
    V = MultiplicativeVoronoi(D, step=step).run()
    print(json.dumps(record(V, certified_gamma)))


def main(argv=None):
    """hermite-humbert-voronoi FIELDS [--table TSV] [--out DIR] [-j N] [--timeout S]:
    all perfect forms (multiplicative normalisation) of each field, in parallel
    (one subprocess per field).  Writes OUT/D<D>.json and OUT/summary.tsv."""
    import argparse
    import csv
    import json
    import os
    import subprocess
    import sys
    from .sweep import parse_fields
    ap = argparse.ArgumentParser(prog="hermite-humbert-voronoi", description=main.__doc__)
    ap.add_argument("fields", nargs="*", help="D values or ranges (2-300); default: the certified fields of --table")
    ap.add_argument("--table", default="results/gamma_K2.tsv", help="certified gamma_K2 table (for the cross-check)")
    ap.add_argument("--out", default="results/perfect")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--step", type=float, default=0.05)
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
            cmd = [sys.executable, "-m", "hermite_humbert.voronoi", "--child", str(D), str(a.step),
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
                rec["seconds"] = round(time.time() - t0)
            else:
                msg = "timeout" if time.time() - t0 >= a.timeout else (err.strip().splitlines() or ["?"])[-1]
                rec = {"D": D, "status": "ERROR: " + msg[:200], "seconds": round(time.time() - t0)}
            with open(os.path.join(a.out, "D%d.json" % D), "w") as fh:
                json.dump(rec, fh, indent=1)
            print("  D = %3d  %s  perfect %s  extreme %s  %ds" % (D, rec["status"], rec.get("n_perfect", "-"),
                                                                rec.get("n_extreme", "-"), rec["seconds"]), flush=True)
        running = still
        if running:
            time.sleep(0.5)
    write_summary(a.out)


def write_summary(out):
    import glob
    import json
    import os
    rows = []
    for f in glob.glob(os.path.join(out, "D*.json")):
        r = json.load(open(f))
        rows.append(r)
    rows.sort(key=lambda r: r["D"])
    cols = ["D", "n_perfect", "n_extreme", "n_weakly_eutactic", "gamma_max", "gamma_max_matches_certified",
            "edge_balance_violations", "status", "seconds"]
    with open(os.path.join(out, "summary.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


if __name__ == "__main__":
    main()
