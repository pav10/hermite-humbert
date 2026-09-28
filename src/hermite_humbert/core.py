#!/usr/bin/env python3
"""
Hermite--Humbert constant gamma_{K,2} of a real quadratic field K = Q(sqrt D)
(any class number; binary forms, Icaza's normalisation):

    gamma_{K,2} = sup_S  min_{0 != v in O_K^2} S1[v] S2[v'] / sqrt(det S1 det S2).

Method (combining [BCIO01], [PW05], [PW09] with a certified global search):
 1. Forms <-> points z of H^2; gamma(S) = G(z) = min_v f_v(z).  By [BCIO, Prop 1.1]
    gamma_{K,2} = 1 / (height of the lowest point of the Hilbert modular fundamental
    domain) (Cohn's "floor"), and every orbit with G >= gamma_low has a
    representative in an explicit compact box R (HumbertProblem).  For h_K > 1
    there is one such box per ideal class (Steinitz charts, ideals.py).
 2. Extreme forms = perfect + eutactic [Coulangeon 2001] = vertices of G at which
    0 is a strictly positive combination of the gradients.  They are located by
    trust-region LP ascent from quasi-random seeds and polished by Newton at
    60+ digits; perfection rank and eutactic coefficients are computed.
 3. Global maximality: branch-and-bound over R with the exact bound
    sup_B f_v <= prod_j max(...) (certify.py): gamma_{K,2} <= gamma*(1+delta).
 4. Exact mode: a perfect eutactic vertex is a strict local maximum on an
    explicit ball (analysis.ball_certificate); the B&B then proves G <= gamma*
    on all of R, i.e. gamma_{K,2} = gamma*.
 5. gamma* is identified as an algebraic number by PSLQ; forms are reported in
    the Pohst--Wagner normal form ((1, b1; b1, c1), (1, b2; b2, c2)), m(S) = 1.

What is rigorous and what is not: see README.md.
"""
import argparse
import math
import sys
import time

import numpy as np
import mpmath
from fractions import Fraction

from .field import RealQuadraticField

from .humbert import HumbertProblem
from .vertex import VertexFinder
from .certify import BranchAndBound, verify_certificate, _below
from .analysis import classify, classify_global, normal_form, closed_form_gamma, identify, ball_certificate


def local_vertex_factory(P, VF, dps):
    """For the exact-mode B&B: vertex near a point (no reduction to the region)."""
    def local_vertex(c):
        # c = box centre in (x1, x2, s, t)
        p = np.array([c[0], c[1], math.exp(c[3] - c[2]), math.exp(c[3] + c[2])])
        phi, _ = VF.phi_grad(p)
        m = float(np.min(phi))
        A = np.where(phi <= m + 1e-3)[0]
        pn, idx = VF.newton_float(p, A)
        if idx is None:
            return None
        vert = VF.polish(pn, idx, dps=dps)
        if vert is None:
            return None
        g = float(vert["gamma"])
        qc = np.array([float(vert["q"][i]) for i in range(4)])
        if not vert["extreme"]:
            return (g, qc, None, None)
        rho, kappa, sc = ball_certificate(vert)
        return (g, qc, rho, sc)
    return local_vertex


def _problems(D):
    """One HumbertProblem per chart: [O_K^2] if h_K = 1, else one per ideal class
    (a class and its inverse share a chart via Galois)."""
    from .ideals import class_group_charts
    F = RealQuadraticField(D)
    F.check_float_range()
    if F.class_number_is_one():
        return [HumbertProblem(D)], 1
    charts, h = class_group_charts(F)
    return [HumbertProblem(D, require_h1=False, chart=c) for c in charts], h


def compute(D, n_samples=3000, n_seeds=24, dps=100, delta=1e-9, exact=True, verify=False,
            verbose=True, max_boxes=5e7):
    """Compute gamma_{K,2} for K = Q(sqrt D) (any class number).  Returns a dict
    with the value, closed form, extreme forms found (per chart), and the
    certificate status.  The certificate covers every chart."""
    t0 = time.time()
    problems, h = _problems(D)
    F = problems[0].F
    say = print if verbose else (lambda *a, **k: None)
    say("K = Q(sqrt(%d)), d_K = %d, eps0 = %s (norm %+d), h_K = %d"
        % (D, F.disc, F.fmt(F.fundamental_unit()), F.unit_norm(), h))
    VFs = [VertexFinder(P) for P in problems]
    for P in problems:
        say("  chart %s: |V| = %d" % ("O_K^2" if P.chart is None else P.chart.label(), len(P.V)))
    verts = []
    for k, VF in enumerate(VFs):
        # fewer seeds when V is large: a missed maximum is recovered by the B&B
        seeds = max(6, min(n_seeds, int(n_seeds * 20000 / max(1, len(problems[k].V)))))
        for v in VF.multistart(n_samples=n_samples, n_seeds=seeds, dps=dps):
            v["chart"] = k
            verts.append(v)
    if not verts:
        raise RuntimeError("no extreme form found")
    notes = []

    def recover(k, point):
        """Vertex search from a point where a B&B saw G > gamma*."""
        v = VFs[k].find_vertex(point, dps=dps)
        if v is not None and v["gamma"] > gstar:
            v["chart"] = k
            verts.append(v)
            notes.append("multistart missed the maximum; recovered from the B&B (chart %d)" % k)
            return True
        return False

    exact_res, cert = None, None
    for _ in range(5):
        verts.sort(key=lambda v: -v["gamma"])
        gstar = verts[0]["gamma"]
        restart = False
        # ---- exact mode first: proves G <= gamma* on every chart
        if exact:
            exact_res = {"proved": True, "balls": 0, "stats": [], "counterexample": None}
            for k, P in enumerate(problems):
                bbx = BranchAndBound(P, float(gstar), delta=0.0)
                bbx.run(max_boxes=max_boxes, keep_certificate=verify,
                        exact=(float(gstar), local_vertex_factory(P, VFs[k], dps)))
                if bbx.overflow or bbx.gamma_low > float(gstar) * (1 + 1e-12) or bbx.counterexample is not None:
                    pt = bbx.best_point
                    if bbx.counterexample is not None:
                        qc = bbx.counterexample[1]
                        pt = np.array([qc[0], qc[1], math.exp(qc[2]), math.exp(qc[3])])
                    if pt is not None and recover(k, pt):
                        restart = True
                        break
                    exact_res["counterexample"] = bbx.counterexample or ("G", bbx.gamma_low)
                ok = (bbx.stats["unresolved"] == 0 and bbx.counterexample is None and not bbx.overflow
                      and bbx.gamma_low <= float(gstar) * (1 + 1e-12))
                st = dict(bbx.stats)
                if verify and ok:
                    vr = verify_certificate(bbx, _below(gstar))
                    st["verification"] = vr
                    ok = sum(vr["failed"].values()) == 0 and abs(vr["volume_ratio"] - 1) < 1e-9
                    exact_res.setdefault("verification", []).append(vr)
                exact_res["stats"].append(st)
                exact_res["balls"] += len(bbx.balls)
                exact_res["proved"] &= ok
            if restart:
                continue
            say("exact mode: %s  [%d certified balls around maximisers, %d boxes, %.1fs]%s"
                % ("G <= gamma* on every chart => gamma_{K,2} = gamma*" if exact_res["proved"] else "NOT completed",
                   exact_res["balls"], sum(st["boxes"] for st in exact_res["stats"]),
                   sum(st["time"] for st in exact_res["stats"]),
                   ("\n   outward-rounded re-check: %s" % exact_res["verification"]) if "verification" in exact_res else ""))
            if exact_res["proved"]:
                cert = {"delta": 0.0, "upper": float(gstar), "stats": exact_res["stats"]}
                break
        # ---- tolerance mode (fallback, or if exact mode was not requested)
        bbs = []
        for k, P in enumerate(problems):
            bb = BranchAndBound(P, float(gstar), delta=delta)
            bb.run(max_boxes=max_boxes)
            bbs.append(bb)
            if (bb.overflow or bb.gamma_low > float(gstar) * (1 + 1e-12)) and bb.best_point is not None \
                    and recover(k, bb.best_point):
                restart = True
                break
        if restart:
            continue
        cert = {"delta": delta, "upper": max(bb.gamma_upper for bb in bbs),
                "stats": [dict(bb.stats) for bb in bbs]}
        say("B&B (delta = %.0e): gamma_{K,2} <= %.15f   [%d boxes in %d chart(s), %d unresolved]"
            % (delta, cert["upper"], sum(bb.stats["boxes"] for bb in bbs), len(bbs),
               sum(bb.stats["unresolved"] for bb in bbs)))
        break

    # ---- classes of extreme forms, across charts (up to GL_2(O_K) and scaling)
    out_classes = []
    for c in classify_global(problems, verts):
        rep = c["rep"]
        P = problems[rep["chart"]]
        nf = normal_form(P, rep)
        cf = closed_form_gamma(rep["gamma"], dps=dps)
        out_classes.append({"gamma": rep["gamma"], "closed_form": cf, "normal_form": nf,
                            "n_min": len(rep["vectors"]), "perf_rank": rep["perf_rank"],
                            "eutaxy_margin": rep["eutaxy_margin"], "rho": rep["rho"],
                            "self_conjugate": c["self_conjugate"], "conjugate_of": c["conjugate_of"],
                            "copies_found": len(c["members"]), "chart": rep["chart"],
                            "charts_seen": sorted(set(m["chart"] for m in c["members"])),
                            "chart_label": "O_K^2" if P.chart is None else P.chart.label()})
    out_classes.sort(key=lambda c: -c["gamma"])
    gstar = out_classes[0]["gamma"]
    res = {"D": D, "d": F.disc, "h": h, "dps": dps, "gamma": gstar, "classes": out_classes,
           "certificate": cert, "exact": exact_res, "notes": notes, "time": time.time() - t0}
    if verbose:
        report(problems[0], res)
    return res


def gamma_of_form(D, S1, S2):
    """gamma(S) = m(S)/sqrt(det S1 det S2) and the minimal vectors (up to units)
    of a given binary Humbert form S = (S1, S2) over Q(sqrt D) (2x2 real
    positive definite matrices; S1 goes with the embedding w -> w1 > 0).
    Uses the exact enumeration HumbertProblem.minimum (independent of the
    fundamental-domain machinery; works for any class number)."""
    P = HumbertProblem(D, require_h1=False)
    S1, S2 = np.asarray(S1, dtype=float), np.asarray(S2, dtype=float)
    p = P.point_of_forms(S1, S2)
    m, M = P.minimum(p)
    return m, [(P.F.fmt(a), P.F.fmt(b)) for a, b in M], M


def chart_point(D, p, charts=None, problems=None):
    """Map the form S(p) on O_K^2 (any class number) to a highest point of its
    chart: take a minimal vector v with content c', its Steinitz frame
    O_K^2 = c'^{-1} v + c' w, the chart representative c ~ c' (Galois conjugate
    form if the class of c' was dropped), mu with c' = mu c, and change basis by
    (mu^{-1} v | mu w).  Returns (chart index, point in the chart region)."""
    from .ideals import class_group_charts, steinitz_frame, locate_chart, Ideal
    F = RealQuadraticField(D)
    if charts is None:
        charts, _ = class_group_charts(F)
    if problems is None:
        problems = [HumbertProblem(D, require_h1=False, chart=c) for c in charts]
    P0 = HumbertProblem(D, require_h1=False)
    m, M = P0.minimum(p)
    v = M[0]
    cp, _ = steinitz_frame(F, v)
    ch, conj = locate_chart(F, charts, cp)
    if conj:
        p = P0.galois(p)
        v = (F.conj(v[0]), F.conj(v[1]))
        cp = cp.conj()
    cp, w = steinitz_frame(F, v)
    c = ch.c
    g = cp.mul(c.conj()).generator()
    mu = F.mul(g, (Fraction(1) / c.norm(), Fraction(0)))
    assert abs(F.norm(mu)) == 1
    mui = F.inv(mu)
    U = ((F.mul(mui, v[0]), F.mul(mu, w[0])), (F.mul(mui, v[1]), F.mul(mu, w[1])))
    k = charts.index(ch)
    Pc = problems[k]
    q = Pc.to_region(Pc.act(p, U))
    return k, q


def report(P, res):
    F = P.F
    print()
    print("=" * 78)
    g = res["gamma"]
    cf = res["classes"][0]["closed_form"][1]
    print("gamma_{K,2}(Q(sqrt %d)) = %s" % (res["D"], mpmath.nstr(g, 30)))
    if cf:
        print("                    = %s   (PSLQ at %d digits, relation verified to >= %d digits)" % (cf, res["dps"], res["dps"] - 15))
    for n in res["notes"]:
        print("note: " + n)
    if res["exact"] and res["exact"]["proved"]:
        v = " + outward-rounded re-check" if "verification" in res["exact"] else ""
        print("status: certified (branch-and-bound + local certificate%s), see README" % v)
    else:
        print("status: gamma* <= gamma_{K,2} <= %.15f" % res["certificate"]["upper"])
    print("-" * 78)
    for k, c in enumerate(res["classes"]):
        nf = c["normal_form"]
        tag = "critical" if k == 0 else "extreme"
        conj = "Galois self-conjugate" if c["self_conjugate"] else (
            "Galois conjugate of class %d" % (c["conjugate_of"] + 1) if c.get("conjugate_of") is not None else "")
        if res.get("h", 1) > 1:
            conj += "  [chart %s; highest points found in charts %s]" % (c["chart_label"], c["charts_seen"])
        print("class %d (%s): gamma = %s  %s" % (k + 1, tag, mpmath.nstr(c["gamma"], 20),
                                                ("= " + c["closed_form"][1]) if c["closed_form"][1] else ""))
        print("   #M(S) = %d, perfection rank %d, eutactic margin %.4g, unimodular pair: %s  %s"
              % (c["n_min"], c["perf_rank"], c["eutaxy_margin"], nf["unimodular_pair"], conj))
        S1, S2 = nf["S1"], nf["S2"]
        print("   S1 = [[1, %s], [., %s]]" % (mpmath.nstr(S1[0, 1], 18), mpmath.nstr(S1[1, 1], 18)))
        print("   S2 = [[1, %s], [., %s]]" % (mpmath.nstr(S2[0, 1], 18), mpmath.nstr(S2[1, 1], 18)))
        vs = ", ".join("(%s, %s)" % (F.fmt(a), F.fmt(b)) for a, b in nf["vectors"])
        if nf.get("e1_minimal") is False:
            print("   (form normalised by a minimal vector; e1 is not minimal at this vertex)")
        print("   minimal vectors (this basis): %s" % vs)
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("D", type=int, nargs="+", help="squarefree D > 1 (K = Q(sqrt D))")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--dps", type=int, default=100)
    ap.add_argument("--delta", type=float, default=1e-9)
    ap.add_argument("--no-exact", action="store_true", help="skip the exact-mode certificate")
    ap.add_argument("--verify", action="store_true",
                    help="record the exact-mode certificate and re-check it with outward rounding")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    rows = []
    for D in a.D:
        r = compute(D, n_samples=a.samples, n_seeds=a.seeds, dps=a.dps, delta=a.delta,
                    exact=not a.no_exact, verify=a.verify, verbose=not a.quiet)
        rows.append(r)
    if len(rows) > 1:
        print("\n%5s %6s  %-22s %-34s %s" % ("D", "d_K", "gamma_{K,2}", "closed form", "status"))
        for r in rows:
            st = "certified" if r["exact"] and r["exact"]["proved"] else "<= %.12f" % r["certificate"]["upper"]
            print("%5d %6d  %-22s %-34s %s" % (r["D"], r["d"], mpmath.nstr(r["gamma"], 18),
                                             r["classes"][0]["closed_form"][1] or "?", st))


if __name__ == "__main__":
    main()
