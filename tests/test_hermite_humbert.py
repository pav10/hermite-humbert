"""
Tests for the Hermite--Humbert constant code.

    pytest -m "not slow"      # unit tests (~2 min)
    pytest                    # + full pipeline on the published fields (~10 min)

Reference values are those of the papers in the project:
  [BCIO01] Baeza, Coulangeon, Icaza, O'Ryan, Exp. Math. 10 (2001) 543-551
  [PW05]   Pohst, Wagner, J. Theor. Nombres Bordeaux 17 (2005) 905-920
  [PW09]   Pohst, Wagner, J. Algebra 322 (2009) 936-947
"""
import itertools
import math
import random
import sys

import numpy as np
import pytest
import mpmath

from hermite_humbert.field import RealQuadraticField
from hermite_humbert.humbert import HumbertProblem, fincke_pohst
from hermite_humbert.vertex import VertexFinder
from hermite_humbert.certify import BranchAndBound, verify_certificate
from hermite_humbert.analysis import equivalence, galois_vectors, identify
from hermite_humbert.core import compute, gamma_of_form

mpmath.mp.dps = 50
S = mpmath.sqrt

PUBLISHED = {
    2: (4 / (2 * S(6) - 3), "BCIO01 Thm 3.4"),
    3: (mpmath.mpf(4), "BCIO01 Thm 3.6"),
    5: (4 / S(5), "BCIO01 Thm 3.2"),
    6: (mpmath.mpf(5), "PW09 Sec. 5"),
    13: (S((1476 + 144 * S(91)) / 175), "PW05 Sec. 4, PW09 (21)"),
    21: (mpmath.mpf(16) / 3, "PW09 Sec. 5"),
    7: (S((5584 + 640 * S(70)) / 605), "PW09, conjectured"),
    11: (mpmath.mpf(28) / 3, "PW09, conjectured"),
    17: (S((784 + 128 * S(34)) / 225), "PW05, local maximum (global value open there)"),
}


# ----------------------------------------------------------------- field
def test_fundamental_units():
    known = {2: (1, 1), 3: (2, 1), 5: (0, 1), 6: (5, 2), 7: (8, 3), 13: (1, 1), 17: (3, 2),
             21: (2, 1), 31: (1520, 273), 46: (24335, 3588)}      # PARI quadunit, same basis
    for D, u in known.items():
        assert RealQuadraticField(D).fundamental_unit() == u, D


def test_class_number_one():
    h1 = [2, 3, 5, 6, 7, 11, 13, 14, 17, 19, 21, 22, 23, 29, 31, 33, 37, 38, 41, 43, 46, 47]
    hbig = [10, 15, 26, 30, 34, 35, 39, 42, 51, 55, 58, 65, 66, 70, 74, 78, 79, 82, 85, 87, 91, 95]
    for D in h1:
        assert RealQuadraticField(D).class_number_is_one(), D
    for D in hbig:
        assert not RealQuadraticField(D).class_number_is_one(), D


def test_class_number_against_pari():
    cypari2 = pytest.importorskip("cypari2")
    pari = cypari2.Pari()
    for D in range(2, 200):
        try:
            F = RealQuadraticField(D)
        except ValueError:
            continue
        h = int(pari.bnfinit(pari("x^2-%d" % D), 1).bnf_get_no())
        assert (h == 1) == F.class_number_is_one(), D
        assert abs(math.log(F.eps0()) - float(pari.quadregulator(pari.quaddisc(D)))) < 1e-9 * math.log(F.eps0())


def test_primitive_and_sl2():
    rng = random.Random(3)
    for D in (5, 13, 19):
        F = RealQuadraticField(D)
        for _ in range(200):
            a = (rng.randint(-9, 9), rng.randint(-9, 9))
            b = (rng.randint(-9, 9), rng.randint(-9, 9))
            if a == (0, 0) and b == (0, 0):
                continue
            if F.is_primitive(a, b):
                g, d = F.complete_to_sl2(a, b)
                assert F.sub(F.mul(a, d), F.mul(b, g)) == (1, 0)
            else:   # a non-unit common divisor exists: then no SL2 completion
                try:
                    F.complete_to_sl2(a, b)
                    assert False, "completed a non-primitive vector"
                except ValueError:
                    pass


# ----------------------------------------------------------- enumeration
def test_fincke_pohst_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(20):
        A = rng.normal(size=(4, 4))
        G = A @ A.T + 0.05 * np.eye(4)
        B = float(rng.uniform(0.5, 4)) * np.min(np.diag(G))
        fp = set(fincke_pohst(G, B))
        R = 6
        bf = set()
        for x in itertools.product(range(-R, R + 1), repeat=4):
            if any(x) and np.array(x) @ G @ np.array(x) <= B:
                y = x if next(c for c in x if c) > 0 else tuple(-t for t in x)
                bf.add(y)
        assert bf == {x for x in fp if max(map(abs, x)) <= R}


def test_minimum_at_published_forms():
    t, tp = (1 + 5 ** .5) / 2, (1 - 5 ** .5) / 2
    g, vs, _ = gamma_of_form(5, [[1, -t / 2], [-t / 2, 1]], [[1, -tp / 2], [-tp / 2, 1]])
    assert abs(g - 4 / 5 ** .5) < 1e-12 and len(vs) == 5                   # BCIO01 Thm 3.2
    u, up = 2 + 3 ** .5, 2 - 3 ** .5
    g, vs, _ = gamma_of_form(3, [[1, u / 2], [u / 2, u]], [[1, up / 2], [up / 2, up]])
    assert abs(g - 4) < 1e-12 and len(vs) == 12                            # BCIO01 Thm 3.6
    r2, r6 = 2 ** .5, 6 ** .5
    g, vs, _ = gamma_of_form(2, [[1, (1 + r2) / 2], [(1 + r2) / 2, (r6 + r2) / 2]],
                             [[1, (1 - r2) / 2], [(1 - r2) / 2, (r6 - r2) / 2]])
    assert abs(g - 4 / (2 * r6 - 3)) < 1e-12 and len(vs) == 5              # BCIO01 Thm 3.4
    r13, r7, r91 = 13 ** .5, 7 ** .5, 91 ** .5
    b1, c1 = (-8 + r13 + 7 * r7 - 2 * r91) / 18, (7 - 2 * r13 - 14 * r7 + 4 * r91) / 9
    b2, c2 = (-8 - r13 - 7 * r7 - 2 * r91) / 18, (7 + 2 * r13 + 14 * r7 + 4 * r91) / 9
    g, vs, _ = gamma_of_form(13, [[1, b1], [b1, c1]], [[1, b2], [b2, c2]])
    assert abs(g - float(PUBLISHED[13][0])) < 1e-11 and len(vs) == 6        # PW05 p. 916
    r17, r34 = 17 ** .5, 34 ** .5
    g, vs, _ = gamma_of_form(17, [[1, .5], [.5, -4 + r17 - 4 * r2 + r34]], [[1, .5], [.5, -4 - r17 + 4 * r2 + r34]])
    assert abs(g - float(PUBLISHED[17][0])) < 1e-11 and len(vs) == 5        # PW05 p. 918
    r21 = 21 ** .5
    g, vs, _ = gamma_of_form(21, [[1, (r21 - 2) / 4], [(r21 - 2) / 4, (7 - r21) / 4]],
                             [[1, -(r21 + 2) / 4], [-(r21 + 2) / 4, (7 + r21) / 4]])
    assert abs(g - 16 / 3) < 1e-11 and len(vs) == 6                         # PW09 Sec. 5
    g, vs, _ = gamma_of_form(6, [[1, (2 - 2 * r6) / 5], [(2 - 2 * r6) / 5, (7 - 2 * r6) / 5]],
                             [[1, (2 + 2 * r6) / 5], [(2 + 2 * r6) / 5, (7 + 2 * r6) / 5]])
    assert abs(g - 5) < 1e-11 and len(vs) == 8                              # PW09 Sec. 5


def test_V_exact_and_invariance():
    rng = np.random.default_rng(7)
    for D in (13, 21, 19):
        P = HumbertProblem(D)
        P.build_V(P.d / 2.0)
        for _ in range(25):
            p = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3),
                          math.exp(rng.uniform(-2.5, 1)), math.exp(rng.uniform(-2.5, 1))])
            m_raw, _ = P.minimum(p)
            q = P.reduce_to_top(p)
            assert P.in_exact_region(q)
            mq, _ = P.minimum(q)
            gv, _ = P.G_V(q)
            assert abs(m_raw - mq) < 1e-9 * mq            # G is invariant under the reduction
            assert abs(gv[0] - mq) < 1e-9 * mq            # V is exact on the region
            assert abs(mq * q[2] * q[3] - 1) < 1e-9       # e1 is minimal at the highest point


def test_identify():
    mpmath.mp.dps = 60
    c, s = identify(1 + mpmath.sqrt(2))
    assert c == [1, -2, -1]
    c, s = identify(mpmath.mpf(28) / 3)
    assert c == [3, -28]
    mpmath.mp.dps = 50


def test_galois_equivalence_Q5():
    # BCIO01 p. 549: the form and its Galois conjugate are GL2(O_K)-equivalent
    P = HumbertProblem(5)
    t, tp = (1 + 5 ** .5) / 2, (1 - 5 ** .5) / 2
    _, _, M = gamma_of_form(5, [[1, -t / 2], [-t / 2, 1]], [[1, -tp / 2], [-tp / 2, 1]])
    assert equivalence(P.F, M, galois_vectors(P.F, M)) is not None


# ------------------------------------------------------------ pipeline
def _check(D, expect_new=False):
    r = compute(D, verbose=False, exact=True, verify=True, n_samples=2000, n_seeds=16)
    assert r["exact"] is not None and r["exact"]["proved"], D
    for ver in r["exact"]["verification"]:          # one re-check per chart
        assert sum(ver["failed"].values()) == 0 and abs(ver["volume_ratio"] - 1) < 1e-9
    if D in PUBLISHED:
        assert abs(r["gamma"] - PUBLISHED[D][0]) < mpmath.mpf(10) ** -40 * PUBLISHED[D][0], (D, r["gamma"])
    return r


@pytest.mark.slow
def test_pipeline_bcio():
    for D in (2, 3, 5):
        _check(D)


@pytest.mark.slow
def test_pipeline_pohst_wagner():
    for D in (6, 7, 11, 13, 17, 21):
        _check(D)


def test_certificate_detects_false_claim():
    """Soundness check of the verifier: a certificate produced for gamma* cannot
    be re-verified for a smaller threshold."""
    P = HumbertProblem(13)
    g = float(PUBLISHED[13][0])
    bb = BranchAndBound(P, g, delta=1e-9)
    bb.run(keep_certificate=True)
    ok = verify_certificate(bb, g * (1 + 1e-9) * (1 + 1e-11))
    assert sum(ok["failed"].values()) == 0
    bad = verify_certificate(bb, g * (1 - 1e-4))
    assert bad["failed"][0] > 0


# ------------------------------------------------------- class number > 1
def test_class_group():
    from hermite_humbert.ideals import class_group_charts
    known = {10: 2, 15: 2, 26: 2, 79: 3, 82: 4, 226: 8, 5: 1, 13: 1}
    for D, h in known.items():
        charts, hh = class_group_charts(RealQuadraticField(D))
        assert hh == h, (D, hh)
    cypari2 = pytest.importorskip("cypari2")
    pari = cypari2.Pari()
    for D in range(2, 200):
        try:
            F = RealQuadraticField(D)
        except ValueError:
            continue
        if F.eps0() > 1e12:
            continue
        h = int(pari.bnfinit(pari("x^2-%d" % D), 1).bnf_get_no())
        assert class_group_charts(F)[1] == h, D


def test_chart_covering():
    """Random forms on O_K^2 are mapped (Steinitz frame of a minimal vector) to a
    highest point of the chart of its content class; the value must agree with
    the independent O_K^2 minimum and with the chart's vector set V."""
    from hermite_humbert.ideals import class_group_charts
    from hermite_humbert import chart_point
    rng = np.random.default_rng(11)
    for D in (10, 79):
        F = RealQuadraticField(D)
        charts, h = class_group_charts(F)
        probs = [HumbertProblem(D, require_h1=False, chart=c) for c in charts]
        for P in probs:
            P.build_V(P.d / 2.0)
        P0 = HumbertProblem(D, require_h1=False)
        for _ in range(30):
            p = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3),
                          math.exp(rng.uniform(-2.5, 1)), math.exp(rng.uniform(-2.5, 1))])
            g0, _ = P0.minimum(p)
            k, q = chart_point(D, p, charts, probs)
            P = probs[k]
            assert P.in_exact_region(q)
            g1, _ = P.minimum(q)
            gv, _ = P.G_V(q)
            assert abs(g1 * q[2] * q[3] - 1) < 1e-9          # e1 minimal: highest point
            assert abs(g1 - g0) < 1e-8 * g0 and abs(gv[0] - g0) < 1e-8 * g0


def test_chart_pullback():
    """Points of the non-trivial chart regions, written back on O_K^2 through a
    Steinitz basis, have the same minimum (targeted O_K^2 enumeration)."""
    from hermite_humbert.ideals import class_group_charts, steinitz_frame
    from hermite_humbert.analysis import m_inv
    rng = np.random.default_rng(3)
    for D in (82, 15):
        F = RealQuadraticField(D)
        charts, h = class_group_charts(F)
        P0 = HumbertProblem(D, require_h1=False)
        for ch in charts[1:]:
            P = HumbertProblem(D, require_h1=False, chart=ch)
            P.build_V(P.d / 2)
            p1 = (ch.c.e, ch.c.f)
            _, w = steinitz_frame(F, p1)
            Pinv = m_inv(F, ((p1[0], w[0]), (p1[1], w[1])))
            for _ in range(15):
                q = P.rst_to_p(rng.uniform(-.5, .5), rng.uniform(0, .5),
                               rng.uniform(P.s_min, P.s_max), rng.uniform(P.t_lo, 0.3))
                gc, _ = P.minimum(q)
                gv, _ = P.G_V(q)
                sv = P0.short_vectors(P0.act(q, Pinv), gc * (1 + 1e-7), primitive_only=False)
                g0 = min(f for _, f in sv)
                assert abs(gc - g0) < 1e-7 * g0 and abs(gv[0] - g0) < 1e-7 * g0


@pytest.mark.slow
def test_pipeline_class_number_two():
    """Regression values computed by this code (not in the project papers)."""
    expect = {10: (80 + 16 * S(10)) / 15, 15: (112 + 16 * S(159)) / 25}
    for D, val in expect.items():
        r = _check(D)
        assert abs(r["gamma"] - val) < mpmath.mpf(10) ** -40 * val, D


def test_corner_rule_soundness():
    """Rule 4 (second-order corner bound): starting far below the maximum, the
    claimed upper bound must still be >= the true value, and the interval
    re-check must reject the certificate for a lowered threshold."""
    P = HumbertProblem(13)
    g = float(PUBLISHED[13][0])
    bb = BranchAndBound(P, 3.8, delta=1e-9)      # 6% below the maximum
    up = bb.run(keep_certificate=True)
    assert not bb.overflow and bb.gamma_low > 3.8
    assert up >= g, (up, g)
    ok = verify_certificate(bb, bb.gamma_low * (1 + bb.delta) * (1 + bb.safety))
    assert sum(ok["failed"].values()) == 0 and abs(ok["volume_ratio"] - 1) < 1e-9
    bb2 = BranchAndBound(P, g, delta=1e-9)
    bb2.run(keep_certificate=True)
    bad = verify_certificate(bb2, g * (1 - 1e-4))
    assert bad["failed"][4] > 0




def test_parallel_branch_and_bound():
    """run_parallel must produce a certificate that verifies (tiling + all rules)
    and the same value as the serial run."""
    for D in (13, 10):
        r = compute(D, verbose=False, exact=True, verify=True, n_samples=2000, n_seeds=16, jobs=2)
        assert r["exact"]["proved"], D
        for ver in r["exact"]["verification"]:
            assert sum(ver["failed"].values()) == 0 and abs(ver["volume_ratio"] - 1) < 1e-9
    assert abs(r["gamma"] - (80 + 16 * S(10)) / 15) < mpmath.mpf(10) ** -40


def test_sweep_cli(tmp_path):
    """hermite-humbert-sweep: isolated subprocess per field, table + JSON written,
    a timed-out field is recorded (not lost), and re-running resumes."""
    import csv
    from hermite_humbert.sweep import main as sweep_main
    out = tmp_path / "t.tsv"
    sweep_main([str(out), "5,13,46", "-j", "2", "--timeout", "15"])
    rows = {int(r["D"]): r for r in csv.DictReader(open(out), delimiter="\t")}
    assert rows[5]["status"].startswith("certified") and rows[13]["status"].startswith("certified")
    assert rows[46]["status"] == "timeout(15s)"
    assert (tmp_path / "fields" / "D13.json").exists()
    sweep_main([str(out), "5,13,46", "--timeout", "15"])      # nothing left to do
    assert len(list(csv.DictReader(open(out), delimiter="\t"))) == 3


def test_parallel_block_death_is_retried(monkeypatch, tmp_path):
    """A block whose process dies is retried (and never hangs the run); if it
    keeps dying it is left unresolved, so the proof fails instead of passing."""
    import os
    from hermite_humbert import certify
    orig = certify._par_task

    def die_once(args):
        # first attempt of every block dies (marker file per block), the retry succeeds
        marker = tmp_path / ("block_%s" % "_".join("%.12g" % x for x in args[0][0]))
        if not marker.exists():
            marker.touch()
            os._exit(1)
        return orig(args)
    monkeypatch.setattr(certify, "_par_task", die_once)
    r = compute(13, verbose=False, exact=True, verify=True, n_samples=2000, n_seeds=16, jobs=2)
    assert r["exact"]["proved"]
    assert all(st.get("lost_blocks", 0) == 0 for st in r["exact"]["stats"])
    # blocks that always die: not proved, and no hang
    monkeypatch.setattr(certify, "_par_task", lambda args: os._exit(1))
    r = compute(13, verbose=False, exact=True, verify=False, n_samples=2000, n_seeds=16, jobs=2)
    assert not r["exact"]["proved"]


def test_verify_parallel_matches_serial():
    """The chunked, forked re-check gives exactly the serial counts and volume,
    and still rejects a certificate checked against a lowered threshold."""
    from hermite_humbert.certify import verify_parallel
    P = HumbertProblem(13)
    g = float(PUBLISHED[13][0])
    bb = BranchAndBound(P, g, delta=1e-9)
    bb.run(keep_certificate=True)
    for thr in (g * (1 + 1e-9) * (1 + 1e-11), g * (1 - 1e-4)):
        a = verify_certificate(bb, thr)
        b = verify_parallel(bb, thr, jobs=3)
        assert a["verified"] == b["verified"] and a["failed"] == b["failed"], (a, b)
        assert abs(a["volume_ratio"] - b["volume_ratio"]) < 1e-12
    assert b["failed"][4] > 0
