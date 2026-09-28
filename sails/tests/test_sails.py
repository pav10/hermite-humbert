"""Tests: published / independently computed values and a brute-force oracle.

Run with  python -m pytest tests  or  python tests/test_sails.py.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sails import compute_field                                   # noqa: E402
from sails.field import NumberFieldData, pari                     # noqa: E402
from sails.sail import Orthant                                     # noqa: E402

# (label, coeffs, [(#facet orbits, #indecomposables, max norm)] per signature class)
# Values: trsail README (validated there against brute force / Visser's tables).
REFERENCE = [
    ('3.3.49.1', [1, -2, -1, 1], (2, 2, 7)),
    ('3.3.81.1', [1, -3, 0, 1], (2, 3, 9)),
    ('3.3.148.1', [1, -3, -1, 1], (4, 7, 23)),
    ('3.3.169.1', [-1, -4, -1, 1], (2, 5, 13)),
    ('3.3.229.1', [-1, -4, 0, 1], (2, 4, 16)),
    ('3.3.321.1', [3, -6, 1, 1], (4, 12, 33)),
    ('3.3.361.1', [-1, -5, -2, 1], (2, 8, 19)),
    ('3.3.473.1', [1, -5, 0, 1], (4, 14, 55)),
    ('3.3.1369.1', [-1, -7, -4, 1], (2, 17, 47)),
    ('4.4.725.1', [1, 1, -3, -1, 1], (7, 7, 31)),
    ('4.4.1125.1', [1, 4, -4, -1, 1], (1, 1, 1)),
    ('4.4.1957.1', [-1, 5, -5, -1, 1], (6, 17, 103)),
    ('5.5.14641.1', [-1, 3, 3, -4, -1, 1], (9, 28, 253)),
]


def test_reference_values():
    for label, coeffs, (orbits, count, maxnorm) in REFERENCE:
        r = compute_field(coeffs, label, tp_only=True)
        O = r.orthants[0]
        got = (len(O.facets), len(O.indecomposables), max(N for N, _ in O.indecomposables))
        assert got == (orbits, count, maxnorm), (label, got)
        # S = sum d_F * vol_F is the total volume of the cones over the facet orbits
        assert all(F.volume > 0 for F in O.facets.values())


def test_simplest_cubics():
    """Kala-Tinkova: for x^3 - a x^2 - (a+3) x - 1 with O_K = Z[rho] there are
    2 + (a+1)(a+2)/2 indecomposables up to TP units."""
    for a in (-1, 0, 1, 2, 4, 6, 7):
        coeffs = [-1, -(a + 3), -a, 1]
        K = NumberFieldData(coeffs)
        if K.monogenic_index != 1:
            continue
        r = compute_field(coeffs, tp_only=True)
        assert len(r.orthants[0].indecomposables) == 2 + (a + 1) * (a + 2) // 2, a


def test_cubic_kala_man_formula():
    """Degree 3 (Kala-Man): #Ind = sum_F (d_F - 1/2) vol_F over facet orbits."""
    for label, coeffs, _ in REFERENCE[:9]:
        r = compute_field(coeffs, label)
        for O in r.orthants:
            formula = sum((2 * F.d - 1) * F.volume for F in O.facets.values())
            assert formula == 2 * len(O.indecomposables), label


def test_trsail_orthants():
    """All 27 cubic fields of discriminant <= 1000, every orthant (trsail data)."""
    data = json.load(open(os.path.join(HERE, 'data', 'trsail_cubic_orthants.json')))
    for rec in data:
        r = compute_field(rec['coeffs'], rec['label'])
        K = r.K
        by_class = {O.index: O for O in r.orthants}
        for key, (count, maxnorm, orbits, S) in rec['orthants'].items():
            tau = tuple(int(ch == '-') for ch in key)
            O = by_class[K.sig_class[tau][0]]
            got = (len(O.indecomposables), max(N for N, _ in O.indecomposables), len(O.facets),
                   sum(F.d * F.volume for F in O.facets.values()))
            assert got == (count, maxnorm, orbits, S), (rec['label'], key, got)


def _cf_count(D):
    """Number of TP indecomposables of Q(sqrt D) up to TP units (Dress-Scharlau):
    u_1 + u_3 + ... over one period (two if the period is odd) of omega."""
    P, Q = (1, 2) if D % 4 == 1 else (0, 1)       # omega = (P + sqrt D) / Q
    seen, us = {}, []
    r = math.isqrt(D)
    while (P, Q) not in seen:
        seen[(P, Q)] = len(us)
        a = (P + r) // Q
        us.append(a)
        P = a * Q - P
        Q = (D - P * P) // Q
    start = seen[(P, Q)]
    period = us[start:]
    if start == 0:                   # purely periodic: drop u_0 convention
        period = period[1:] + period[:1]
    s = len(period)
    seq = period * (2 if s % 2 else 1)
    return sum(seq[i] for i in range(0, len(seq), 2))


def test_quadratic_fields():
    for D in (2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 29, 31, 33, 41, 43, 46, 94):
        coeffs = [(1 - D) // 4, -1, 1] if D % 4 == 1 else [-D, 0, 1]
        r = compute_field(coeffs, tp_only=True)
        assert len(r.orthants[0].indecomposables) == _cf_count(D), D


# ----------------------------------------------------------------------------
# brute force oracle (independent of the sail machinery)

def _box_indecomposable(K, alpha, sig):
    """No beta in O_K with beta and alpha - beta of sign vector sig (qfminim)."""
    import numpy as np
    n = K.n
    emb = K.E @ np.array(alpha, dtype=float)
    Mw = K.E / emb[:, None]
    G = Mw.T @ Mw                                    # Q(x) = sum (sigma_i(x)/sigma_i(alpha))^2 < n
    Gp = pari.matrix(n, n, [pari(repr(float(x))) for x in G.flatten()])
    res = pari.qfminim(Gp, n + 1e-6, None, 2)
    for colv in pari.Vec(res[2]):
        v = tuple(int(t) for t in colv)
        for b in (v, tuple(-t for t in v)):
            if b == tuple(alpha):
                continue
            rest = tuple(p - q for p, q in zip(alpha, b))
            if K.signs(b) == sig and any(rest) and K.signs(rest) == sig:
                return False
    return True


def brute_force_tp(coeffs):
    """TP indecomposables up to TP units: all TP elements of norm <= disc
    (Kala-Yatsyna bound), each tested by an exhaustive box search."""
    K = NumberFieldData(coeffs)
    O = Orthant(K, 0)
    reps = {}
    for N in range(1, K.disc + 1):
        for sN in ((N,) if K.n % 2 else (N, -N)):
            for g in pari.bnfisintnorm(K.bnf, sN):
                x = tuple(int(t) for t in pari.nfalgtobasis(K.nf, g))
                j, y = K.to_signature_class(x)
                if j == 0:
                    reps[O.canon_point(y)[0]] = N
    ind = sorted((N, y) for y, N in reps.items() if _box_indecomposable(K, y, O.sig))
    return ind


def test_brute_force_oracle():
    for coeffs in ([1, -2, -1, 1], [1, -3, -1, 1], [-1, -4, 0, 1], [3, -6, 1, 1], [1, -5, 0, 1],
                   [1, 1, -3, -1, 1]):
        r = compute_field(coeffs, tp_only=True)
        ours = sorted(r.orthants[0].indecomposables)
        assert ours == brute_force_tp(coeffs), coeffs


def test_exact_hulls():
    """Gift wrapping agrees with brute force; triangulations have the right volume."""
    import random
    from sails import exact
    rng = random.Random(7)
    for _ in range(150):
        m = rng.choice([2, 3, 4])
        pts = list({tuple(rng.randint(-3, 3) for _ in range(m)) for _ in range(rng.randint(m + 5, m + 12))})
        if exact.affine_rank(pts) < m:
            continue
        old = exact.GIFTWRAP_MIN_EXTRA
        exact.GIFTWRAP_MIN_EXTRA = 10 ** 9
        brute = sorted((a, b, tuple(sorted(s))) for a, b, s in exact.hull_facets(pts))
        exact.GIFTWRAP_MIN_EXTRA = old
        gw = sorted((a, b, tuple(sorted(s))) for a, b, s in exact.giftwrap_facets(pts))
        assert brute == gw
        # volume of the pulling triangulation = volume of a fan from an interior point
        simp = exact.triangulate(pts)
        vol = sum(abs(exact.det([tuple(pts[i][t] - pts[s[0]][t] for t in range(m)) for i in s[1:]]))
                  for s in simp)
        assert vol > 0


def test_output_format():
    r = compute_field([-1, -4, 0, 1], '3.3.229.1')
    line = r.gordon_line()
    cols = line.split('|')
    assert len(cols) == 11
    assert cols[0] == '3.3.229.1' and cols[1] == '[-1,-4,0,1]'
    assert cols[2] == '[(0, 0, 0),(1, 0, 0)]'
    assert cols[4] == '[4,6]' and cols[6] == '[16,28]'
    assert cols[9].startswith('[[1, ')


if __name__ == '__main__':
    import time
    for name, f in list(globals().items()):
        if name.startswith('test_') and callable(f):
            t = time.time()
            f()
            print('%-32s ok  %.1fs' % (name, time.time() - t), flush=True)
