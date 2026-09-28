"""
Post-processing of vertices: equivalence classes, normal forms, exact values,
and the local certificate that a perfect eutactic vertex is a strict local
maximum of G on an explicit ball.
"""
import math
import itertools
from fractions import Fraction

import numpy as np
import mpmath
from scipy.spatial import ConvexHull


# ------------------------------------------------------------ 2x2 over K
def m_mul(F, X, Y):
    return tuple(tuple(F.add(F.mul(X[i][0], Y[0][j]), F.mul(X[i][1], Y[1][j])) for j in range(2)) for i in range(2))


def m_det(F, X):
    return F.sub(F.mul(X[0][0], X[1][1]), F.mul(X[0][1], X[1][0]))


def m_inv(F, X):
    di = F.inv(m_det(F, X))
    return ((F.mul(X[1][1], di), F.mul(F.neg(X[0][1]), di)),
            (F.mul(F.neg(X[1][0]), di), F.mul(X[0][0], di)))


def m_vec(F, X, v):
    return (F.add(F.mul(X[0][0], v[0]), F.mul(X[0][1], v[1])),
            F.add(F.mul(X[1][0], v[0]), F.mul(X[1][1], v[1])))


def cols(u, w):
    return ((u[0], w[0]), (u[1], w[1]))


def associated_vec(F, v, w):
    """w = eps * v for a unit eps?"""
    for i in range(2):
        if (Fraction(v[i][0]), Fraction(v[i][1])) != (0, 0):
            e = F.div(w[i], v[i])
            if not F.is_unit(e):
                return False
            return all(F.mul(e, v[j]) == tuple(Fraction(c) for c in w[j]) for j in range(2))
    return False


def equivalence(F, M1, M2, in_aut=None):
    """Pohst--Wagner (JTNB 2005, Alg. 11), for perfect forms S, T with minimal
    classes M1 = M(S), M2 = M(T): return U in GL_2(O_K) with U M(T) = M(S)
    (then S ~ T up to scaling), else None."""
    if len(M1) != len(M2) or len(M1) < 3:
        return None
    M1 = [tuple(tuple(Fraction(c) for c in x) for x in v) for v in M1]
    M2 = [tuple(tuple(Fraction(c) for c in x) for x in v) for v in M2]
    a1 = M1[0]
    a2 = next(v for v in M1[1:] if m_det(F, cols(a1, v)) != (0, 0))
    a3 = next(v for v in M1 if v is not a1 and v is not a2)
    A = cols(a1, a2)
    lam = m_vec(F, m_inv(F, A), a3)
    for b1, b2 in itertools.permutations(M2, 2):
        B = cols(b1, b2)
        if m_det(F, B) == (0, 0):
            continue
        Binv = m_inv(F, B)
        for b3 in M2:
            if b3 is b1 or b3 is b2:
                continue
            mu = m_vec(F, Binv, b3)
            if mu[0] == (0, 0) or mu[1] == (0, 0):
                continue
            e1, e2 = F.div(lam[0], mu[0]), F.div(lam[1], mu[1])
            if not (F.is_unit(e1) and F.is_unit(e2)):
                continue
            Dg = ((e1, (0, 0)), ((0, 0), e2))
            U = m_mul(F, m_mul(F, A, Dg), Binv)
            if in_aut is not None:
                if not in_aut(U):
                    continue
            else:
                if not all(F.is_integral(U[i][j]) for i in range(2) for j in range(2)):
                    continue
                if not F.is_unit(m_det(F, U)):
                    continue
            images = [m_vec(F, U, w) for w in M2]
            if all(any(associated_vec(F, s, im) for s in M1) for im in images):
                return U
    return None


def galois_vectors(F, M):
    return [(F.conj(a), F.conj(b)) for a, b in M]


def classify(problem, verts):
    """Group vertices into classes under GL(Lambda_c) (GL_2(O_K) for O_K^2), and
    record Galois relations where the chart is Galois-stable."""
    F = problem.F
    in_aut = problem.chart.in_aut if problem.chart is not None else None
    classes = []
    for v in verts:
        placed = False
        for c in classes:
            rep = c["rep"]
            if abs(rep["gamma"] - v["gamma"]) > mpmath.mpf(10) ** (-30) * rep["gamma"]:
                continue
            if equivalence(F, rep["vectors"], v["vectors"], in_aut) is not None:
                c["members"].append(v)
                placed = True
                break
        if not placed:
            classes.append({"rep": v, "members": [v]})
    for c in classes:
        c["self_conjugate"] = False
        c["conjugate_of"] = None
        if problem.galois_ok:
            rep = c["rep"]
            c["self_conjugate"] = equivalence(F, rep["vectors"], galois_vectors(F, rep["vectors"]), in_aut) is not None
    if problem.galois_ok:
        for i, c in enumerate(classes):
            for j, d in enumerate(classes):
                if i < j and c["conjugate_of"] is None and not c["self_conjugate"] and \
                        abs(c["rep"]["gamma"] - d["rep"]["gamma"]) < mpmath.mpf(10) ** (-30) * c["rep"]["gamma"] and \
                        equivalence(F, c["rep"]["vectors"], galois_vectors(F, d["rep"]["vectors"]), in_aut) is not None:
                    d["conjugate_of"] = i
    return classes


# ------------------------------------------------------------ normal form
def normal_form(problem, vert):
    """Normal form ((1, b1; b1, c1), (1, b2; b2, c2)) with m(S) = 1 and e1 minimal.
    O_K^2 (h_K = 1): PW normal form, U in GL_2(O_K) with minimal first (and, if
    a unimodular pair exists, second) column.  Chart Lambda_c: the form is kept
    in chart coordinates; if e1 is not minimal at the vertex, the form is
    normalised by a minimal vector instead (flag e1_minimal = False)."""
    F = problem.F
    M = vert["vectors"]
    q = vert["q"]
    old_dps = mpmath.mp.dps
    mpmath.mp.dps = vert["dps"] + 10
    x1, x2, y1, y2 = q[0], q[1], mpmath.exp(q[2]), mpmath.exp(q[3])
    base = [mpmath.matrix([[1, x], [x, x * x + y * y]]) / y for x, y in ((x1, y1), (x2, y2))]
    if problem.chart is None:
        pair = None
        for u, w in itertools.combinations(M, 2):
            if F.is_unit(m_det(F, cols(u, w))):
                pair = (u, w)
                break
        if pair is not None:
            U = cols(*pair)
        else:
            u = M[0]
            g, d = F.complete_to_sl2(*u)
            U = ((u[0], g), (u[1], d))
        out = []
        for j in range(2):
            Uj = mpmath.matrix([[F.emb_mp(U[i][k])[j] for k in range(2)] for i in range(2)])
            Sp = Uj.T * base[j] * Uj
            out.append(Sp / Sp[0, 0])
        mpmath.mp.dps = old_dps
        Uinv = m_inv(F, U)
        newM = [tuple(F.as_int(c) for c in m_vec(F, Uinv, v)) for v in M]
        return {"S1": out[0], "S2": out[1], "U": U, "unimodular_pair": pair is not None,
                "vectors": newM, "gamma": vert["gamma"], "e1_minimal": True}
    e1min = any(v[1] == (0, 0) or tuple(v[1]) == (0, 0) for v in M)
    out = []
    if e1min:
        out = [b / b[0, 0] for b in base]
    else:
        u = M[0]
        for j in range(2):
            a, b = F.emb_mp(u[0])[j], F.emb_mp(u[1])[j]
            val = base[j][0, 0] * a * a + 2 * base[j][0, 1] * a * b + base[j][1, 1] * b * b
            out.append(base[j] / val)
    mpmath.mp.dps = old_dps
    return {"S1": out[0], "S2": out[1], "U": None, "unimodular_pair": None,
            "vectors": list(M), "gamma": vert["gamma"], "e1_minimal": e1min}


# ------------------------------------------------------------ identification
def identify(x, max_degree=8, maxcoeff=10 ** 9, dps=None):
    """Minimal integer polynomial of x found by PSLQ (mpmath.findpoly), searched
    by increasing degree, at the working precision of x (>= 50 digits).
    A relation is accepted only if it holds to ~(dps - 15) digits and uses at
    most ~60% of the available digits (otherwise PSLQ can return spurious
    relations).  Returns (coeffs, description) or (None, None)."""
    dps = dps or max(mpmath.mp.dps, 50)
    with mpmath.workdps(dps):
        x = mpmath.mpf(x)
        for n in range(1, max_degree + 1):
            c = mpmath.findpoly(x, n, maxcoeff=maxcoeff, maxsteps=200000)
            if not c:
                continue
            c = [int(t) for t in c]
            g = math.gcd(*c)
            c = [t // g for t in c]
            if c[0] < 0:
                c = [-t for t in c]
            size = max(abs(t) for t in c)
            resid = abs(mpmath.polyval(c, x)) / (size * max(1, abs(x)) ** n)
            if resid > mpmath.mpf(10) ** (-(dps - 15)):
                continue
            if (n + 1) * math.log10(size + 1) > 0.6 * dps:
                continue
            return c, describe(c, x)
    return None, None


def _squarefree_part(n):
    k, m = 1, n
    p = 2
    while p * p <= m:
        while m % (p * p) == 0:
            m //= p * p
            k *= p
        p += 1
    return k, m


def describe(c, x):
    if len(c) == 2:
        return str(Fraction(-c[1], c[0]))
    if len(c) == 3:
        A, B, C = c
        disc = B * B - 4 * A * C
        if disc < 0:
            return None
        k, m = _squarefree_part(disc)
        if m == 1:
            r1, r2 = Fraction(-B + k, 2 * A), Fraction(-B - k, 2 * A)
            return str(r1 if abs(float(r1) - float(x)) < abs(float(r2) - float(x)) else r2)
        for sgn in (1, -1):
            val = (-B + sgn * k * mpmath.sqrt(m)) / (2 * A)
            if abs(val - x) < mpmath.mpf(10) ** (-20) * abs(x):
                P, Q, R = -B, sgn * k, 2 * A
                g = math.gcd(math.gcd(P, Q), R)
                P, Q, R = P // g, Q // g, R // g
                if R < 0:
                    P, Q, R = -P, -Q, -R
                rad = ("" if abs(Q) == 1 else "%d*" % abs(Q)) + "sqrt(%d)" % m
                if P == 0:
                    num = ("-" if Q < 0 else "") + rad
                    return num if R == 1 else "%s/%d" % (num, R)
                num = "%d %s %s" % (P, "+" if Q > 0 else "-", rad)
                return num if R == 1 else "(%s)/%d" % (num, R)
    return "root of " + " ".join("%+d*x^%d" % (t, len(c) - 1 - i) for i, t in enumerate(c))


def closed_form_gamma(gamma, dps=100):
    """Closed form of gamma, searched in order of simplicity: gamma rational or
    quadratic; gamma^2 quadratic (gamma = sqrt of a quadratic irrationality);
    gamma of degree <= 4; gamma^2 of degree <= 8.  Every relation is verified
    to >= dps - 15 digits and must use at most ~60% of the available digits."""
    with mpmath.workdps(dps):
        tries = [(gamma, 2, 10 ** 12, False), (gamma ** 2, 2, 10 ** 14, True),
                 (gamma, 4, 10 ** 12, False), (gamma ** 2, 8, 10 ** 14, True)]
        for x, deg, mc, sq in tries:
            c, s = identify(x, max_degree=deg, maxcoeff=mc, dps=dps)
            if c is None:
                continue
            if sq:
                if s is not None and not s.startswith("root"):
                    return c, "sqrt(%s)" % s
                return c, "sqrt(x), x " + (s or "")
            return c, s
    return None, None


# ------------------------------------------------------------ local certificate
def ball_certificate(vert, safety=0.9):
    """Certified neighbourhood of a perfect eutactic vertex on which G < gamma*
    (except at the vertex).  Works in scaled coordinates
        q~ = (x1/s1, x2/s2, log y1, log y2),   s_j = y_j at the vertex,
    which removes the anisotropy of very skewed vertices.  For a unit vector u
    (Euclidean in q~) and h > 0,
        min_{v in M} phi_v(q~* + h u) <= phi* - kappa h + H h^2 / 2,
      kappa = dist(0, boundary conv{grad~ phi_v})  (> 0: perfect + eutactic),
      H = sup of ||Hess~ phi_v|| over the ball <= (4 e^{4 rho} + (128/27) e^{2 rho} + 1)^(1/2),
    from |phi_xx| <= 2/y^2, |phi_xu| <= 8/(3 sqrt3 y), |phi_uu| <= 1 and y_j >= s_j e^{-rho}.
    Returns (rho, kappa, scale) with scale = (s1, s2, 1, 1)."""
    q = vert["q"]
    y = np.exp([float(q[2]), float(q[3])])
    scale = np.array([y[0], y[1], 1.0, 1.0])
    g = np.array([[float(c) for c in gr] for gr in vert["grads"]]) * scale[None, :]
    try:
        hull = ConvexHull(g)
    except Exception:
        return None, None, None
    offs = hull.equations[:, -1]
    if np.any(offs >= 0):
        return None, None, None
    kappa = float(np.min(-offs))

    def H(r):
        return math.sqrt(4 * math.exp(4 * r) + (128.0 / 27) * math.exp(2 * r) + 1)
    rho = 0.1
    for _ in range(100):
        new = min(0.5, safety * 2 * kappa / H(rho))
        if abs(new - rho) < 1e-14:
            break
        rho = new
    assert rho < 2 * kappa / H(rho)
    return rho, kappa, scale


# ------------------------------------------------ classification across charts
def chart_basis(problem):
    """Matrix P with P Lambda_c = O_K^2 (columns p1, w of a Steinitz frame with
    content(p1) = c); identity for the O_K^2 problem."""
    F = problem.F
    if problem.chart is None:
        return ((1, 0), (0, 0)), ((0, 0), (1, 0))
    from .ideals import steinitz_frame
    c = problem.chart.c
    p1 = (c.e, c.f)
    _, w = steinitz_frame(F, p1)
    return ((p1[0], w[0]), (p1[1], w[1]))


def classify_global(problems, verts):
    """Classes of extreme forms up to GL_2(O_K) and scaling, over all charts:
    minimal vectors are mapped to O_K^2 by the chart bases, then compared with
    [PW05, Alg. 11] in O_K^2 (valid for any class number).  A representative at
    a highest point of its chart (e1 minimal) is preferred."""
    F = problems[0].F
    bases = [chart_basis(P) for P in problems]
    for v in verts:
        B = bases[v["chart"]]
        v["vectors_OK"] = [m_vec(F, B, u) for u in v["vectors"]]
        v["e1_min"] = any(tuple(u[1]) == (0, 0) for u in v["vectors"])
    order = sorted(verts, key=lambda v: (-v["gamma"], not v["e1_min"]))
    classes = []
    for v in order:
        for c in classes:
            r = c["rep"]
            if abs(r["gamma"] - v["gamma"]) <= mpmath.mpf(10) ** (-30) * r["gamma"] and \
                    equivalence(F, r["vectors_OK"], v["vectors_OK"]) is not None:
                c["members"].append(v)
                break
        else:
            classes.append({"rep": v, "members": [v]})
    for c in classes:
        M = c["rep"]["vectors_OK"]
        c["self_conjugate"] = equivalence(F, M, galois_vectors(F, M)) is not None
        c["conjugate_of"] = None
    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            if i < j and d["conjugate_of"] is None and not c["self_conjugate"] and \
                    abs(c["rep"]["gamma"] - d["rep"]["gamma"]) < mpmath.mpf(10) ** (-30) * c["rep"]["gamma"] and \
                    equivalence(F, c["rep"]["vectors_OK"], galois_vectors(F, d["rep"]["vectors_OK"])) is not None:
                d["conjugate_of"] = i
    return classes
