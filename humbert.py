"""
Binary Humbert forms over a real quadratic field K, viewed as points of H^2.

Conventions (Icaza; [BCIO] Baeza--Coulangeon--Icaza--O'Ryan, Exp. Math. 10 (2001);
[PW05] Pohst--Wagner, JTNB 17 (2005); [PW09] Pohst--Wagner, J. Algebra 322 (2009)):

  S = (S1, S2) positive definite 2x2,  S[v] = S1[v^(1)] * S2[v^(2)]  for v in O_K^2,
  m(S) = min_{v != 0} S[v],   gamma(S) = m(S) / sqrt(det S1 det S2),
  gamma_{K,2} = sup_S gamma(S).

A point z = (z1, z2) in H^2, z_j = x_j + i y_j, gives the form
  M(z_j) = (1/y_j) [[1, x_j], [x_j, x_j^2 + y_j^2]]   (det = 1),
with M(z_j)[(a, b)] = |a + b z_j|^2 / y_j.  Every S is a positive multiple (per
component) of such a form, so gamma(S) = G(z) where

  f_v(z) = |a1 + b1 z1|^2 |a2 + b2 z2|^2 / (y1 y2),   v = (alpha, beta),
  G(z)   = min_{v != 0} f_v(z)                   (a_j = sigma_j(alpha), b_j = sigma_j(beta)).

G is invariant under GL_2(O_K) (acting on forms by S -> S[U]) and under
swapping the two embeddings (Galois).  Units act trivially on f_v.
"""
import math
import itertools

import numpy as np
import mpmath

from field import RealQuadraticField


# ----------------------------------------------------------------------------
# small lattice tools on Z^n (n = 4 here)
# ----------------------------------------------------------------------------
def lll_gram(G, delta=0.99):
    """LLL-reduce the lattice with Gram matrix G (float).  Returns an integer
    unimodular matrix T (list of lists, columns = new basis in old coords)."""
    n = G.shape[0]
    L = np.linalg.cholesky(G)          # G = L L^T ; basis vectors = columns of L^T
    Bm = L.T.copy()
    T = [[int(i == j) for j in range(n)] for i in range(n)]

    def gso(Bm):
        Bs = np.zeros_like(Bm)
        mu = np.zeros((n, n))
        bb = np.zeros(n)
        for i in range(n):
            v = Bm[:, i].copy()
            for j in range(i):
                mu[i, j] = Bm[:, i] @ Bs[:, j] / bb[j]
                v -= mu[i, j] * Bs[:, j]
            Bs[:, i] = v
            bb[i] = v @ v
        return mu, bb

    k = 1
    mu, bb = gso(Bm)
    it = 0
    while k < n and it < 10000:
        it += 1
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q:
                Bm[:, k] -= q * Bm[:, j]
                for r in range(n):
                    T[r][k] -= q * T[r][j]
                mu, bb = gso(Bm)
        if bb[k] >= (delta - mu[k, k - 1] ** 2) * bb[k - 1]:
            k += 1
        else:
            Bm[:, [k - 1, k]] = Bm[:, [k, k - 1]]
            for r in range(n):
                T[r][k - 1], T[r][k] = T[r][k], T[r][k - 1]
            mu, bb = gso(Bm)
            k = max(k - 1, 1)
    return T


def fincke_pohst(G, bound):
    """All x in Z^n, x != 0, with x^T G x <= bound (one of each pair +-x).
    G float positive definite.  LLL-preconditioned; small relative slack."""
    n = G.shape[0]
    T = lll_gram(G)
    Tm = np.array(T, dtype=float)
    Gr = Tm.T @ G @ Tm
    Q = Gr.copy()
    for i in range(n):
        for j in range(i + 1, n):
            Q[j, i] = Q[i, j]
            Q[i, j] = Q[i, j] / Q[i, i]
        for k in range(i + 1, n):
            for l in range(k, n):
                Q[k, l] -= Q[k, i] * Q[i, l]
    slack = 1e-9 * bound + 1e-12
    res = []
    x = [0] * n

    def rec(i, rem):
        c = -sum(Q[i, j] * x[j] for j in range(i + 1, n))
        r = math.sqrt(max(rem, 0.0) / Q[i, i])
        for xi in range(math.ceil(c - r - 1e-9), math.floor(c + r + 1e-9) + 1):
            t = Q[i, i] * (xi - c) ** 2
            if t > rem + slack:
                continue
            x[i] = xi
            if i == 0:
                res.append(tuple(x))
            else:
                rec(i - 1, rem - t)
        x[i] = 0

    rec(n - 1, bound)
    out = []
    for y in res:
        if not any(y):
            continue
        z = tuple(sum(T[r][c] * y[c] for c in range(n)) for r in range(n))
        # keep one of +-z: first nonzero coordinate positive
        for c in z:
            if c:
                if c < 0:
                    z = tuple(-t for t in z)
                break
        out.append(z)
    return sorted(set(out))


# ----------------------------------------------------------------------------
# the problem for one field
# ----------------------------------------------------------------------------
class HumbertProblem:
    """Binary Humbert forms over K = Q(sqrt D).

    Search region R (coordinates r1, r2, s, t):
        x_j = r1 + r2 w_j,   y1 = exp(t - s),   y2 = exp(t + s),
        r1 in [-1/2, 1/2],  r2 in [0, 1/2],  s in [0, log(eps0)/2],
        t in [t_lo, t_hi],  t_lo = log(2/d)/2,  t_hi = log(1/gamma_low)/2.
    Justification (h_K = 1): every orbit has a highest point z0 (max y1*y2),
    and there G(z0) = 1/(y1 y2) because e1 is then minimal [BCIO Prop 1.1];
    G <= d/2 [Cohn; also gamma_4 = sqrt2 + AM-GM] gives y1 y2 >= 2/d; translations
    by O_K, z -> eps0^k z (any unit, via diag(1, eps0^k) in GL_2(O_K)), x -> -x
    (diag(1,-1)) and the Galois swap preserve G and y1 y2 and move z0 into R.
    """

    def __init__(self, D, require_h1=True, chart=None):
        self.F = F = RealQuadraticField(D)
        F.check_float_range()
        self.h1 = F.class_number_is_one()
        if require_h1 and not self.h1:
            raise ValueError("h_K > 1 for D = %d: the region argument needs h_K = 1" % D)
        self.D, self.d = D, F.disc
        self.w1, self.w2 = F.w1, F.w2
        self.eps = F.eps0()
        self.unit = F.fundamental_unit()
        self.s_max = 0.5 * math.log(self.eps)
        self.t_lo = 0.5 * math.log(2.0 / self.d)
        self.V = None
        # chart: lattice Lambda_c = A (+) B with A = c^{-1}, B = c, translations by T = c^{-2}
        # (chart None = O_K^2, the h_K = 1 setting).  Galois (s >= 0) only if the chart allows it.
        self.chart = chart
        from ideals import KLattice
        if chart is None:
            O = KLattice(F, (1, 0), (0, 1))
            self.A = self.B = self.T = O
            self.galois_ok = True
        else:
            self.A, self.B, self.T = chart.A, chart.B, chart.T
            self.galois_ok = chart.galois
        self.s_min = 0.0 if self.galois_ok else -self.s_max
        self.g1, self.g2 = self.T.emb_e, self.T.emb_f      # embedded translation basis
        # primitive filter (v = lambda v0 with lambda a non-unit) is only used when h_K = 1
        self.use_primitive = self.h1 and chart is None

    # ------------------------------------------------------------ coordinates
    def rst_to_p(self, r1, r2, s, t):
        g1, g2 = self.g1, self.g2
        return np.array([r1 * g1[0] + r2 * g2[0], r1 * g1[1] + r2 * g2[1], math.exp(t - s), math.exp(t + s)])

    def x_to_r(self, x1, x2):
        g1, g2 = self.g1, self.g2
        det = g1[0] * g2[1] - g2[0] * g1[1]
        return (x1 * g2[1] - x2 * g2[0]) / det, (g1[0] * x2 - g1[1] * x1) / det

    def p_to_rst(self, p):
        x1, x2, y1, y2 = p
        r1, r2 = self.x_to_r(x1, x2)
        return r1, r2, 0.5 * (math.log(y2) - math.log(y1)), 0.5 * (math.log(y1) + math.log(y2))

    # ------------------------------------------------------------- group action
    def forms_of(self, p):
        x1, x2, y1, y2 = p
        return [np.array([[1.0, x], [x, x * x + y * y]]) / y for x, y in ((x1, y1), (x2, y2))]

    def point_of_forms(self, S1, S2):
        out = []
        for S in (S1, S2):
            a, b, c = S[0, 0], S[0, 1], S[1, 1]
            out.append((b / a, math.sqrt(a * c - b * b) / a))
        return np.array([out[0][0], out[1][0], out[0][1], out[1][1]])

    def emb_matrix(self, U):
        """U = ((u11, u12), (u21, u22)) with entries in K -> its two real embeddings."""
        E = [[self.F.emb(u) for u in row] for row in U]
        return [np.array([[E[i][j][k] for j in range(2)] for i in range(2)]) for k in range(2)]

    def act(self, p, U):
        """Point of the form S(p)[U] = (U1^T S1 U1, U2^T S2 U2)."""
        S1, S2 = self.forms_of(p)
        U1, U2 = self.emb_matrix(U)
        return self.point_of_forms(U1.T @ S1 @ U1, U2.T @ S2 @ U2)

    def galois(self, p):
        x1, x2, y1, y2 = p
        return np.array([x2, x1, y2, y1])

    def to_region(self, p):
        """Equivalent point with r in [-1/2,1/2] x [0,1/2] and s in [s_min, s_max]
        (t unchanged).  Uses unit scaling, (if allowed) the Galois swap,
        translations by T and x -> -x; all preserve Lambda_c and G."""
        x1, x2, y1, y2 = [float(c) for c in p]
        le = math.log(self.eps)
        k = round(math.log(y2 / y1) / (2 * le))
        if k:
            u1, u2 = self.F.emb(self.F.unit_power(k))  # z_j -> u_j x_j + i |u_j| y_j
            x1, x2, y1, y2 = u1 * x1, u2 * x2, abs(u1) * y1, abs(u2) * y2
        if self.galois_ok and y2 < y1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        r1, r2 = self.x_to_r(x1, x2)
        r1, r2 = r1 - round(r1), r2 - round(r2)
        if r2 < 0:
            r1, r2 = -r1, -r2
        g1, g2 = self.g1, self.g2
        return np.array([r1 * g1[0] + r2 * g2[0], r1 * g1[1] + r2 * g2[1], y1, y2])

    def in_exact_region(self, p, tol=1e-12):
        r1, r2, s, t = self.p_to_rst(p)
        return (-0.5 - tol <= r1 <= 0.5 + tol and -tol <= r2 <= 0.5 + tol
                and self.s_min - tol <= s <= self.s_max + tol and t >= self.t_lo - tol)

    # ---------------------------------------------------------- f_v and G(z)
    @staticmethod
    def fvals(p, A1, B1, A2, B2):
        x1, x2, y1, y2 = p
        return ((A1 + B1 * x1) ** 2 + (B1 * y1) ** 2) * ((A2 + B2 * x2) ** 2 + (B2 * y2) ** 2) / (y1 * y2)

    def vec_emb(self, vecs):
        E = np.array([self.F.emb(a) + self.F.emb(b) for a, b in vecs])  # (a1, a2, b1, b2)
        return E[:, 0], E[:, 2], E[:, 1], E[:, 3]                   # A1, B1, A2, B2

    def f_exact_list(self, p, vecs):
        A1, B1, A2, B2 = self.vec_emb(vecs)
        return self.fvals(p, A1, B1, A2, B2)

    # ---------------------------------------------- candidate vectors on R
    def x_ranges(self):
        """Ranges of x1, x2 over the translation half-cell r1 in [-1/2,1/2], r2 in [0,1/2]."""
        g1, g2 = self.g1, self.g2
        cs = [(r1 * g1[0] + r2 * g2[0], r1 * g1[1] + r2 * g2[1]) for r1 in (-0.5, 0.5) for r2 in (0.0, 0.5)]
        return (min(c[0] for c in cs), max(c[0] for c in cs)), (min(c[1] for c in cs), max(c[1] for c in cs))

    def candidate_vectors(self, C):
        """All unit classes of v = (alpha, beta) in Lambda_c = A (+) B with f_v(z) <= C
        for some z in R_V = {r in the half-cell, s in [s_min, s_max], t >= t_lo}
        (primitive ones only when h_K = 1).  Rigorous up to the 1e-9 float slack:
          beta = 0:  alpha e1 with alpha in c^{-1}, |N(alpha)| >= 1 (c has minimal norm
                     in its class), so e1 dominates;
          beta != 0: f_v >= N(beta)^2 y1 y2 >= 2 N(beta)^2/d, and
            (alpha/beta + x1)^2 <= C y1/(N(beta)^2 y2),  (alpha'/beta' + x2)^2 <= C y2/(N(beta)^2 y1),
          with y1/y2 <= 1 (Galois chart, s >= 0) or <= eps0, and y2/y1 <= eps0.
        On R_V, min_{v in V} f_v(z) = G(z) whenever this minimum is <= C."""
        F = self.F
        vecs = [((1, 0), (0, 0))]
        Nmax = int(math.floor(math.sqrt(C * self.d / 2.0) + 1e-9))
        X1, X2 = self.x_ranges()
        reps = self.B.norm_reps(Nmax) if Nmax >= 1 else {}
        r12 = 1.0 if self.galois_ok else self.eps
        for n, blist in sorted(reps.items()):
            n = float(n)
            R1 = math.sqrt(C * r12) / n
            R2 = math.sqrt(C * self.eps) / n
            for beta in blist:
                b1, b2 = F.emb(beta)
                lo1, hi1 = -X1[1] - R1, -X1[0] + R1
                lo2, hi2 = -X2[1] - R2, -X2[0] + R2
                I1 = sorted((b1 * lo1, b1 * hi1))
                I2 = sorted((b2 * lo2, b2 * hi2))
                for alpha in self.A.points_in_intervals(I1, I2):
                    if not self.use_primitive or F.is_primitive(alpha, beta):
                        vecs.append((alpha, beta))
        return (vecs,) + tuple(self.vec_emb(vecs))

    def build_V(self, C):
        """candidate_vectors(C), stored as the problem's working set V."""
        vecs, A1, B1, A2, B2 = self.candidate_vectors(C)
        self.V, self.VC = vecs, C
        self.VA1, self.VB1, self.VA2, self.VB2 = A1, B1, A2, B2
        return vecs

    def G_V(self, P, max_elems=2e6):
        """min_v f_v over V at points P (array (N, 4) or (4,)); returns (values, argmins).
        Evaluated in chunks of about max_elems (points x vectors)."""
        P = np.atleast_2d(P)
        step = max(1, int(max_elems // max(1, len(self.V))))
        vals, args = [], []
        for k in range(0, len(P), step):
            Q = P[k:k + step]
            f = self.fvals((Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], Q[:, 3:4]),
                           self.VA1[None, :], self.VB1[None, :], self.VA2[None, :], self.VB2[None, :])
            i = np.argmin(f, axis=1)
            vals.append(f[np.arange(len(Q)), i])
            args.append(i)
        return np.concatenate(vals), np.concatenate(args)

    # ------------------------------------------ exact minimum (independent of V)
    def short_vectors(self, p, bound, primitive_only=True):
        """All unit classes [v], v in O_K^2 \\ 0, with f_v(p) <= bound (float).
        Unit balancing: rho(v) = S1[v]/S2[v'] -> eps0^4 rho under v -> eps0 v, so each
        class has a member with rho in [eps0^-2, eps0^2).  That window is cut into
        k pieces of log-width <= 2; on the piece with centre rho_c the positive
        definite Z^4-form q = S1[v]/sqrt(rho_c) + sqrt(rho_c) S2[v'] satisfies
        q(v) <= sqrt(bound) (w^{1/4} + w^{-1/4}) (AM-GM), w = piece width."""
        F = self.F
        S1, S2 = self.forms_of(p)
        ae, af, be, bf = self.A.emb_e, self.A.emb_f, self.B.emb_e, self.B.emb_f
        E1 = np.array([[ae[0], af[0], 0, 0], [0, 0, be[0], bf[0]]])
        E2 = np.array([[ae[1], af[1], 0, 0], [0, 0, be[1], bf[1]]])
        G1, G2 = E1.T @ S1 @ E1, E2.T @ S2 @ E2
        Lw = 4.0 * math.log(self.eps)
        k = max(1, int(math.ceil(Lw / 2.0)))
        wl = Lw / k
        fac = math.exp(wl / 4) + math.exp(-wl / 4)
        found = {}
        for j in range(k):
            c = math.exp(0.5 * (-0.5 * Lw + (j + 0.5) * wl))
            for z in fincke_pohst(G1 / c + c * G2, math.sqrt(bound) * fac):
                v = (self.A.elem(z[0], z[1]), self.B.elem(z[2], z[3]))
                fv = float(self.f_exact_list(p, [v])[0])
                if fv <= bound * (1 + 1e-9):
                    if primitive_only and self.use_primitive and not F.is_primitive(*v):
                        continue
                    key = self.canonical(v, p)
                    found[key] = fv
        # exact dedupe of float-ambiguous canonical forms
        items = sorted(found.items(), key=lambda kv: kv[1])
        out = []
        for v, fv in items:
            if not any(abs(fv - fw) <= 1e-9 * fv and self.unit_associated(v, w) for w, fw in out):
                out.append((v, fv))
        return out

    def canonical(self, v, p):
        """Unit-associate of v with S1[v]/S2[v'] in [eps0^-2, eps0^2), sign-normalised."""
        F = self.F
        S1, S2 = self.forms_of(p)
        (a, b) = v
        a1, a2 = F.emb(a)
        b1, b2 = F.emb(b)
        rho = (S1[0, 0] * a1 * a1 + 2 * S1[0, 1] * a1 * b1 + S1[1, 1] * b1 * b1) / \
              (S2[0, 0] * a2 * a2 + 2 * S2[0, 1] * a2 * b2 + S2[1, 1] * b2 * b2)
        k = math.floor((math.log(rho) + 2 * math.log(self.eps)) / (4 * math.log(self.eps)))
        if k:
            u = F.unit_power(-k)
            a, b = F.mul(a, u), F.mul(b, u)
        flat = a + b
        for c in flat:
            if c:
                if c < 0:
                    a, b = F.neg(a), F.neg(b)
                break
        return (a, b)

    def unit_associated(self, v, w):
        F = self.F
        (a, b), (c, d) = v, w
        if (a == (0, 0)) != (c == (0, 0)) or (b == (0, 0)) != (d == (0, 0)):
            return False
        if a != (0, 0):
            e = F.div(c, a)
            if not F.is_unit(e):
                return False
            return F.mul(e, b) == d
        e = F.div(d, b)
        return F.is_unit(e) and F.mul(e, a) == c

    def minimum(self, p, rel_tol=1e-10):
        """(m, list of minimal classes) of the form at p, independent of V."""
        m0 = 1.0 / (p[2] * p[3])
        sv = self.short_vectors(p, m0 * (1 + 1e-9))
        m = min(fv for _, fv in sv)
        return m, [v for v, fv in sv if fv <= m * (1 + rel_tol)]

    def reduce_to_top(self, p):
        """Equivalent point where e1 is a minimal vector (highest point of the
        orbit), then moved into the region.  Uses the exact minimum routine."""
        if not self.use_primitive:
            raise NotImplementedError("reduction to the highest point across charts (h_K > 1)")
        m, M = self.minimum(p)
        if m < (1.0 / (p[2] * p[3])) * (1 - 1e-12):
            alpha, beta = M[0]
            gamma, delta = self.F.complete_to_sl2(alpha, beta)
            p = self.act(p, ((alpha, gamma), (beta, delta)))
        return self.to_region(p)
