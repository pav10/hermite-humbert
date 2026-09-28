"""
Ideals, class group and Steinitz charts for binary Humbert forms over a real
quadratic field of arbitrary class number.

Steinitz.  If v in O_K^2 has content c' = alpha O_K + beta O_K, then with
gamma, delta in c'^{-1}, alpha*delta - beta*gamma = 1, and w = (gamma, delta),
    O_K^2 = c'^{-1} v  (+)  c' w.
If v is a minimal vector of S, then c' has minimal norm in its ideal class
(lambda in c'^{-1} with |N(lambda)| < 1 would give a shorter vector lambda*v).
Fix, for every class C, an integral ideal c of minimal norm in C.  Then c' = mu c
with |N(mu)| = 1, and in the K-basis (mu^{-1} v, mu w) the lattice O_K^2 becomes
    Lambda_c = c^{-1} (+) c,
the change of basis has |N(det)| = 1, and e1 is a minimal vector.  So every form
is equivalent to a form on some Lambda_c whose first basis vector is minimal:
one search region ("chart") per ideal class = the cusps of SL_2(O_K).
The stabiliser of the line K e1 in GL(Lambda_c) acts by translations by c^{-2}
and by units.  Galois maps the chart of C to the chart of C^{-1} = [c^sigma].
"""
import math
from fractions import Fraction

from field import RealQuadraticField, _primes_upto


def _F(x):
    return (Fraction(x[0]), Fraction(x[1]))


def _norm_int(x):
    """Tuple with integer entries where possible (hash/equality friendly)."""
    return tuple(int(c) if Fraction(c).denominator == 1 else Fraction(c) for c in x)


# ---------------------------------------------------------------- Z-lattices
def _xgcd(a, b):
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def hnf2(vecs, track=False):
    """Z-span of integer vectors in Z^2 (rank 2).  Returns basis ((gx, 0), (px, gy))
    with gx, gy > 0, 0 <= px < gx; with track=True also integer coefficient
    vectors expressing the two basis vectors in terms of the input."""
    n = len(vecs)
    rows = [[v[0], v[1]] + ([int(i == j) for j in range(n)] if track else []) for i, v in enumerate(vecs)]
    rows = [r for r in rows if r[0] or r[1]] if not track else rows
    # eliminate second coordinate down to a single pivot
    piv = None
    for r in rows:
        if r[1] == 0:
            continue
        if piv is None:
            piv = r
            continue
        g, s, t = _xgcd(piv[1], r[1])
        a, b = piv[1] // g, r[1] // g
        newp = [s * x + t * y for x, y in zip(piv, r)]
        newr = [b * x - a * y for x, y in zip(piv, r)]
        piv[:] = newp
        r[:] = newr
    if piv is None:
        raise ValueError("rank < 2")
    if piv[1] < 0:
        piv[:] = [-x for x in piv]
    # first coordinates of the remaining rows (second coordinate 0)
    xr = None
    for r in rows:
        if r is piv or r[0] == 0:
            continue
        if xr is None:
            xr = r
            continue
        g, s, t = _xgcd(xr[0], r[0])
        a, b = xr[0] // g, r[0] // g
        newx = [s * x + t * y for x, y in zip(xr, r)]
        newr = [b * x - a * y for x, y in zip(xr, r)]
        xr[:] = newx
        r[:] = newr
    if xr is None:
        raise ValueError("rank < 2")
    if xr[0] < 0:
        xr[:] = [-x for x in xr]
    q = piv[0] // xr[0]
    piv[:] = [x - q * y for x, y in zip(piv, xr)]
    if track:
        return (xr[0], 0), (piv[0], piv[1]), xr[2:], piv[2:]
    return (xr[0], 0), (piv[0], piv[1])


class KLattice:
    """Z-lattice in K with basis (e, f) (elements a + b w as pairs of rationals)."""

    def __init__(self, F, e, f):
        self.F = F
        self.e, self.f = _norm_int(e), _norm_int(f)
        self.std = (self.e == (1, 0) and self.f == (0, 1))
        d = Fraction(self.e[0]) * Fraction(self.f[1]) - Fraction(self.e[1]) * Fraction(self.f[0])
        if d == 0:
            raise ValueError("degenerate basis")
        self.det = d
        self.index = abs(d)                    # covolume relative to O_K
        self.emb_e, self.emb_f = F.emb(self.e), F.emb(self.f)

    @classmethod
    def from_gens(cls, F, gens):
        den = 1
        for g in gens:
            for c in g:
                den = math.lcm(den, Fraction(c).denominator)
        iv = [(int(Fraction(g[0]) * den), int(Fraction(g[1]) * den)) for g in gens]
        b1, b2 = hnf2(iv)
        return cls(F, (Fraction(b1[0], den), Fraction(b1[1], den)), (Fraction(b2[0], den), Fraction(b2[1], den)))

    def elem(self, a, b):
        if self.std:
            return (a, b)
        return _norm_int((a * Fraction(self.e[0]) + b * Fraction(self.f[0]),
                          a * Fraction(self.e[1]) + b * Fraction(self.f[1])))

    def coords(self, x):
        x0, x1 = Fraction(x[0]), Fraction(x[1])
        e0, e1, f0, f1 = map(Fraction, (self.e[0], self.e[1], self.f[0], self.f[1]))
        return ((x0 * f1 - x1 * f0) / self.det, (e0 * x1 - e1 * x0) / self.det)

    def contains(self, x):
        a, b = self.coords(x)
        return a.denominator == 1 and b.denominator == 1

    def points_near(self, c1, h1, c2, h2):
        """Elements x with |sigma_1(x) - c1| <= h1, |sigma_2(x) - c2| <= h2
        (Gauss-reduced enumeration; tiny outward margin)."""
        F = self.F
        h1 = h1 * (1 + 1e-9) + 1e-12
        h2 = h2 * (1 + 1e-9) + 1e-12
        u = [self.emb_e[0] / h1, self.emb_e[1] / h2]
        v = [self.emb_f[0] / h1, self.emb_f[1] / h2]
        U = [[1, 0], [0, 1]]

        def dot(p, q):
            return p[0] * q[0] + p[1] * q[1]
        for _ in range(200):
            if dot(u, u) > dot(v, v):
                u, v = v, u
                U = [[U[0][1], U[0][0]], [U[1][1], U[1][0]]]
            mu = round(dot(u, v) / dot(u, u))
            if mu == 0:
                break
            v = [v[0] - mu * u[0], v[1] - mu * u[1]]
            U = [[U[0][0], U[0][1] - mu * U[0][0]], [U[1][0], U[1][1] - mu * U[1][0]]]
        t = [c1 / h1, c2 / h2]
        det = u[0] * v[1] - u[1] * v[0]
        y0 = [(t[0] * v[1] - t[1] * v[0]) / det, (u[0] * t[1] - u[1] * t[0]) / det]
        g11, g12, g22 = dot(u, u), dot(u, v), dot(v, v)
        q22 = g22 - g12 * g12 / g11
        R2 = 2.0 * (1 + 1e-9)
        out = []
        r2 = math.sqrt(R2 / q22)
        for y2 in range(math.floor(y0[1] - r2), math.ceil(y0[1] + r2) + 1):
            rem = R2 - q22 * (y2 - y0[1]) ** 2
            if rem < 0:
                continue
            cen = y0[0] - g12 / g11 * (y2 - y0[1])
            r1 = math.sqrt(rem / g11)
            for y1 in range(math.floor(cen - r1), math.ceil(cen + r1) + 1):
                a = U[0][0] * y1 + U[0][1] * y2
                b = U[1][0] * y1 + U[1][1] * y2
                x = self.elem(a, b)
                s1, s2 = F.emb(x)
                if abs(s1 - c1) <= h1 and abs(s2 - c2) <= h2:
                    out.append(x)
        return out

    def points_in_intervals(self, I1, I2):
        (l1, u1), (l2, u2) = I1, I2
        return self.points_near(0.5 * (l1 + u1), 0.5 * (u1 - l1), 0.5 * (l2 + u2), 0.5 * (u2 - l2))

    def norm_reps(self, B):
        """Nonzero elements of the lattice with |N| <= B modulo units (balanced
        representatives: eps^-1 <= |x1/x2| < eps, x1 > 0), grouped by |N|."""
        F = self.F
        eps = F.eps0()
        L = 2.0 * math.log(eps)
        m = max(1, int(math.ceil(L)))
        groups, seen = {}, set()
        for j in range(m):
            rl = math.exp(-0.5 * L + L * j / m)
            rh = math.exp(-0.5 * L + L * (j + 1) / m)
            for x in self.points_near(0.0, math.sqrt(B * rh), 0.0, math.sqrt(B / rl)):
                if x in seen:
                    continue
                seen.add(x)
                n = abs(Fraction(F.norm(x)))
                if n == 0 or n > B:
                    continue
                reps = groups.setdefault(n, [])
                if not any(F.associated(x, y) for y in reps):
                    reps.append(x)
        e = F.fundamental_unit()
        einv = F.as_int(F.inv(e))
        for n, reps in groups.items():
            for i, x in enumerate(reps):
                for _ in range(200):
                    a1, a2 = F.emb(x)
                    r = abs(a1 / a2)
                    if r >= eps:
                        x = F.mul(x, einv)
                    elif r < 1.0 / eps:
                        x = F.mul(x, e)
                    else:
                        break
                if F.emb(x)[0] < 0:
                    x = F.neg(x)
                reps[i] = _norm_int(x)
        return groups


# ------------------------------------------------------------------- ideals
class Ideal(KLattice):
    """Fractional ideal of O_K given by a Z-basis."""

    @classmethod
    def generated_by(cls, F, elems):
        gens = []
        for x in elems:
            gens += [x, F.mul(x, (0, 1))]
        return cls.from_gens(F, gens)

    def norm(self):
        return self.index

    def mul(self, other):
        F = self.F
        gens = [F.mul(a, b) for a in (self.e, self.f) for b in (other.e, other.f)]
        return Ideal.from_gens(F, gens)

    def scale(self, x):
        F = self.F
        return Ideal(F, F.mul(self.e, x), F.mul(self.f, x))

    def conj(self):
        F = self.F
        return Ideal(F, F.conj(self.e), F.conj(self.f))

    def inverse(self):
        n = self.norm()
        return self.conj().scale((Fraction(1) / n, Fraction(0)))

    def is_integral(self):
        return all(Fraction(c).denominator == 1 for c in self.e + self.f)

    def generator(self):
        """A generator if the (integral) ideal is principal, else None."""
        assert self.is_integral()
        n = self.norm()
        reps = self.norm_reps(int(n))
        g = reps.get(n)
        return g[0] if g else None

    def __repr__(self):
        return "Ideal<%s, %s; N=%s>" % (self.F.fmt(self.e), self.F.fmt(self.f), self.norm())


def integral_ideals_of_norm(F, n):
    """All integral ideals of norm n: Z-bases {a, b + c w} with a c = n, c | a,
    c | b, 0 <= b < a, closed under multiplication by w."""
    out = []
    for c in range(1, n + 1):
        if n % c:
            continue
        a = n // c
        if a % c:
            continue
        for b in range(0, a, c):
            k = b // c + F.tw                # w (b + c w) = -c nw + (b + c tw) w
            if (c * F.nw + k * b) % a == 0:
                out.append(Ideal(F, (a, 0), (b, c)))
    return out


def equivalent(I, J):
    """I ~ J  <=>  I J^sigma is principal (both integral)."""
    return I.mul(J.conj()).generator() is not None


class Chart:
    """Search chart for the ideal class of c (integral, minimal norm in its class):
    alpha in A = c^{-1}, beta in B = c, translations T = c^{-2}."""

    def __init__(self, F, c, index, inverse_index=None, galois=False):
        self.F, self.c, self.index = F, c, index
        self.A = c.inverse()
        self.B = c
        self.T = c.inverse().mul(c.inverse())
        self.T2 = c.mul(c)                    # U21 must lie in c^2
        self.trivial = c.norm() == 1
        self.inverse_index = inverse_index
        self.galois = galois                  # Galois symmetry usable inside this chart

    def label(self):
        if self.trivial:
            return "O_K"
        return "c = <%s, %s> (norm %s)" % (self.F.fmt(self.c.e), self.F.fmt(self.c.f), self.c.norm())

    def in_aut(self, U):
        """U in GL(Lambda_c): U11, U22 in O_K, U12 in c^-2, U21 in c^2, det a unit."""
        F = self.F
        ok = (F.is_integral(U[0][0]) and F.is_integral(U[1][1])
              and self.T.contains(U[0][1]) and self.T2.contains(U[1][0]))
        if not ok:
            return False
        d = F.sub(F.mul(U[0][0], U[1][1]), F.mul(U[0][1], U[1][0]))
        return F.is_unit(d)


def class_group_charts(F):
    """One chart per ideal class, representative of minimal norm; for a pair of
    mutually inverse classes C != C^{-1} only C is kept (Galois maps the chart
    of C onto that of C^{-1}).  Returns (charts, h)."""
    M = int(math.floor(math.sqrt(F.disc) / 2.0))
    reps = [Ideal(F, (1, 0), (0, 1))]
    for n in range(2, M + 1):
        for I in integral_ideals_of_norm(F, n):
            if not any(equivalent(I, R) for R in reps):
                reps.append(I)
    h = len(reps)
    inv = []
    for R in reps:
        inv.append(next(j for j, S in enumerate(reps) if equivalent(R.conj(), S)))
    charts, dropped = [], set()
    for i, R in enumerate(reps):
        if i in dropped:
            continue
        j = inv[i]
        if j != i:
            dropped.add(j)
        charts.append(Chart(F, R, i, inverse_index=j, galois=(i == 0)))
    return charts, h


def steinitz_frame(F, v):
    """For v = (alpha, beta) in O_K^2 with content c': (c', w) with w in K^2,
    det(v, w) = 1 and O_K^2 = c'^{-1} v + c' w."""
    alpha, beta = v
    cp = Ideal.generated_by(F, [alpha, beta])
    ci = cp.inverse()
    e, f = ci.e, ci.f
    gens = [F.mul(alpha, e), F.mul(alpha, f), F.neg(F.mul(beta, e)), F.neg(F.mul(beta, f))]
    den = 1
    for g in gens:
        for c in g:
            den = math.lcm(den, Fraction(c).denominator)
    assert den == 1, "alpha c'^-1 + beta c'^-1 should be O_K"
    iv = [(int(g[0]), int(g[1])) for g in gens]
    b1, b2, t1, t2 = hnf2(iv, track=True)
    assert b1 == (1, 0) and b2[1] == 1, (b1, b2)
    t = t1                                    # combination giving (1, 0)
    delta = F.add(F.mul((t[0], 0), e), F.mul((t[1], 0), f))
    gamma = F.add(F.mul((t[2], 0), e), F.mul((t[3], 0), f))
    assert F.sub(F.mul(alpha, delta), F.mul(beta, gamma)) == (1, 0)
    return cp, (_norm_int(gamma), _norm_int(delta))


def locate_chart(F, charts, cp):
    """Chart of the class of the integral ideal cp: (chart, conj) where conj=True
    means the class of cp is the (dropped) inverse of the chart's class, i.e. the
    Galois conjugate form belongs to the chart."""
    for ch in charts:
        if equivalent(cp, ch.c):
            return ch, False
    for ch in charts:
        if equivalent(cp.conj(), ch.c):
            return ch, True
    raise RuntimeError("ideal class not found")
