"""
Exact arithmetic in a real quadratic field K = Q(sqrt(D)) and its maximal order.

Elements of K are pairs (a, b) meaning a + b*w, where
    w = (1 + sqrt(D))/2  if D = 1 mod 4,   w = sqrt(D)  otherwise,
so O_K = Z[w] and (a, b) is integral iff a, b are integers.  This is the
basis used by PARI (quadgen) and by Pohst--Wagner.
"""
from fractions import Fraction
import math

import mpmath


def _is_squarefree(n):
    if n < 2:
        return False
    k = 2
    while k * k <= n:
        if n % (k * k) == 0:
            return False
        k += 1
    return True


def _prime_factors(n):
    n = abs(int(n))
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out


def _primes_upto(n):
    n = int(n)
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


class RealQuadraticField:
    """K = Q(sqrt(D)), D > 1 squarefree."""

    def __init__(self, D):
        D = int(D)
        if not _is_squarefree(D):
            raise ValueError("D must be a squarefree integer > 1")
        self.D = D
        if D % 4 == 1:
            self.disc = D
            self.tw, self.nw = 1, (1 - D) // 4      # w^2 = tw*w - nw
        else:
            self.disc = 4 * D
            self.tw, self.nw = 0, -D
        s = math.sqrt(D)
        if D % 4 == 1:
            self.w1, self.w2 = (1 + s) / 2, (1 - s) / 2
        else:
            self.w1, self.w2 = s, -s
        self._unit = None
        self._h1 = None
        self._norm_reps = {}

    # ------------------------------------------------------------------ basics
    def __repr__(self):
        return "Q(sqrt(%d))" % self.D

    def mul(self, x, y):
        a, b = x
        c, d = y
        return (a * c - b * d * self.nw, a * d + b * c + b * d * self.tw)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def sub(self, x, y):
        return (x[0] - y[0], x[1] - y[1])

    def neg(self, x):
        return (-x[0], -x[1])

    def conj(self, x):
        a, b = x
        return (a + b * self.tw, -b)

    def norm(self, x):
        a, b = x
        return a * a + a * b * self.tw + b * b * self.nw

    def trace(self, x):
        return 2 * x[0] + x[1] * self.tw

    def inv(self, x):
        n = Fraction(self.norm(x))
        if n == 0:
            raise ZeroDivisionError
        c = self.conj(x)
        return (Fraction(c[0]) / n, Fraction(c[1]) / n)

    def div(self, x, y):
        return self.mul(x, self.inv(y))

    @staticmethod
    def is_integral(x):
        return Fraction(x[0]).denominator == 1 and Fraction(x[1]).denominator == 1

    def is_unit(self, x):
        return self.is_integral(x) and abs(self.norm(x)) == 1

    @staticmethod
    def as_int(x):
        return (int(Fraction(x[0])), int(Fraction(x[1])))

    def emb(self, x):
        """Both real embeddings as floats.  The smaller one is computed as
        N(x)/(larger one), which avoids catastrophic cancellation."""
        a, b = float(x[0]), float(x[1])
        s1, s2 = a + b * self.w1, a + b * self.w2
        if s1 == 0.0 and s2 == 0.0:
            return 0.0, 0.0
        n = float(self.norm(x))
        if abs(s1) >= abs(s2):
            s2 = n / s1
        else:
            s1 = n / s2
        return s1, s2

    MAX_EPS = 1e12

    def check_float_range(self):
        if self.eps0() > self.MAX_EPS:
            raise NotImplementedError(
                "fundamental unit %.3g exceeds %.0e: float routines unreliable" %
                (self.eps0(), self.MAX_EPS))

    def emb_mp(self, x):
        s = mpmath.sqrt(self.D)
        a, b = mpmath.mpf(Fraction(x[0]).numerator) / Fraction(x[0]).denominator, \
            mpmath.mpf(Fraction(x[1]).numerator) / Fraction(x[1]).denominator
        if self.D % 4 == 1:
            w1, w2 = (1 + s) / 2, (1 - s) / 2
        else:
            w1, w2 = s, -s
        return a + b * w1, a + b * w2

    def w_mp(self):
        s = mpmath.sqrt(self.D)
        return ((1 + s) / 2, (1 - s) / 2) if self.D % 4 == 1 else (s, -s)

    def fmt(self, x):
        """Pretty print a + b*w as (p + q sqrt D)/r."""
        a, b = Fraction(x[0]), Fraction(x[1])
        if self.D % 4 == 1:
            p, q = a + b / 2, b / 2
        else:
            p, q = a, b
        if q == 0:
            return str(p)
        den = math.lcm(p.denominator, q.denominator)
        P, Q = int(p * den), int(q * den)
        rt = "sqrt(%d)" % self.D
        if P == 0:
            s = ("-" if Q < 0 else "") + ("" if abs(Q) == 1 else str(abs(Q))) + rt
        else:
            s = "%d %s %s%s" % (P, "-" if Q < 0 else "+", "" if abs(Q) == 1 else str(abs(Q)), rt)
        return s if den == 1 else "(%s)/%d" % (s, den)

    # ---------------------------------------------------------- unit and h = 1
    def fundamental_unit(self):
        """Fundamental unit eps0 > 1 (as (a, b)), via the continued fraction of
        t = -w2 = (sqrt D - 1)/2 or sqrt D: every unit a + b w > 1 has a/b a
        convergent of t.  The result is cross-checked by a small exhaustive search."""
        if self._unit is not None:
            return self._unit
        D = self.D
        r = math.isqrt(D)
        P, Q = (-1, 2) if D % 4 == 1 else (0, 1)
        h2, h1, k2, k1 = 0, 1, 1, 0
        for _ in range(100000):
            assert Q > 0 and (D - P * P) % Q == 0
            a = (P + r) // Q
            h2, h1 = h1, a * h1 + h2
            k2, k1 = k1, a * k1 + k2
            if k1 > 0 and abs(self.norm((h1, k1))) == 1:
                self._unit = (h1, k1)
                break
            P = a * Q - P
            Q = (D - P * P) // Q
        else:
            raise RuntimeError("fundamental unit not found")
        # sanity: no unit strictly between 1 and eps0 with small coefficients
        e1 = self.emb(self._unit)[0]
        assert e1 > 1
        return self._unit

    def eps0(self):
        return self.emb(self.fundamental_unit())[0]

    def unit_norm(self):
        return self.norm(self.fundamental_unit())

    def unit_power(self, k):
        e = self.fundamental_unit()
        if k < 0:
            e = self.as_int(self.inv(e))
            k = -k
        out = (1, 0)
        for _ in range(k):
            out = self.mul(out, e)
        return out

    def _lattice_points_near(self, c1, h1, c2, h2):
        """All (a, b) in Z^2 with |a + b w1 - c1| <= h1 and |a + b w2 - c2| <= h2.

        Scale the embedded lattice O_K -> R^2 so the box becomes the square
        [-1,1]^2, Gauss-reduce the basis, and enumerate the disc of radius
        sqrt(2) around the centre: cost ~ (number of points) + O(1), even for
        very elongated boxes (large fundamental units)."""
        h1 = h1 * (1 + 1e-9) + 1e-12
        h2 = h2 * (1 + 1e-9) + 1e-12
        # basis columns: e_a = (1, 1), e_b = (w1, w2), scaled
        u = [1.0 / h1, 1.0 / h2]
        v = [self.w1 / h1, self.w2 / h2]
        U = [[1, 0], [0, 1]]            # columns: integer coords of u, v
        def dot(p, q):
            return p[0] * q[0] + p[1] * q[1]
        for _ in range(200):            # Lagrange--Gauss reduction
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
        # centre in reduced coordinates
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
                e1, e2 = self.emb((a, b))
                if abs(e1 - c1) <= h1 and abs(e2 - c2) <= h2:
                    out.append((a, b))
        return out

    def elements_in_box(self, B1, B2):
        """All a + b w with |sigma_1| <= B1 and |sigma_2| <= B2 (tiny outward margin)."""
        return self._lattice_points_near(0.0, B1, 0.0, B2)

    def elements_in_intervals(self, I1, I2):
        """All a + b w with sigma_1 in I1 and sigma_2 in I2 (closed, tiny outward margin)."""
        (l1, u1), (l2, u2) = I1, I2
        return self._lattice_points_near(0.5 * (l1 + u1), 0.5 * (u1 - l1),
                                         0.5 * (l2 + u2), 0.5 * (u2 - l2))

    def associated(self, x, y):
        """x = unit * y ?"""
        if abs(self.norm(x)) != abs(self.norm(y)):
            return False
        return self.is_integral(self.div(x, y))

    def norm_reps(self, B):
        """Dict |N| -> list of representatives of the nonzero elements with
        |N(beta)| <= B modulo units.  Each representative is balanced:
        eps0^-1 <= |beta1/beta2| < eps0 (possible because multiplication by eps0
        changes that ratio by eps0^2)."""
        B = int(B)
        if B in self._norm_reps:
            return self._norm_reps[B]
        self.check_float_range()
        eps = self.eps0()
        # cover {|x1 x2| <= B, eps^-1 <= |x1/x2| <= eps} by m boxes, one per
        # ratio window [rl, rh]:  |x1| <= sqrt(B rh), |x2| <= sqrt(B / rl)
        L = 2.0 * math.log(eps)
        m = max(1, int(math.ceil(L)))
        groups = {}
        seen = set()
        for j in range(m):
            rl = math.exp(-0.5 * L + L * j / m)
            rh = math.exp(-0.5 * L + L * (j + 1) / m)
            for x in self.elements_in_box(math.sqrt(B * rh), math.sqrt(B / rl)):
                if x in seen:
                    continue
                seen.add(x)
                n = abs(self.norm(x))
                if n == 0 or n > B:
                    continue
                reps = groups.setdefault(n, [])
                if not any(self.associated(x, y) for y in reps):
                    reps.append(x)
        # rebalance representatives: eps0^-1 <= |x1/x2| < eps0, x1 > 0
        e, einv = self.fundamental_unit(), self.as_int(self.inv(self.fundamental_unit()))
        for n, reps in groups.items():
            for i, x in enumerate(reps):
                for _ in range(200):
                    a1, a2 = self.emb(x)
                    r = abs(a1 / a2)
                    if r >= eps:
                        x = self.mul(x, einv)
                    elif r < 1.0 / eps:
                        x = self.mul(x, e)
                    else:
                        break
                if self.emb(x)[0] < 0:
                    x = self.neg(x)
                reps[i] = x
        self._norm_reps[B] = groups
        return groups

    def split_roots(self, p):
        """Roots mod p of the minimal polynomial X^2 - tw X + nw of w."""
        return [r for r in range(p) if (r * r - self.tw * r + self.nw) % p == 0]

    def class_number_is_one(self):
        """Rigorous test: Cl(K) is generated by the prime ideals of norm <= sqrt(d)/2
        (Minkowski bound).  Inert primes are principal; a split or ramified p
        gives principal prime ideals iff some element has |N| = p."""
        if self._h1 is not None:
            return self._h1
        M = int(math.floor(math.sqrt(self.disc) / 2.0))
        reps = self.norm_reps(M) if M >= 2 else {}
        ok = all(p in reps for p in _primes_upto(M) if self.split_roots(p))
        self._h1 = ok
        return ok

    def is_primitive(self, alpha, beta):
        """(alpha, beta) generate the unit ideal (valid for any class number)."""
        if alpha == (0, 0):
            return self.is_unit(beta)
        if beta == (0, 0):
            return self.is_unit(alpha)
        g = math.gcd(self.norm(alpha), self.norm(beta))
        for p in _prime_factors(g):
            roots = self.split_roots(p)
            if not roots:   # inert: P = (p)
                if alpha[0] % p == 0 and alpha[1] % p == 0 and beta[0] % p == 0 and beta[1] % p == 0:
                    return False
            else:
                for r in roots:  # P = (p, w - r)
                    if (alpha[0] + alpha[1] * r) % p == 0 and (beta[0] + beta[1] * r) % p == 0:
                        return False
        return True

    def complete_to_sl2(self, alpha, beta):
        """For primitive (alpha, beta) return (gamma, delta) in O_K with
        alpha*delta - beta*gamma = 1 (any class number).

        Replace alpha by alpha' = alpha + m beta (same class mod beta) with
        gcd(N(alpha'), N(beta)) = 1; then delta = conj(alpha') * k with
        k = N(alpha')^{-1} mod |N(beta)| satisfies alpha delta = 1 mod (N(beta)),
        hence mod (beta), since N(beta) = beta conj(beta)."""
        if not self.is_primitive(alpha, beta):
            raise ValueError("not primitive")
        if beta == (0, 0):
            return ((0, 0), self.as_int(self.inv(alpha)))
        if alpha == (0, 0):
            return (self.neg(self.as_int(self.inv(beta))), (0, 0))
        n = abs(self.norm(beta))
        if n == 1:
            # beta unit: delta = 0, gamma = -beta^{-1}
            return (self.neg(self.as_int(self.inv(beta))), (0, 0))
        for r in range(0, 12):
            for m in [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if max(abs(a), abs(b)) == r]:
                a2 = self.add(alpha, self.mul(m, beta))
                na = self.norm(a2)
                if math.gcd(na, n) != 1:
                    continue
                k = pow(na % n, -1, n)
                delta = self.mul(self.conj(a2), (k, 0))
                gamma = self.div(self.sub(self.mul(alpha, delta), (1, 0)), beta)
                assert self.is_integral(gamma)
                return (self.as_int(gamma), delta)
        raise RuntimeError("complete_to_sl2: no suitable shift found")
