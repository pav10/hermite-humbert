"""Number field data for sail computations (pure Python + cypari2).

Conventions
-----------
* Elements are integer column vectors in the PARI integral basis nf.zk
  ("integral basis coordinates", IBC); omega_1 = 1.
* Real embeddings sigma_1 < ... < sigma_n are ordered by the value of the root
  of the defining polynomial, as in Sage's ``K.embeddings(RR)``.
* A linear functional x -> Tr(delta x) with delta in the codifferent is an
  integer vector c (dual basis coordinates of delta): Tr(delta x) = c.x.
* Signs are GF(2) vectors: 0 for positive, 1 for negative (as in the notebook).

Exact decisions (signs, norms, comparisons) use integers or PARI; floating point
is only used to locate candidates.
"""
import math
from fractions import Fraction
from itertools import product

import numpy as np
import mpmath
import cypari2

from . import exact

pari = cypari2.Pari()
try:
    pari.allocatemem(2 ** 28, 2 ** 32, silent=True)
except TypeError:  # older cypari2
    pari.allocatemem(2 ** 28, 2 ** 32)
except Exception:
    pass

HP_DIGITS = 100          # precision of the high precision embedding matrix
_MEMBERS = {}


def member(obj, name):
    f = _MEMBERS.get(name)
    if f is None:
        f = _MEMBERS[name] = pari('(o)->o.%s' % name)
    return f(obj)


def to_frac(g):
    return Fraction(int(pari.numerator(g)), int(pari.denominator(g)))


def col(x):
    return pari.Col([int(t) for t in x])


def _mpf(g):
    """PARI real/rational -> mpmath (PARI prints exponents as ' E-30')."""
    return mpmath.mpf(str(g).replace(' E', 'e'))


# ----------------------------------------------------------------------------
# GF(2) helpers (vectors are tuples of 0/1)

def gf2_rank(vectors):
    piv = {}
    for v in vectors:
        x = sum(b << i for i, b in enumerate(v))
        while x:
            h = x.bit_length() - 1
            if h in piv:
                x ^= piv[h]
            else:
                piv[h] = x
                break
    return len(piv)


def gf2_rref(vectors, length):
    rows = [list(v) for v in vectors]
    r = 0
    for c in range(length):
        piv = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[r])]
        r += 1
    return [tuple(row) for row in rows[:r]]


def gf2_left_kernel(rows, ncols):
    """Echelonized basis of {v : v.A = 0} over GF(2) (as Sage's left_kernel().basis())."""
    m = len(rows)
    At = [tuple(rows[i][j] for i in range(m)) for j in range(ncols)]   # ncols x m
    R = gf2_rref(At, m)
    pivots = [next(c for c in range(m) if row[c]) for row in R]
    basis = []
    for f in range(m):
        if f in pivots:
            continue
        v = [0] * m
        v[f] = 1
        for row, pc in zip(R, pivots):
            v[pc] = row[f]
        basis.append(tuple(v))
    return gf2_rref(basis, m)


def sage_tuples01(k):
    """Tuples([0,1], k) in Sage's order (first coordinate varies fastest)."""
    return [tuple(reversed(p)) for p in product((0, 1), repeat=k)]


def gf2_add(u, v):
    return tuple(a ^ b for a, b in zip(u, v))


# ----------------------------------------------------------------------------
# printing in Sage's format

def sage_poly_str(coeffs, var='a'):
    """Sage-style string of sum coeffs[i] var^i (coeffs are Fractions)."""
    terms = []
    for i in reversed(range(len(coeffs))):
        c = Fraction(coeffs[i])
        if c == 0:
            continue
        mon = '' if i == 0 else (var if i == 1 else '%s^%d' % (var, i))
        neg = c < 0
        a = -c if neg else c
        if mon == '':
            body = str(a)
        elif a == 1:
            body = mon
        else:
            body = '%s*%s' % (a, mon)
        terms.append((neg, body))
    if not terms:
        return '0'
    s = ('-' if terms[0][0] else '') + terms[0][1]
    for neg, body in terms[1:]:
        s += (' - ' if neg else ' + ') + body
    return s


def sage_vector_str(v):
    return '(' + ', '.join(str(Fraction(x)) for x in v) + ')'


# ----------------------------------------------------------------------------

class NumberFieldData:
    """All data about a totally real field needed to compute its sails."""

    def __init__(self, coeffs, label=None, certify=True, lll_units=True):
        self.label = label
        self.coeffs = [int(c) for c in coeffs]
        n = self.n = len(self.coeffs) - 1
        if n < 1 or self.coeffs[-1] != 1:
            raise ValueError('need a monic polynomial of degree >= 1')
        self.pol = pari.Pol(list(reversed(self.coeffs)))
        if n == 1:
            raise ValueError('degree 1 is not supported')
        bnf = self.bnf = pari.bnfinit(self.pol, 1)
        self.certified = False
        if certify:
            if int(pari.bnfcertify(bnf)) != 1:
                raise ArithmeticError('bnfcertify failed')
            self.certified = True
        nf = self.nf = member(bnf, 'nf')
        r1, r2 = [int(t) for t in member(nf, 'sign')]
        if r1 != n:
            raise ValueError('field is not totally real')
        self.disc = int(member(nf, 'disc'))
        self.class_number = int(member(bnf, 'no'))
        self.regulator = float(member(bnf, 'reg'))

        # integral basis in power basis coordinates: rows of B
        zk = member(nf, 'zk')
        self.zk = zk
        self.B = [[to_frac(t) for t in pari.Vecrev(zk[j], n)] for j in range(n)]
        self.monogenic_index = abs(int(member(nf, 'index')))

        # multiplication table and trace matrix
        I = pari.matid(n)
        self._e = [I[j] for j in range(n)]
        self.mult = [[None] * n for _ in range(n)]
        for j in range(n):
            for k in range(j, n):
                v = pari.nfalgtobasis(nf, pari.nfeltmul(nf, self._e[j], self._e[k]))
                self.mult[j][k] = self.mult[k][j] = tuple(int(t) for t in v)
        tr = [int(pari.nfelttrace(nf, self._e[i])) for i in range(n)]
        self.T = tuple(tuple(sum(self.mult[j][k][i] * tr[i] for i in range(n))
                             for k in range(n)) for j in range(n))

        # embeddings, sorted by the value of the root
        old_prec = pari.get_real_precision() if hasattr(pari, 'get_real_precision') else None
        pari.set_real_precision(HP_DIGITS + 20)      # digits printed by str()
        bits = int((HP_DIGITS + 30) * 3.33)
        roots = [r for r in pari.polrootsreal(self.pol, precision=bits)]
        roots.sort(key=lambda r: float(r))
        mpmath.mp.dps = HP_DIGITS
        self.E_hp = [[_mpf(pari.subst(zk[j], 'x', r)) for j in range(n)] for r in roots]
        nfroots = member(nf, 'roots')
        self.perm = []
        for r in roots:
            dists = [abs(float(nfroots[p]) - float(r)) for p in range(n)]
            self.perm.append(min(range(n), key=lambda p: dists[p]))
        if sorted(self.perm) != list(range(n)):
            raise ArithmeticError('could not match real places')
        if old_prec is not None:
            pari.set_real_precision(old_prec)
        self.E = np.array([[float(x) for x in row] for row in self.E_hp])
        self.Eabs = np.abs(self.E)
        Einv = np.linalg.inv(self.E)
        self.EinvT = Einv.T.copy()
        self.EinvTabs = np.abs(self.EinvT)
        Tinv_hp = mpmath.matrix([[mpmath.mpf(x) for x in row] for row in self.T]) ** -1
        self.EinvT_hp = mpmath.matrix(self.E_hp) * Tinv_hp   # sigma(delta_c) = E T^-1 c
        # 80-bit extended precision copies (an intermediate tier before mpmath)
        self.E_ld = np.array([[np.longdouble(mpmath.nstr(x, 30)) for x in row] for row in self.E_hp])
        self.EinvT_ld = np.array([[np.longdouble(mpmath.nstr(self.EinvT_hp[i, j], 30)) for j in range(n)]
                                  for i in range(n)])
        self._ld_ok = np.finfo(np.longdouble).eps < 1e-18

        # units: gens = [-1, fu_1, ..., fu_{n-1}] as in Sage's gens_values()
        fu = member(bnf, 'fu')
        self.fu = [tuple(int(t) for t in pari.nfalgtobasis(nf, u)) for u in fu]
        minus_one = tuple([-1] + [0] * (n - 1))
        self.unit_gens = [minus_one] + self.fu
        self.unit_gen_signs = [self.signs(u) for u in self.unit_gens]

        self.sail_signatures = self._sail_signatures()
        self.tpfu_gordon = self._tpfu_gordon()
        self.tpfu = self._lll_units(self.tpfu_gordon) if lll_units else list(self.tpfu_gordon)
        self._setup_unit_action()
        self._setup_signature_classes()

    # --- basic arithmetic -------------------------------------------------

    def mul(self, x, y):
        n = self.n
        out = [0] * n
        for j, xj in enumerate(x):
            if xj:
                for k, yk in enumerate(y):
                    if yk:
                        m = self.mult[j][k]
                        p = xj * yk
                        for i in range(n):
                            out[i] += p * m[i]
        return tuple(out)

    def mult_matrix(self, x):
        """Integer matrix M with coords(x*y) = M coords(y)."""
        n = self.n
        cols = [self.mul(x, tuple(int(i == k) for i in range(n))) for k in range(n)]
        return exact.transpose(cols)

    def t2(self, x):
        """Tr(x^2) = x^T T x (exact)."""
        T = self.T
        return sum(x[j] * sum(T[j][k] * x[k] for k in range(self.n)) for j in range(self.n) if x[j])

    def norm(self, x):
        return int(pari.nfeltnorm(self.nf, col(x)))

    def power(self, x, e):
        return tuple(int(t) for t in pari.nfalgtobasis(self.nf, pari.nfeltpow(self.nf, col(x), e)))

    def signs(self, x):
        """Exact sign vector of x != 0 (0 = positive, 1 = negative)."""
        xf = np.array([float(t) for t in x])
        emb = self.E @ xf
        scale = self.Eabs @ np.abs(xf)
        if np.all(np.abs(emb) > 1e-9 * scale):
            return tuple(int(v < 0) for v in emb)
        s = pari.nfeltsign(self.nf, col(x))
        return tuple(int(int(s[self.perm[i]]) < 0) for i in range(self.n))

    def embed_hp(self, x):
        return [mpmath.fsum(e * t for e, t in zip(row, x)) for row in self.E_hp]

    def log_embed(self, x):
        """log|sigma_i(x)| (float), accurate to ~1e-10 relative."""
        xf = np.array([float(t) for t in x])
        emb = self.E @ xf
        scale = self.Eabs @ np.abs(xf)
        if np.all(np.abs(emb) > 1e-6 * scale):
            return np.log(np.abs(emb))
        if self._ld_ok and max(abs(t) for t in x) < 2 ** 63:
            emb = self.E_ld @ np.array(x, dtype=np.longdouble)
            if np.all(np.abs(emb) > 1e-8 * scale):
                return np.log(np.abs(emb)).astype(float)
        return np.array([float(mpmath.log(abs(v))) for v in self.embed_hp(x)])

    def dual_embed(self, c):
        """sigma(delta) for the functional c (float), accurate to ~1e-10 relative."""
        cf = np.array([float(t) for t in c])
        a = self.EinvT @ cf
        scale = self.EinvTabs @ np.abs(cf)
        if np.all(np.abs(a) > 1e-6 * scale):
            return a
        if self._ld_ok and max(abs(t) for t in c) < 2 ** 63:
            a = self.EinvT_ld @ np.array(c, dtype=np.longdouble)
            if np.all(np.abs(a) > 1e-8 * scale):
                return a.astype(float)
        v = self.EinvT_hp * mpmath.matrix([mpmath.mpf(int(t)) for t in c])
        return np.array([float(v[i]) for i in range(self.n)])

    def to_power_basis(self, x):
        n = self.n
        return [sum((x[j] * self.B[j][k] for j in range(n)), Fraction(0)) for k in range(n)]

    def elt_str(self, x):
        return sage_poly_str(self.to_power_basis(x))

    def vec_str(self, x):
        return sage_vector_str(self.to_power_basis(x))

    # --- signatures and totally positive units (as in the notebook) ------

    def _sail_signatures(self):
        """Representatives of GF(2)^n / sgn(units) (nf_sail_signature_finder)."""
        n = self.n
        unit_sigs = list(self.unit_gen_signs)
        basis = []
        for s in sage_tuples01(n):
            if gf2_rank(unit_sigs + basis + [s]) > gf2_rank(unit_sigs + basis):
                basis.append(s)
        out = []
        zero = tuple([0] * n)
        for coeffs in sage_tuples01(len(basis)):
            v = zero
            for c, b in zip(coeffs, basis):
                if c:
                    v = gf2_add(v, b)
            out.append(v)
        return out

    def _tpfu_gordon(self):
        """Totally positive fundamental units exactly as nf_tpfu_finder."""
        n = self.n
        ker = gf2_left_kernel(self.unit_gen_signs, n)
        signs = [v[0] for v in ker] + [0] * (n - 1 - len(ker))
        exps = [list(v[1:]) for v in ker]
        for j in range(n - 1):
            e = [0] * (n - 1)
            e[j] = 2
            if exact.rank(exps + [e], n - 1) > exact.rank(exps, n - 1):
                exps.append(e)
        units = []
        for i in range(n - 1):
            u = tuple([1] + [0] * (n - 1))
            for j in range(n - 1):
                if exps[i][j]:
                    u = self.mul(u, self.power(self.fu[j], exps[i][j]))
            if signs[i]:
                u = tuple(-t for t in u)
            units.append(u)
        for u in units:
            if any(self.signs(u)):
                raise ArithmeticError('TPFU is not totally positive')
        return units

    def _unit_logs(self, units):
        return np.array([self.log_embed(u) for u in units])

    def _lll_units(self, units):
        """LLL-reduce the TP unit lattice in log space (better conditioned domains)."""
        m = len(units)
        if m <= 1:
            return list(units)
        L = self._unit_logs(units)
        G = L @ L.T
        Gp = pari.matrix(m, m, [pari(repr(float(x))) for x in G.flatten()])
        U = pari.qflllgram(Gp)
        U = [[int(U[i, j]) for j in range(m)] for i in range(m)]
        new = []
        for j in range(m):
            u = tuple([1] + [0] * (self.n - 1))
            for i in range(m):
                if U[i][j]:
                    u = self.mul(u, self.power(units[i], U[i][j]))
            new.append(u)
        if abs(exact.det(U)) != 1:
            raise ArithmeticError('LLL transformation not unimodular')
        return new

    # --- action of the TP units ------------------------------------------

    def _setup_unit_action(self):
        n = self.n
        self.U = [self.mult_matrix(u) for u in self.tpfu]
        self.Uinv = [self.mult_matrix(self.power(u, -1)) for u in self.tpfu]
        self._pow_cache = {}
        self.L = self._unit_logs(self.tpfu)                     # (n-1) x n
        if n >= 2:
            self.Linv = np.linalg.inv(self.L[:, :n - 1])
            self.Lsub_inv = [np.linalg.inv(np.delete(self.L, j, axis=1)) for j in range(n)]
            self.keep_cols = [np.array([i for i in range(n) if i != j]) for j in range(n)]

    def unit_power_matrix(self, j, e):
        key = (j, e)
        M = self._pow_cache.get(key)
        if M is None:
            if e == 0:
                M = exact.identity(self.n)
            elif e > 0:
                M = exact.matmul(self.unit_power_matrix(j, e - 1), self.U[j])
            else:
                M = exact.matmul(self.unit_power_matrix(j, e + 1), self.Uinv[j])
            self._pow_cache[key] = M
        return M

    def apply_unit(self, k, x):
        """prod eps_j^k_j * x (exact)."""
        for j, e in enumerate(k):
            if e:
                x = exact.matvec(self.unit_power_matrix(j, int(e)), x)
        return x

    def apply_unit_dual(self, k, c):
        """Functional c transported by u = prod eps_j^k_j: c'.(u x) = c.x."""
        for j, e in enumerate(k):
            if e:
                M = self.unit_power_matrix(j, -int(e))
                c = tuple(sum(M[i][t] * c[i] for i in range(self.n)) for t in range(self.n))
        return c

    def unit_element(self, k):
        return self.apply_unit(k, tuple([1] + [0] * (self.n - 1)))

    def reduced_coords(self, logv):
        """Coordinates of the trace-zero part of a log vector in the TPFU log basis."""
        v = logv - logv.mean()
        return v[:self.n - 1] @ self.Linv

    # --- signature classes -------------------------------------------------

    def _setup_signature_classes(self):
        """For every sign vector tau: (class index j, unit u) with sgn(u) + tau = sig_j."""
        n = self.n
        units_modsq, sigs = [], []
        for u, s in zip(self.unit_gens, self.unit_gen_signs):
            if gf2_rank(sigs + [s]) > gf2_rank(sigs):
                units_modsq.append(u)
                sigs.append(s)
        combos = {}
        for coeffs in product((0, 1), repeat=len(units_modsq)):
            u = tuple([1] + [0] * (n - 1))
            s = tuple([0] * n)
            for c, v, sv in zip(coeffs, units_modsq, sigs):
                if c:
                    u = self.mul(u, v)
                    s = gf2_add(s, sv)
            combos.setdefault(s, u)
        self.sig_class = {}
        for tau in product((0, 1), repeat=n):
            for j, sj in enumerate(self.sail_signatures):
                need = gf2_add(tau, sj)
                if need in combos:
                    self.sig_class[tau] = (j, combos[need])
                    break
            else:
                raise ArithmeticError('signature %s not covered' % (tau,))

    def to_signature_class(self, x):
        """Return (j, x*u) with x*u of sign vector exactly sail_signatures[j]."""
        j, u = self.sig_class[self.signs(x)]
        return j, self.mul(x, u)

    # --- output helpers ------------------------------------------------------

    def poly_coeffs(self):
        return list(self.coeffs)
