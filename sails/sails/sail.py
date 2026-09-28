"""Sails of totally real fields modulo totally positive units.

For a sign vector sigma, P_sigma = {x in O_K \\ 0 : sgn sigma_i(x) = sigma_i}; its
sail is the boundary of conv(P_sigma).  A facet is stored as (c, d): c a primitive
integer vector (dual coordinates of its normal delta, so Tr(delta x) = c.x) and
d = min{c.x : x in P_sigma} its integer distance.

Algorithm (all decisions exact):

1. Orbit hull.  For a finite set S of points of P_sigma, the hull of the orbit
   U^+ S is computed modulo U^+ by gift wrapping: start from one facet, rotate the
   supporting hyperplane about every ridge of every facet orbit, canonicalise the
   new facet modulo U^+, repeat until closed.  Candidate points q = u s with
   c.q <= D are enumerated in the unit lattice (a small simplex in log space),
   exact integer arithmetic decides.
2. Refinement.  The half-open parallelepipeds over a triangulation of the orbit
   hull contain every lattice point lying strictly below it.  They are added to S
   and step 1 is repeated until no lattice point lies below: the orbit hull is then
   the sail of P_sigma.
"""
import math
from collections import deque
from fractions import Fraction
from itertools import product

import numpy as np

from . import exact
from .field import pari

AMBIG = 1e-6          # reduced coordinates this close to an integer are ambiguous
MAX_REFINE = 60
MAX_FACETS = 200000


class SailError(RuntimeError):
    pass


def size_key(v):
    return (sum(t * t for t in v), v)


# ----------------------------------------------------------------------------
# lattice points of half-open parallelepipeds

def parallelepiped(V):
    """Lattice points of {sum lambda_j V_j : 0 <= lambda_j < 1}.

    Returns (M, R, ad): M the matrix with columns V_j, ad = |det M|, and R an int64
    array whose rows r give the points (M r)/ad with lambda = r/ad.
    """
    n = len(V)
    M = tuple(tuple(V[j][i] for j in range(n)) for i in range(n))
    D = exact.det(M)
    ad = abs(D)
    if ad == 0:
        raise SailError('degenerate simplex')
    if ad == 1:
        return M, np.zeros((1, n), dtype=np.int64), 1
    snf = pari.matsnf(pari.matrix(n, n, [x for row in M for x in row]), 1)
    U, Dm = snf[0], snf[2]
    Uinv = U ** -1
    Uinv = [[int(Uinv[i, j]) for j in range(n)] for i in range(n)]
    diag = [abs(int(Dm[i, i])) for i in range(n)]
    adj = exact.adjugate(M)
    sgn = 1 if D > 0 else -1
    A = [[sgn * sum(adj[i][t] * Uinv[t][j] for t in range(n)) % ad for j in range(n)]
         for i in range(n)]
    dims = [i for i in range(n) if diag[i] > 1]
    ranges = [np.arange(diag[i], dtype=np.int64) for i in dims]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), -1).reshape(-1, len(dims))
    Am = np.array([[A[i][j] for j in dims] for i in range(n)], dtype=np.int64)
    R = (grid @ Am.T) % ad
    return M, R, ad


def unit_boxes(K, B):
    """For each row b of B, all integer k in the bounding box of the simplex
    {k : (kL)_i <= b_i for all i}.  Returns (row index array, k array)."""
    n = K.n
    V = np.stack([B[:, K.keep_cols[j]] @ K.Lsub_inv[j] for j in range(n)])   # n x m x (n-1)
    lo = np.ceil(V.min(axis=0) - 1e-6).astype(np.int64)
    hi = np.floor(V.max(axis=0) + 1e-6).astype(np.int64)
    sizes = np.maximum(hi - lo + 1, 0)
    counts = sizes.prod(axis=1)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, n - 1), dtype=np.int64)
    idx = np.repeat(np.arange(len(B)), counts)
    rem = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    ks = np.empty((total, n - 1), dtype=np.int64)
    for d in range(n - 1):
        s = sizes[idx, d]
        ks[:, d] = lo[idx, d] + rem % s
        rem //= s
    return idx, ks


def pp_point(M, r, ad):
    n = len(M)
    return tuple(sum(M[i][j] * int(r[j]) for j in range(n)) // ad for i in range(n))


# ----------------------------------------------------------------------------

class Reps:
    """Canonical representatives of points of P_sigma modulo U^+ with their logs."""

    def __init__(self, K, items):
        self.x = [p for p, _ in items]
        self.logs = np.array([lg for _, lg in items]).reshape(len(items), K.n)
        self.t = np.array([K.reduced_coords(lg) for lg in self.logs]).reshape(len(items), K.n - 1)

    def __len__(self):
        return len(self.x)


class Ridge:
    __slots__ = ('h', 'e', 'members', 'center_log', 'neighbor')

    def __init__(self, h, e, members, center_log):
        self.h, self.e, self.members, self.center_log = h, e, members, center_log
        self.neighbor = None


class Facet:
    """A facet (c, d) with its orbit points W on the hyperplane c.x = d."""

    def __init__(self, c, d, W):
        self.c, self.d, self.W = c, d, W
        self.ridges = []
        self.vertices = []
        self.j0 = None
        self.proj = None
        self._coarse = None
        self._coarse_pp = None
        self.lattice_points = None      # all lattice points of P on the facet (final sail)
        self.fine = None
        self.volume = None              # normalized (n-1)-volume in the facet lattice

    def coarse_simplices(self):
        """Triangulation (pulling) using vertices only, as tuples of points."""
        if self._coarse is None:
            idx = exact.triangulate(self.proj)
            self._coarse = [tuple(self.W[i] for i in s) for s in idx]
        return self._coarse

    def coarse_parallelepipeds(self):
        if self._coarse_pp is None:
            self._coarse_pp = [parallelepiped(s) for s in self.coarse_simplices()]
        return self._coarse_pp


class Orthant:
    """Sail and indecomposables of one sail signature class."""

    def __init__(self, K, index):
        self.K = K
        self.n = K.n
        self.index = index
        self.sig = K.sail_signatures[index]
        self.sg = np.array([-1.0 if s else 1.0 for s in self.sig])
        self.facets = None
        self.indecomposables = None
        self.stats = {'enumerations': 0, 'rotations': 0, 'fallback_rotations': 0,
                      'refinements': 0}
        n = self.n
        self._offsets = np.array(list(product((-1, 0, 1), repeat=n - 1)), dtype=np.int64)

    # --- canonical forms modulo U^+ ---------------------------------------

    def _shifts(self, t):
        """Shifts k with t + k in [-1/2, 1/2), all of them near the boundary."""
        k0 = [-math.floor(tj + 0.5) for tj in t]
        choices = []
        for j, tj in enumerate(t):
            f = tj + k0[j]
            if f < -0.5 + AMBIG:
                choices.append((k0[j], k0[j] + 1))
            elif f > 0.5 - AMBIG:
                choices.append((k0[j], k0[j] - 1))
            else:
                choices.append((k0[j],))
        return list(product(*choices))

    def canon_point(self, x, lg=None):
        """(canonical representative of x U^+, its log embedding).

        The representative minimises T2(y) = Tr(y^2) (exact integer) over the orbit,
        ties broken by the integral coordinates."""
        K, n = self.K, self.n
        if lg is None:
            lg = K.log_embed(x)
        k0 = -np.rint(K.reduced_coords(lg))
        with np.errstate(over='ignore'):
            B = float(np.exp(2 * (lg + k0 @ K.L)).sum()) * (1 + 1e-9)
            _, G = unit_boxes(K, (0.5 * math.log(B) - lg)[None, :])
            vals = np.exp(2 * (lg + G @ K.L)).sum(axis=1)
        best = None
        for k in G[vals <= vals.min() * (1 + 1e-9)]:
            y = K.apply_unit(k, x)
            key = (K.t2(y), y)
            if best is None or key < best[0]:
                best = (key, y, k)
        _, y, k = best
        return y, lg + np.array(k, dtype=float) @ K.L

    def canon_facet(self, c):
        """(canonical translate of the facet functional c, the unit shift k used)."""
        K = self.K
        lg = -np.log(np.abs(K.dual_embed(c)))
        best = None
        for k in self._shifts(K.reduced_coords(lg)):
            c2 = K.apply_unit_dual(k, c)
            key = size_key(c2)
            if best is None or key < best[0]:
                best = (key, c2, k)
        return best[1], best[2]

    def valid(self, c):
        a = self.K.dual_embed(c)
        return bool(np.all(a * self.sg > 0))

    # --- enumeration of orbit points below a functional ---------------------

    def orbit_below(self, c, D, reps):
        """{q = u s : s in reps, u in U^+, c.q <= D} as a dict q -> index of s."""
        K, n = self.K, self.n
        self.stats['enumerations'] += 1
        a = K.dual_embed(c)
        if np.any(a * self.sg <= 0):
            raise SailError('functional is not of the orthant signature')
        la = np.log(np.abs(a))
        logD = math.log(D)
        W = reps.logs + la
        ok = np.nonzero(W.mean(axis=1) <= math.log(D / n) + 1e-7)[0]
        found = {}
        if len(ok) == 0:
            return found
        W = W[ok]
        idx, ks = unit_boxes(K, logD - W)          # k with (kL)_i <= logD - w_i for all i
        if len(idx) == 0:
            return found
        with np.errstate(over='ignore'):
            vals = np.exp(W[idx] + ks @ K.L).sum(axis=1)
        sel = np.nonzero(vals <= D * (1 + 1e-7))[0]
        for t in sel:
            i = ok[idx[t]]
            q = K.apply_unit(ks[t], reps.x[i])
            if exact.dot(c, q) <= D:
                found[q] = i
        return found

    # --- rotation of a supporting hyperplane --------------------------------

    def _normalise_direction(self, c, d, h, e):
        """Replace (h, e) by (h - lam c, e - lam d) so that the rotation leaves
        the valid cone at a finite parameter."""
        a = self.K.dual_embed(c) * self.sg
        b = self.K.dual_embed(h) * self.sg
        lam = math.floor(float(np.min(b / a))) + 1
        if lam > 0:
            h = tuple(hi - lam * ci for hi, ci in zip(h, c))
            e = e - lam * d
        return h, e

    def _hint_ratios(self, c, d, h, e, reps, center_log, keep=4):
        """Exact ratios for a few orbit points near center_log (upper bounds for s*)."""
        K = self.K
        a = np.abs(K.dual_embed(c))
        b = K.dual_embed(h) * self.sg
        tR = K.reduced_coords(center_log)
        k0 = np.rint(tR[None, :] - reps.t).astype(np.int64)             # |S| x (n-1)
        kk = k0[:, None, :] + self._offsets[None, :, :]                  # |S| x 3^(n-1) x (n-1)
        with np.errstate(over='ignore', invalid='ignore'):
            ex = np.exp(reps.logs[:, None, :] + kk @ K.L)
            cq = ex @ a
            hq = ex @ b
            den = e - hq
            scale = np.abs(ex) @ np.abs(b) + abs(e) + 1.0
            ok = den > 1e-9 * scale
            ratio = np.where(ok, (cq - d) / np.where(ok, den, 1.0), np.inf)
        flat = np.argsort(ratio, axis=None)[:keep]
        out = []
        for f in flat:
            i, j = np.unravel_index(f, ratio.shape)
            if not np.isfinite(ratio[i, j]):
                break
            q = K.apply_unit(kk[i, j], reps.x[i])
            hq_ = exact.dot(h, q)
            if hq_ < e:
                out.append(Fraction(exact.dot(c, q) - d, e - hq_))
        return out

    def rotate(self, c, d, h, e, reps, center_log):
        """Smallest s > 0 with (c + s h, d + s e) supporting U^+ reps.

        Precondition: (c, d) supports, and h.q >= e for every q with c.q = d.
        Returns (s, argmin points)."""
        K = self.K
        self.stats['rotations'] += 1
        h, e = self._normalise_direction(c, d, h, e)
        a = K.dual_embed(c) * self.sg
        b = K.dual_embed(h) * self.sg
        neg = b < 0
        s_max = float(np.min(a[neg] / -b[neg]))
        hints = [s for s in self._hint_ratios(c, d, h, e, reps, center_log)
                 if float(s) < s_max * (1 - 1e-9)]

        def tries():
            if hints:
                yield min(hints)
            for j in range(1, 60):
                yield Fraction(s_max * (1 - 2.0 ** -j)).limit_denominator(2 ** 40)

        for attempt, s_try in enumerate(tries()):
            if attempt == 1:
                self.stats['fallback_rotations'] += 1
            if s_try <= 0:
                continue
            num, den = s_try.numerator, s_try.denominator
            c2 = tuple(den * ci + num * hi for ci, hi in zip(c, h))
            D2 = den * d + num * e
            if not self.valid(c2):
                continue
            best, arg = None, []
            for q in self.orbit_below(c2, D2, reps):
                hq = exact.dot(h, q)
                if hq < e:
                    r = Fraction(exact.dot(c, q) - d, e - hq)
                    if best is None or r < best:
                        best, arg = r, [q]
                    elif r == best:
                        arg.append(q)
            if best is not None:
                return best, h, e, arg
        raise SailError('rotation failed')

    @staticmethod
    def _combine(c, d, h, e, s):
        num, den = s.numerator, s.denominator
        v = tuple(den * ci + num * hi for ci, hi in zip(c, h)) + (den * d + num * e,)
        v = exact.primitive(v)
        return v[:-1], v[-1]

    # --- facets ---------------------------------------------------------------

    def initial_facet(self, reps):
        K, n = self.K, self.n
        s0 = reps.x[0]
        c = exact.primitive(exact.matvec(K.T, s0))           # Tr(s0 x)
        D = min(exact.dot(c, s) for s in reps.x)
        pts = list(self.orbit_below(c, D, reps))
        d = min(exact.dot(c, q) for q in pts)
        touch = [q for q in pts if exact.dot(c, q) == d]
        while exact.affine_rank(touch) < n - 1:
            cd = tuple(c) + (d,)
            ker = [v for v in exact.kernel([tuple(t) + (-1,) for t in touch], n + 1)
                   if exact.rank([v, cd], n + 1) == 2]
            v = min(ker, key=lambda v: sum(abs(t) for t in v))
            center = np.mean([K.log_embed(t) for t in touch], axis=0)
            s, h, e, arg = self.rotate(c, d, v[:n], v[n], reps, center)
            c, d = self._combine(c, d, h, e, s)
            touch = list(dict.fromkeys(touch + arg))
        return c, d

    def make_facet(self, c, d, reps):
        K, n = self.K, self.n
        pts = self.orbit_below(c, d, reps)
        W = sorted(pts)
        if any(exact.dot(c, q) != d for q in W):
            raise SailError('hyperplane is not supporting')
        F = Facet(c, d, W)
        nz = [j for j in range(n) if c[j] != 0]
        F.j0 = j0 = min(nz, key=lambda j: (abs(c[j]), j))
        F.proj = [exact.drop(w, j0) for w in W]
        if exact.affine_rank(F.proj) != n - 1:
            raise SailError('facet of wrong dimension')
        hf = exact.hull_facets(F.proj) if n > 1 else []
        F.vertices = [W[i] for i in exact.vertex_indices(F.proj, hf)]
        logs = {w: K.log_embed(w) for w in W}
        for a, b, S in hf:
            h = a[:j0] + (0,) + a[j0:]
            members = sorted(S)
            center = np.mean([logs[W[i]] for i in members], axis=0)
            F.ridges.append(Ridge(h, b, members, center))
        return F

    def orbit_hull(self, reps):
        """Facet orbits of conv(U^+ reps), as a dict canonical c -> Facet."""
        K = self.K
        c0, d0 = self.initial_facet(reps)
        c0, _ = self.canon_facet(c0)
        facets = {}
        known = {}           # canonical c -> ridges (point sets) whose neighbour is known
        queue = deque([(c0, d0)])
        while queue:
            c, d = queue.popleft()
            if c in facets:
                continue
            if len(facets) >= MAX_FACETS:
                raise SailError('too many facet orbits')
            F = self.make_facet(c, d, reps)
            facets[c] = F
            done = known.get(c, ())
            for R in F.ridges:
                members = frozenset(F.W[i] for i in R.members)
                if members in done:
                    self.stats['ridges_skipped'] = self.stats.get('ridges_skipped', 0) + 1
                    continue
                s, h, e, _ = self.rotate(c, d, R.h, R.e, reps, R.center_log)
                c2, d2 = self._combine(c, d, h, e, s)
                c2, k = self.canon_facet(c2)
                R.neighbor = c2
                if c2 not in facets:
                    known.setdefault(c2, set()).add(frozenset(K.apply_unit(k, p) for p in members))
                    queue.append((c2, d2))
        return facets

    # --- the sail -----------------------------------------------------------

    def compute_sail(self, seeds):
        """seeds: points of P_sigma.  Sets self.facets to the sail's facet orbits."""
        items = {}
        for x in seeds:
            y, lg = self.canon_point(x)
            items[y] = lg
        for it in range(MAX_REFINE):
            reps = Reps(self.K, sorted(items.items(), key=lambda kv: size_key(kv[0])))
            facets = self.orbit_hull(reps)
            below = {}
            for F in facets.values():
                for M, R, ad in F.coarse_parallelepipeds():
                    lev = R.sum(axis=1)
                    for r in R[(lev > 0) & (lev < ad)]:
                        y, lg = self.canon_point(pp_point(M, r, ad))
                        below[y] = lg
            if not below:
                self.facets = self._dedupe_facets(facets)
                self.stats['refinements'] = it
                self.reps = reps
                return self.facets
            items = {}
            for F in facets.values():
                for v in F.vertices:
                    y, lg = self.canon_point(v)
                    items[y] = lg
            items.update(below)
        raise SailError('refinement did not converge')

    def _dedupe_facets(self, facets):
        """Exact safety net for the floating point canonical form: merge facets that
        are U^+-translates of each other (c' = M_u^-T c)."""
        K = self.K
        groups = {}
        for c, F in facets.items():
            groups.setdefault((F.d, len(F.W)), []).append(F)
        keep = {}
        for group in groups.values():
            reps_ = []
            for F in group:
                lg = np.log(np.abs(K.dual_embed(F.c)))
                dup = False
                for G, lgG in reps_:
                    k = np.rint(K.reduced_coords(lg - lgG)).astype(np.int64)
                    if K.apply_unit_dual(k, F.c) == G.c:
                        dup = True
                        break
                if dup:
                    self.stats['duplicate_facets'] = self.stats.get('duplicate_facets', 0) + 1
                else:
                    reps_.append((F, lg))
                    keep[F.c] = F
        return keep

    # --- facet lattice points and fine triangulation -------------------------

    def finish_facets(self):
        """Lattice points on each facet, fine triangulation, integer volume."""
        n = self.n
        for F in self.facets.values():
            pts = set(F.vertices)
            vol = 0
            for M, R, ad in F.coarse_parallelepipeds():
                vol += ad
                lev = R.sum(axis=1)
                for r in R[lev == ad]:
                    pts.add(pp_point(M, r, ad))
            if vol % F.d:
                raise SailError('volume not divisible by distance')
            F.volume = vol // F.d
            F.lattice_points = sorted(pts)
            F.fine = fine_triangulation(F, F.lattice_points)
            fine_vol = sum(abs(exact.det(tuple(tuple(s[j][i] for j in range(n)) for i in range(n))))
                           for s in F.fine)
            if fine_vol != vol:
                raise SailError('fine triangulation has wrong volume')


def fine_triangulation(F, points):
    """Triangulation of facet F using every lattice point as a vertex
    (stellar subdivisions of the pulling triangulation; exact)."""
    j0 = F.j0
    proj = {p: exact.drop(p, j0) + (1,) for p in points}
    simplices = [tuple(s) for s in F.coarse_simplices()]
    verts = set(v for s in simplices for v in s)
    adj = {}
    for p in points:
        if p in verts:
            continue
        pp = proj[p]
        new = []
        for s in simplices:
            if s not in adj:
                A = tuple(tuple(proj[v][i] for v in s) for i in range(len(pp)))
                sgn = 1 if exact.det(A) > 0 else -1
                adj[s] = tuple(tuple(sgn * x for x in row) for row in exact.adjugate(A))
            nu = exact.matvec(adj[s], pp)            # positive multiple of the barycentrics
            if min(nu) < 0:
                new.append(s)
                continue
            for i in range(len(s)):
                if nu[i] > 0:
                    new.append(s[:i] + (p,) + s[i + 1:])
        simplices = new
        verts.add(p)
    return simplices
