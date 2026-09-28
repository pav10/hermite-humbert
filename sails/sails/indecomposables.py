"""Seeds for the sail computation and the indecomposables of a sail.

Every indecomposable of P_sigma lies (up to U^+) in a half-open parallelepiped
over a simplex of a triangulation of the sail, or is a vertex.  With the fine
triangulation (every facet lattice point is a vertex) the height s = c.x/d of such
a point satisfies s < n - 1, and

  s < 2   ==> indecomposable           (a decomposition needs two heights >= 1)
  s = 2   ==> decomposable iff x in W_F + W_F
  s > 2   ==> x decomposable iff x - iota*u in P_sigma for an indecomposable iota
              with N(iota) <= N(x)/2^n and a TP unit u  (superadditivity of N^(1/n)).

In degree 3 every candidate has s < 2, so no test is needed at all.
"""
import math
from fractions import Fraction

import numpy as np

from . import exact
from .field import pari
from .sail import parallelepiped, pp_point, unit_boxes


def small_norm_seeds(K, classes, bound=None):
    """Generators (of the right sign vector) of principal ideals of small norm.

    As in the notebook: norms up to floor(disc^(1/n)), doubled until every
    requested class has an element."""
    n = K.n
    B = bound or max(2, int(math.floor(abs(K.disc) ** (1.0 / n) + 1e-9)))
    seeds = {j: [] for j in classes}
    start = 1
    while True:
        for N in range(start, B + 1):
            for sN in ((N,) if n % 2 else (N, -N)):
                for g in pari.bnfisintnorm(K.bnf, sN):
                    x = tuple(int(t) for t in pari.nfalgtobasis(K.nf, g))
                    j, y = K.to_signature_class(x)
                    if j in seeds:
                        seeds[j].append(y)
        if all(seeds[j] for j in classes):
            return seeds, B
        start, B = B + 1, 2 * B


def compute_indecomposables(O):
    """Indecomposables of the orthant O (after O.compute_sail); sets O.indecomposables
    to a list of (norm, x) sorted by (norm, x), x canonical modulo U^+."""
    K, n = O.K, O.n
    O.finish_facets()
    found = {}                      # canonical representative -> (|N|, log embedding)
    pending = {}                    # height > 2: raw point (facet frame) -> |N|
    n_cand = 0

    def accept(x, lg=None):
        y, lgy = O.canon_point(x, lg)
        if y not in found:
            found[y] = (abs(K.norm(y)), lgy)

    for F in O.facets.values():
        for w in F.lattice_points:
            n_cand += 1
            accept(w)
        sumset = None
        for simplex in F.fine:
            M, R, ad = parallelepiped(simplex)
            for r in R[R.sum(axis=1) > 0]:
                n_cand += 1
                x = pp_point(M, r, ad)
                s = Fraction(exact.dot(F.c, x), F.d)
                if s >= n - 1 and n > 2:
                    raise ArithmeticError('height >= n-1 over a fine triangulation')
                if s < 2:
                    accept(x)
                elif s == 2:
                    if sumset is None:
                        W = F.lattice_points
                        sumset = {tuple(a + b for a, b in zip(u, v)) for u in W for v in W}
                    if x not in sumset:
                        accept(x)
                else:
                    pending[x] = abs(K.norm(x))
    O.stats['candidates'] = n_cand
    O.stats['tested'] = len(pending)
    ind = sorted(((N, y, lg) for y, (N, lg) in found.items()), key=lambda t: (t[0], t[1]))
    for x in sorted(pending, key=lambda x: (pending[x], x)):
        N, lg = pending[x], K.log_embed(x)
        if not is_decomposable(O, x, lg, N, ind):
            y, lgy = O.canon_point(x, lg)
            if y not in found:
                found[y] = (N, lgy)
                ind.append((N, y, lgy))
                ind.sort(key=lambda t: (t[0], t[1]))
    O.indecomposables = [(N, y) for N, y, _ in ind]
    if n == 3:          # Kala-Man: #Ind = sum_F (d_F - 1/2) vol_F  (self-check)
        if 2 * len(ind) != sum((2 * F.d - 1) * F.volume for F in O.facets.values()):
            raise ArithmeticError('Kala-Man identity violated')
    return O.indecomposables


def is_decomposable(O, alpha, lg_alpha, N_alpha, ind):
    """Is alpha - iota*u in P_sigma for some iota in ind (sorted by norm), u in U^+?"""
    K, n = O.K, O.n
    bound = Fraction(N_alpha, 2 ** n)
    for N_i, x_i, lg_i in ind:
        if N_i > bound:
            break
        b = lg_alpha - lg_i
        _, G = unit_boxes(K, b[None, :])
        if len(G) == 0:
            continue
        diff = b - G @ K.L
        maybe = (diff > -1e-9).all(axis=1)
        for k in G[maybe]:
            z = tuple(p - q for p, q in zip(alpha, K.apply_unit(k, x_i)))
            if any(z) and K.signs(z) == O.sig:
                return True
    return False
