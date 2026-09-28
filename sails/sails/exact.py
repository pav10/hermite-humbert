"""Exact integer linear algebra, small convex hulls and triangulations.

Everything here works with Python integers / Fractions only, so results are exact.
The point sets handled are small (points of one sail facet, dimension <= 4): brute
force over subsets for a few points, exact gift wrapping for more.
"""
from fractions import Fraction
from itertools import combinations
from math import gcd

GIFTWRAP_MIN_EXTRA = 4       # gift wrapping above m + 4 points, brute force below


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def primitive(v):
    g = 0
    for x in v:
        g = gcd(g, x)
    if g == 0:
        raise ValueError("zero vector has no primitive multiple")
    return tuple(x // g for x in v)


def matvec(M, x):
    return tuple(sum(a * b for a, b in zip(row, x)) for row in M)


def matmul(A, B):
    Bt = list(zip(*B))
    return tuple(tuple(sum(a * b for a, b in zip(row, col)) for col in Bt) for row in A)


def transpose(M):
    return tuple(tuple(r) for r in zip(*M))


def identity(n):
    return tuple(tuple(int(i == j) for j in range(n)) for i in range(n))


def det(M):
    """Determinant of a square integer matrix (Bareiss, fraction free)."""
    n = len(M)
    if n == 0:
        return 1
    A = [list(r) for r in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    break
            else:
                return 0
        akk = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]
            row_i, row_k = A[i], A[k]
            for j in range(k + 1, n):
                row_i[j] = (row_i[j] * akk - aik * row_k[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def adjugate(M):
    """Adjugate of a square integer matrix: adj(M) M = det(M) I."""
    n = len(M)
    if n == 1:
        return ((1,),)
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [row[:j] + row[j + 1:] for k, row in enumerate(M) if k != i]
            adj[j][i] = (-1) ** (i + j) * det(minor)
    return tuple(tuple(r) for r in adj)


def rref(rows, ncols):
    """Reduced row echelon form over Q; returns (rows as Fractions, pivot columns)."""
    A = [[Fraction(x) for x in r] for r in rows]
    pivots = []
    r = 0
    for c in range(ncols):
        if r == len(A):
            break
        piv = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        if pv != 1:
            A[r] = [x / pv for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        pivots.append(c)
        r += 1
    return A[:r], pivots


def rank(rows, ncols=None):
    """Rank of an integer matrix (fraction-free elimination)."""
    if not rows:
        return 0
    if ncols is None:
        ncols = len(rows[0])
    if any(isinstance(x, Fraction) for r in rows for x in r):
        return len(rref(rows, ncols)[1])
    A = [list(r) for r in rows]
    r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        g = A[r][c]
        for i in range(r + 1, len(A)):
            f = A[i][c]
            if f:
                row = [g * x - f * y for x, y in zip(A[i], A[r])]
                h = 0
                for x in row:
                    h = gcd(h, x)
                A[i] = [x // h for x in row] if h > 1 else row
        r += 1
        if r == len(A):
            break
    return r


def kernel(rows, ncols):
    """Primitive integer vectors spanning {x in Q^ncols : r.x = 0 for all rows r}."""
    R, pivots = rref(rows, ncols)
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for f in free:
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -R[i][f]
        den = 1
        for x in v:
            den = den * x.denominator // gcd(den, x.denominator)
        basis.append(primitive([int(x * den) for x in v]))
    return basis


def affine_rank(points):
    """Dimension of the affine hull of a list of integer points."""
    if len(points) <= 1:
        return 0
    p0 = points[0]
    return rank([tuple(a - b for a, b in zip(p, p0)) for p in points[1:]])


def hyperplane(points):
    """(a, b), a primitive, with a.p = b for the given points, or None if they do
    not span a unique hyperplane.  For m points in Z^m: a is the generalized cross
    product of the differences (integer minors)."""
    m = len(points[0])
    if len(points) != m:
        ker = kernel([tuple(p) + (-1,) for p in points], m + 1)
        if len(ker) != 1:
            return None
        v = ker[0]
        if not any(v[:m]):
            return None
        return tuple(v[:m]), v[m]
    p0 = points[0]
    D = [tuple(x - y for x, y in zip(p, p0)) for p in points[1:]]
    a = tuple((-1) ** j * det([row[:j] + row[j + 1:] for row in D]) for j in range(m))
    if not any(a):
        return None
    a = primitive(a)
    return a, dot(a, p0)


def drop(p, j):
    return p[:j] + p[j + 1:]


def _record(found, a, b, points):
    vals = [dot(a, p) - b for p in points]
    if all(v >= 0 for v in vals):
        pass
    elif all(v <= 0 for v in vals):
        a, b = tuple(-x for x in a), -b
        vals = [-v for v in vals]
    else:
        return False
    if (a, b) not in found:
        found[(a, b)] = frozenset(i for i, v in enumerate(vals) if v == 0)
    return True


def hull_facets(points):
    """Facets of conv(points) for integer points spanning Z^m affinely.

    Returns a list of (a, b, S): a primitive integer vector and b with a.p >= b for
    every point, equality exactly for the indices in S.
    """
    k, m = len(points), len(points[0])
    if m == 0:
        return []
    if m == 1:
        vals = [p[0] for p in points]
        lo, hi = min(vals), max(vals)
        return [((1,), lo, frozenset(i for i, v in enumerate(vals) if v == lo)),
                ((-1,), -hi, frozenset(i for i, v in enumerate(vals) if v == hi))]
    found = {}
    if k == m + 1:
        for i in range(k):
            others = [points[j] for j in range(k) if j != i]
            a, b = hyperplane(others)
            _record(found, a, b, points)
        return [(a, b, s) for (a, b), s in found.items()]
    if k > m + GIFTWRAP_MIN_EXTRA:
        return giftwrap_facets(points)
    covered = []
    for comb in combinations(range(k), m):
        cs = set(comb)
        if any(cs <= s for s in covered):
            continue
        hp = hyperplane([points[i] for i in comb])
        if hp is None:
            continue
        if _record(found, hp[0], hp[1], points):
            covered = list(found.values())
    return [(a, b, s) for (a, b), s in found.items()]


def _rotate(points, a, b, h, e):
    """Smallest s > 0 such that (a + s h, b + s e) still supports the points, and
    the new primitive functional.  (a, b) supports; points with h.p < e exist."""
    best_n = best_d = None
    for p in points:
        hp = dot(h, p)
        if hp < e:
            num, den = dot(a, p) - b, e - hp
            if best_n is None or num * best_d < best_n * den:
                best_n, best_d = num, den
    if best_n is None:
        return None
    v = primitive(tuple(best_d * x + best_n * y for x, y in zip(a, h)) + (best_d * b + best_n * e,))
    return v[:-1], v[-1]


def _normalise(points, a, b, h, e):
    """Replace (h, e) by (h - lam a, e - lam b) so that some point has h.p < e."""
    lam = None
    for p in points:
        u = dot(a, p) - b
        if u > 0:
            t = (dot(h, p) - e) // u + 1
            lam = t if lam is None else min(lam, t)
    if lam is not None and lam > 0:
        h = tuple(x - lam * y for x, y in zip(h, a))
        e = e - lam * b
    return h, e


def giftwrap_facets(points):
    """Exact gift wrapping (same output as hull_facets) for larger point sets."""
    k, m = len(points), len(points[0])
    if m <= 1:
        return hull_facets(points)
    a = (1,) + (0,) * (m - 1)
    b = min(p[0] for p in points)
    touch = [p for p in points if p[0] == b]
    while affine_rank(touch) < m - 1:
        ab = a + (b,)
        ker = [v for v in kernel([tuple(t) + (-1,) for t in touch], m + 1)
               if rank([v, ab], m + 1) == 2]
        h, e = ker[0][:m], ker[0][m]
        if not any(dot(h, p) < e for p in points):
            h, e = tuple(-x for x in h), -e
        a, b = _rotate(points, a, b, h, e)
        touch = [p for p in points if dot(a, p) == b]
    found = {}
    queue = [(a, b)]
    while queue:
        a, b = queue.pop()
        if (a, b) in found:
            continue
        S = frozenset(i for i, p in enumerate(points) if dot(a, p) == b)
        found[(a, b)] = S
        sub = sorted(S)
        j = min((t for t in range(m) if a[t]), key=lambda t: abs(a[t]))
        proj = [drop(points[i], j) for i in sub]
        for ar, br, _ in hull_facets(proj):
            h = ar[:j] + (0,) + ar[j:]
            h, e = _normalise(points, a, b, h, br)
            nxt = _rotate(points, a, b, h, e)
            if nxt is None:
                raise ArithmeticError('gift wrapping failed')
            if nxt not in found:
                queue.append(nxt)
    return [(a, b, s) for (a, b), s in found.items()]


def vertex_indices(points, facets=None):
    """Indices of the extreme points of conv(points)."""
    k, m = len(points), len(points[0])
    if k == m + 1 or m == 0:
        return list(range(k))
    if facets is None:
        facets = hull_facets(points)
    out = []
    for i in range(k):
        normals = [a for a, b, s in facets if i in s]
        if len(normals) >= m and rank(normals, m) == m:
            out.append(i)
    return out


def triangulate(points):
    """Pulling triangulation of conv(points) (points span Z^m affinely).

    Returns index tuples of (m+1) points each; only vertices of the polytope occur.
    """
    k, m = len(points), len(points[0])
    if k == m + 1 or m == 0:
        return [tuple(range(k))]
    v0 = min(range(k), key=lambda i: points[i])
    out = []
    for a, b, s in hull_facets(points):
        if v0 in s:
            continue
        sub = sorted(s)
        j = next(t for t, x in enumerate(a) if x != 0)
        proj = [drop(points[i], j) for i in sub]
        for simp in triangulate(proj):
            out.append((v0,) + tuple(sub[t] for t in simp))
    return out
