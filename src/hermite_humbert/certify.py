"""
Branch-and-bound certificate for gamma_{K,2}, one chart at a time
(O_K^2 if h_K = 1, Lambda_c = c^{-1} + c for each ideal class otherwise).

Coordinates (x1, x2, s, t):  z_j = x_j + i y_j,  y1 = e^{t-s},  y2 = e^{t+s}.
Region (HumbertProblem): x = r1 g1 + r2 g2 ((g1, g2) a basis of the translation
lattice: O_K, or c^{-2}) with r1 in [-1/2, 1/2], r2 in [0, 1/2] (a parallelogram P
in the (x1, x2)-plane), s in [s_min, s_max] (s_min = 0 when Galois is usable),
t in [t_lo, t_hi].  Every orbit has its highest point (the point where e1 is a
minimal vector) in this region [BCIO Prop. 1.1 + translations, units, x -> -x,
Galois], and at a highest point G = f_{e1} = e^{-2t}.

Boxes are axis-parallel in (x1, x2, s, t), so x1 and x2 can be resolved on
their own natural scales (y1 and y2 differ by a factor up to eps0).

A box B is discarded when one of the following holds (witness recorded):
  rule 0 (value):   UB(B) = min_v sup_B f_v <= threshold,  where
                    f_v = g1 g2,  g_j = (a_j + b_j x_j)^2 / y_j + b_j^2 y_j,
                    sup_B g_j = max over the y_j-endpoints of [max_x (a_j + b_j x)^2 / y + b_j^2 y]
                    (convex in y_j); any vector set gives a valid bound;
  rule 1 (no top):  some v has N|alpha + beta z|^2 = Q1 Q2 < 1 on all of B,
                    Q_j = (a_j + b_j x_j)^2 + b_j^2 y_j^2 (increasing in y_j):
                    e1 is minimal nowhere in B, so B contains no highest point;
  rule 2 (ball):    B lies in a certified ball around a maximiser
                    (analysis.ball_certificate) on which G < gamma*;
  rule 3 (outside): B does not meet the parallelogram P.
threshold = gamma_low (1 + delta) in tolerance mode, gamma* in exact mode.
"""
import itertools
import math
import time

import numpy as np


def _below(x):
    """Largest float <= the mpmath/float number x (for thresholds)."""
    f = float(x)
    return f if f <= x else float(np.nextafter(f, -np.inf))


KAPPA_XU = 8.0 / (3.0 * math.sqrt(3.0))


def _proj_simplex(V):
    """Row-wise Euclidean projection onto the probability simplex."""
    k = V.shape[1]
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, k + 1)
    cond = U - css / ind > 0
    rho = k - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(len(V)), rho] / (rho + 1)
    return np.maximum(V - theta[:, None], 0.0)


def corner_bound(phi, g, wq, ymin, iters=40):
    """Second-order upper bound for sup_B min_i phi_i (rule 4).

    phi (N,k): phi_i at the box centre; g (N,k,4): gradients in q = (x1, x2, u1, u2);
    wq (N,4) = (wx1, wx2, ws, wt) half-widths; ymin (N,2).  For any lambda in the
    simplex, min_i phi_i <= sum lambda_i phi_i, and by Taylor with the universal
    bounds |phi_xx| <= 2/y^2, |phi_xu| <= KAPPA_XU/y, |phi_uu| <= 1 (u_j = t -+ s):
      sup_B sum lambda_i phi_i <= sum lambda_i phi_i(c) + |G1| wx1 + |G2| wx2
             + |G3 + G4| wt + |G4 - G3| ws + (1/2) sum_j (2 wx_j^2/y_j^2 + 2 KAPPA_XU wx_j wu_j / y_j + wu_j^2),
    G = sum lambda_i g_i, wu_j = wt + ws.  lambda is chosen by projected
    subgradient descent on this (convex) expression.  Returns (bound, lambda)."""
    N, k = phi.shape
    wx1, wx2, ws, wt = wq[:, 0], wq[:, 1], wq[:, 2], wq[:, 3]
    wu = wt + ws
    hterm = 0.5 * (2 * wx1 ** 2 / ymin[:, 0] ** 2 + 2 * KAPPA_XU * wx1 * wu / ymin[:, 0] + wu ** 2
                   + 2 * wx2 ** 2 / ymin[:, 1] ** 2 + 2 * KAPPA_XU * wx2 * wu / ymin[:, 1] + wu ** 2)
    # linear functionals on G: coefficients for (G1, G2, G3 + G4, G4 - G3)
    Gt = np.stack([g[:, :, 0], g[:, :, 1], g[:, :, 2] + g[:, :, 3], g[:, :, 3] - g[:, :, 2]], axis=2)
    Wt = np.stack([wx1, wx2, wt, ws], axis=1)

    # Work with phi - min(phi): the simplex is shift-invariant, and keeping the
    # common value (~ log gamma) out of the descent avoids catastrophic
    # cancellation in the projection (it previously broke sum(lambda) = 1 at the
    # 1e-8 level for tiny boxes, which made the bound unsound).
    pmin = phi.min(1)
    dphi = phi - pmin[:, None]

    def obj(lam):
        return (lam * dphi).sum(1) + (np.abs((lam[:, :, None] * Gt).sum(1)) * Wt).sum(1)
    lam = np.zeros((N, k))
    lam[np.arange(N), np.argmin(phi, axis=1)] = 1.0
    best, bl = obj(lam), lam.copy()
    lam = np.full((N, k), 1.0 / k)
    scale = dphi.max(1) + (np.abs(Gt) * Wt[:, None, :]).sum(2).max(1) + 1e-300
    for it in range(iters):
        S = (lam[:, :, None] * Gt).sum(1)
        grad = dphi + (np.sign(S)[:, None, :] * Gt * Wt[:, None, :]).sum(2)
        lam = _proj_simplex(lam - (0.5 / math.sqrt(it + 1)) * grad / scale[:, None])
        lam = np.maximum(lam, 0.0)
        lam /= lam.sum(1, keepdims=True)
        o = obj(lam)
        better = o < best
        best = np.where(better, o, best)
        bl[better] = lam[better]
    bl /= bl.sum(1, keepdims=True)
    best = obj(bl)                      # recompute with exactly normalised weights
    return pmin + best + hterm, bl


class BranchAndBound:
    def __init__(self, problem, gamma_low, best_point=None, delta=1e-9, safety=1e-11, use_corner=True):
        self.P = P = problem
        self.delta = delta
        self.safety = safety
        self.gamma_low = float(gamma_low)
        self.best_point = best_point
        # Own vector set: all v with f_v <= C somewhere on the region.  Any set gives
        # valid bounds (rules 0, 1); centre values are exact whenever <= C, and a
        # centre value > C proves G > C there (then gamma_low was far too small:
        # self.overflow is set and no bound is claimed).
        self.C = min(P.d / 2.0, 1.1 * self.gamma_low)
        self.vecs, self.A1, self.B1, self.A2, self.B2 = P.candidate_vectors(self.C, (P.s_min, P.s_max))
        self.overflow = False
        self.X1, self.X2 = P.x_ranges()
        g1, g2 = P.g1, P.g2
        det = g1[0] * g2[1] - g2[0] * g1[1]
        # r = Minv x  (x = r1 g1 + r2 g2)
        self.Minv = np.array([[g2[1], -g2[0]], [-g1[1], g1[0]]]) / det
        self.scale = np.array([self.X1[1] - self.X1[0], self.X2[1] - self.X2[0],
                               max(P.s_max - P.s_min, 1e-9), 1.0])
        self.stats = {}
        self.use_corner = use_corner

    # ------------------------------------------------------------ geometry
    def r_ranges(self, lo, hi):
        """Bounding intervals of r1, r2 over boxes (r is a linear image of x)."""
        out = []
        for i in range(2):
            a, b = self.Minv[i]
            l = np.minimum(a * lo[:, 0], a * hi[:, 0]) + np.minimum(b * lo[:, 1], b * hi[:, 1])
            h = np.maximum(a * lo[:, 0], a * hi[:, 0]) + np.maximum(b * lo[:, 1], b * hi[:, 1])
            out += [l, h]
        return out[0], out[1], out[2], out[3]

    def outside(self, lo, hi, margin=1e-12):
        r1l, r1h, r2l, r2h = self.r_ranges(lo, hi)
        return (r1h < -0.5 - margin) | (r1l > 0.5 + margin) | (r2h < -margin) | (r2l > 0.5 + margin)

    def centre_inside(self, c):
        r1 = self.Minv[0, 0] * c[:, 0] + self.Minv[0, 1] * c[:, 1]
        r2 = self.Minv[1, 0] * c[:, 0] + self.Minv[1, 1] * c[:, 1]
        return (np.abs(r1) <= 0.5) & (r2 >= 0) & (r2 <= 0.5) & (c[:, 2] >= self.P.s_min) & (c[:, 2] <= self.P.s_max)

    @staticmethod
    def yr(lo, hi):
        return (np.exp(lo[:, 3] - hi[:, 2]), np.exp(hi[:, 3] - lo[:, 2]),
                np.exp(lo[:, 3] + lo[:, 2]), np.exp(hi[:, 3] + hi[:, 2]))

    @staticmethod
    def _Lmax2(A, B, xl, xh):
        La = A[None, :] + B[None, :] * xl[:, None]
        Lb = A[None, :] + B[None, :] * xh[:, None]
        return np.maximum(La * La, Lb * Lb)

    def _gsup(self, A, B, xl, xh, yl, yh):
        M = self._Lmax2(A, B, xl, xh)
        B2 = (B * B)[None, :]
        return np.maximum(M / yl[:, None] + B2 * yl[:, None], M / yh[:, None] + B2 * yh[:, None])

    @staticmethod
    def _ginf(A, B, xl, xh, yl, yh):
        La = A[None, :] + B[None, :] * xl[:, None]
        Lb = A[None, :] + B[None, :] * xh[:, None]
        m = np.where(La * Lb <= 0, 0.0, np.minimum(La * La, Lb * Lb))
        B2 = (B * B)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ys = np.where(B2 > 0, np.sqrt(m / np.where(B2 > 0, B2, 1.0)), yh[:, None])
        ys = np.clip(ys, yl[:, None], yh[:, None])
        return m / ys + B2 * ys

    def _qsup(self, A, B, xl, xh, yh):
        return self._Lmax2(A, B, xl, xh) + (B * B)[None, :] * (yh * yh)[:, None]

    @staticmethod
    def _qinf(A, B, xl, xh, yl):
        La = A[None, :] + B[None, :] * xl[:, None]
        Lb = A[None, :] + B[None, :] * xh[:, None]
        m = np.where(La * Lb <= 0, 0.0, np.minimum(La * La, Lb * Lb))
        return m + (B * B)[None, :] * (yl * yl)[:, None]

    def _centre(self, lo, hi, A1, B1, A2, B2):
        c = 0.5 * (lo + hi)
        x1, x2 = c[:, 0], c[:, 1]
        y1, y2 = np.exp(c[:, 3] - c[:, 2]), np.exp(c[:, 3] + c[:, 2])
        L1 = A1[None, :] + B1[None, :] * x1[:, None]
        L2 = A2[None, :] + B2[None, :] * x2[:, None]
        T1 = (B1[None, :] * y1[:, None]) ** 2
        T2 = (B2[None, :] * y2[:, None]) ** 2
        f = (L1 * L1 + T1) * (L2 * L2 + T2) / (y1 * y2)[:, None]
        kk = min(3, f.shape[1])
        if f.shape[1] > kk:
            k = np.argpartition(f, kk - 1, axis=1)[:, :kk]
        else:
            k = np.tile(np.arange(kk), (len(lo), 1))
        rows = np.arange(len(lo))[:, None]
        B1k, B2k = B1[k], B2[k]
        L1k, L2k, T1k, T2k = L1[rows, k], L2[rows, k], T1[rows, k], T2[rows, k]
        Q1k, Q2k = L1k * L1k + T1k, L2k * L2k + T2k
        d1, d2 = 2 * B1k * L1k / Q1k, 2 * B2k * L2k / Q2k
        h1, h2 = 2 * T1k / Q1k - 1, 2 * T2k / Q2k - 1
        sens = np.stack([np.abs(d1).max(1), np.abs(d2).max(1),
                         np.abs(h2 - h1).max(1), np.abs(h1 + h2).max(1)], axis=1)
        self._last = (f, L1, L2, T1, T2, B1, B2, y1, y2)
        return f.min(axis=1), sens, c

    def _pieces(self, rows, kk=6):
        """phi and q-gradients of the kk smallest f_v at the centres of boxes `rows`."""
        f, L1, L2, T1, T2, B1, B2, y1, y2 = self._last
        f, L1, L2, T1, T2 = f[rows], L1[rows], L2[rows], T1[rows], T2[rows]
        kk = min(kk, f.shape[1])
        k = np.argpartition(f, kk - 1, axis=1)[:, :kk] if f.shape[1] > kk else np.tile(np.arange(kk), (len(rows), 1))
        r = np.arange(len(rows))[:, None]
        L1k, L2k, T1k, T2k = L1[r, k], L2[r, k], T1[r, k], T2[r, k]
        Q1, Q2 = L1k * L1k + T1k, L2k * L2k + T2k
        phi = np.log(f[r, k])
        g = np.stack([2 * B1[k] * L1k / Q1, 2 * B2[k] * L2k / Q2, 2 * T1k / Q1 - 1, 2 * T2k / Q2 - 1], axis=2)
        return k, phi, g

    def in_balls(self, lo, hi):
        """Boxes whose 16 corners, mapped linearly to the ball's scaled
        coordinates q~ = (x1/s1, x2/s2, t - s, t + s), lie in one certified ball."""
        if not self.balls:
            return np.zeros(len(lo), dtype=bool)
        corners = np.array(list(itertools.product((0, 1), repeat=4)))
        C = lo[:, None, :] + corners[None, :, :] * (hi - lo)[:, None, :]
        Q = np.stack([C[..., 0], C[..., 1], C[..., 3] - C[..., 2], C[..., 3] + C[..., 2]], axis=-1)
        inside = np.zeros(len(lo), dtype=bool)
        for c, rho, sc in self.balls:
            d = np.sqrt((((Q - c[None, None, :]) / sc[None, None, :]) ** 2).sum(-1)).max(1)
            inside |= d <= rho
        return inside

    def near_ball(self, c, factor=4.0):
        """Is the point c = (x1, x2, s, t) within factor * radius of a known ball?"""
        q = np.array([c[0], c[1], c[3] - c[2], c[3] + c[2]])
        return any(np.sqrt((((q - b) / sc) ** 2).sum()) <= factor * rho for b, rho, sc in self.balls)

    # ----------------------------------------------------------------- run
    def initial_boxes(self, t_hi, n=(4, 4, 2, 2)):
        P = self.P
        edges = [np.linspace(*self.X1, n[0] + 1), np.linspace(*self.X2, n[1] + 1),
                 np.linspace(P.s_min, P.s_max, n[2] + 1), np.linspace(P.t_lo, t_hi, n[3] + 1)]
        lo, hi = [], []
        for idx in itertools.product(*[range(k) for k in n]):
            lo.append([edges[d][idx[d]] for d in range(4)])
            hi.append([edges[d][idx[d] + 1] for d in range(4)])
        return np.array(lo), np.array(hi)

    def run(self, chunk=2048, max_boxes=5e7, min_width=1e-13, verbose=False, keep_certificate=False,
            exact=None, max_elems=2e6, init=None, balls=None):
        """Tolerance mode (exact=None): proves sup G <= gamma_low (1 + delta).
        Exact mode, exact = (gamma_star, local_vertex_fn): proves sup G <= gamma_star;
        a box that becomes small is handed to local_vertex_fn(centre), which returns
        (gamma, q_centre, rho) for a nearby vertex (rho = certified ball radius).
        init = (lo, hi) restricts the run to these boxes of the initial grid
        (used by run_parallel); balls = certified balls already known.

        Work items (lo, hi, vidx) carry the vectors still relevant for their boxes:
        v is dropped below B if inf_B f_v > UB(B) and inf_B N_v >= 1; it can then
        neither lower UB of a sub-box (UB(B') <= UB(B) < inf f_v) nor witness
        rule 1 there.  Centre values stay exact (the minimiser at the centre c has
        inf_B f <= f(c) <= UB(B))."""
        t0 = time.time()
        P = self.P
        self.balls, self.exact, self.counterexample = list(balls or []), exact, None
        self.cert4, self.n_rule4 = [], 0
        thr_exact = _below(exact[0]) if exact is not None else None
        t_hi = 0.5 * math.log(1.0 / (self.gamma_low if exact is None else thr_exact))
        lo, hi = self.initial_boxes(t_hi) if init is None else init
        stack = [(lo, hi, np.arange(len(self.A1)))]
        processed = 0
        counts = np.zeros(4, dtype=np.int64)
        unresolved = []
        cert = [] if keep_certificate else None
        scale = self.scale
        while stack:
            lo, hi, vidx = stack.pop()
            # keep (boxes x vectors) arrays below ~max_elems entries (memory)
            if len(lo) > max(8, min(chunk, int(max_elems // max(1, len(vidx))))) and len(lo) > 1:
                cen = 0.5 * (lo + hi) / scale
                k = int(np.argmax(cen.max(0) - cen.min(0)))
                order = np.argsort(cen[:, k])
                h = len(lo) // 2
                stack.append((lo[order[h:]], hi[order[h:]], vidx))
                stack.append((lo[order[:h]], hi[order[:h]], vidx))
                continue
            processed += len(lo)
            if processed > max_boxes:
                unresolved.append((lo, hi, np.full(len(lo), np.inf)))
                for L, H, _ in stack:
                    unresolved.append((L, H, np.full(len(L), np.inf)))
                stack = []
                break
            # rule 3: outside the translation parallelogram
            out = self.outside(lo, hi)
            if out.any():
                counts[3] += int(out.sum())
                if cert is not None:
                    cert.append((lo[out], hi[out], np.full(out.sum(), -1), np.full(out.sum(), 3)))
                lo, hi = lo[~out], hi[~out]
                if len(lo) == 0:
                    continue
            A1, B1, A2, B2 = self.A1[vidx], self.B1[vidx], self.A2[vidx], self.B2[vidx]
            x1l, x1h, x2l, x2h = lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]
            y1l, y1h, y2l, y2h = self.yr(lo, hi)
            U = self._gsup(A1, B1, x1l, x1h, y1l, y1h) * self._gsup(A2, B2, x2l, x2h, y2l, y2h)
            jarg = np.argmin(U, axis=1)
            ub = U[np.arange(len(lo)), jarg]
            NQ = self._qsup(A1, B1, x1l, x1h, y1h) * self._qsup(A2, B2, x2l, x2h, y2h)
            jtop = np.argmin(NQ, axis=1)
            nq = NQ[np.arange(len(lo)), jtop]
            cv, sens, cen = self._centre(lo, hi, A1, B1, A2, B2)
            ins = self.centre_inside(cen) & (cen[:, 3] >= P.t_lo)
            if ins.any() and np.max(np.where(ins, cv, -np.inf)) > self.C:
                # G > C somewhere: gamma_low was far below the maximum and this
                # vector set cannot certify anything useful -> stop at once
                self.overflow = True
                i = int(np.argmax(np.where(ins, cv, -np.inf)))
                c = cen[i]
                self.best_point = np.array([c[0], c[1], math.exp(c[3] - c[2]), math.exp(c[3] + c[2])])
                unresolved.append((lo, hi, np.full(len(lo), np.inf)))
                for L, H, _ in stack:
                    unresolved.append((L, H, np.full(len(L), np.inf)))
                stack = []
                break
            if exact is not None and ins.any() and np.max(np.where(ins, cv, -np.inf)) > exact[0] * (1 + 1e-12):
                # a point with G > gamma*: the claimed maximum is wrong -> stop at once
                i = int(np.argmax(np.where(ins, cv, -np.inf)))
                self.gamma_low = float(cv[i])
                c = cen[i]
                self.best_point = np.array([c[0], c[1], math.exp(c[3] - c[2]), math.exp(c[3] + c[2])])
                unresolved.append((lo, hi, np.full(len(lo), np.inf)))
                for L, H, _ in stack:
                    unresolved.append((L, H, np.full(len(L), np.inf)))
                stack = []
                break
            if ins.any():
                i = int(np.argmax(np.where(ins, cv, -np.inf)))
                if cv[i] > self.gamma_low:
                    self.gamma_low = float(cv[i])
                    c = cen[i]
                    self.best_point = np.array([c[0], c[1], math.exp(c[3] - c[2]), math.exp(c[3] + c[2])])
            thr = self.gamma_low * (1 + self.delta) if exact is None else thr_exact
            r0 = ub * (1 + self.safety) <= thr
            r1 = (nq * (1 + self.safety) < 1.0) & ~r0
            r2 = (self.in_balls(lo, hi) & ~r0 & ~r1) if exact is not None else np.zeros(len(lo), bool)
            ok = r0 | r1 | r2
            counts[0] += int(r0.sum())
            counts[1] += int(r1.sum())
            counts[2] += int(r2.sum())
            if cert is not None and ok.any():
                rule = np.where(r0, 0, np.where(r1, 1, 2))[ok]
                wit = np.where(r0, vidx[jarg], np.where(r1, vidx[jtop], -1))[ok]
                cert.append((lo[ok], hi[ok], wit, rule))
            keep = ~ok
            if not keep.any():
                continue
            if self.use_corner:
                # the corner bound only pays off where G(centre) is close to the threshold
                thr0 = self.gamma_low * (1 + self.delta) if exact is None else thr_exact
                small = np.all((hi - lo) <= 0.05 * scale, axis=1)
                rows = np.where(keep & small & (cv >= 0.8 * thr0))[0]
            if self.use_corner and len(rows):
                kidx, phi, g = self._pieces(rows)
                wq = 0.5 * (hi[rows] - lo[rows])
                ymin = np.stack([y1l[rows], y2l[rows]], axis=1)
                bnd, lam = corner_bound(phi, g, wq, ymin)
                thr = self.gamma_low * (1 + self.delta) if exact is None else thr_exact
                r4 = bnd + math.log1p(self.safety) <= math.log(thr)
                if r4.any():
                    counts4 = int(r4.sum())
                    self.n_rule4 = getattr(self, "n_rule4", 0) + counts4
                    if cert is not None:
                        rr = rows[r4]
                        self.cert4.append((lo[rr], hi[rr], vidx[kidx[r4]], lam[r4]))
                    keep[rows[r4]] = False
                    if not keep.any():
                        continue
            lo, hi, sens, ub = lo[keep], hi[keep], sens[keep], ub[keep]
            Lf = self._ginf(A1, B1, x1l[keep], x1h[keep], y1l[keep], y1h[keep]) * \
                self._ginf(A2, B2, x2l[keep], x2h[keep], y2l[keep], y2h[keep])
            NI = self._qinf(A1, B1, x1l[keep], x1h[keep], y1l[keep]) * \
                self._qinf(A2, B2, x2l[keep], x2h[keep], y2l[keep])
            useful = ((Lf <= (ub * (1 + 1e-9))[:, None]) | (NI < 1.0)).any(axis=0)
            vidx = vidx[useful]
            width = hi - lo
            if exact is not None:
                small = np.all(width <= 1e-5 * scale, axis=1)
                if small.any():
                    cand = [i for i in np.where(small)[0][:8] if not self.near_ball(0.5 * (lo[i] + hi[i]))]
                    if cand:
                        i = int(cand[0])
                        res = exact[1](0.5 * (lo[i] + hi[i]))
                        self.vertex_calls = getattr(self, "vertex_calls", 0) + 1
                        if res is not None:
                            g, qc, rho, sc = res
                            if g > exact[0] * (1 + 1e-12):
                                self.counterexample = (g, qc)
                                stack = []
                                unresolved.append((lo, hi, np.full(len(lo), np.inf)))
                                break
                            elif rho is not None and not any(np.linalg.norm(qc - b[0]) < 1e-9 for b in self.balls):
                                self.balls.append((qc, rho, sc))
                    inb = self.in_balls(lo, hi)
                    if inb.any():
                        counts[2] += int(inb.sum())
                        if cert is not None:
                            cert.append((lo[inb], hi[inb], np.full(inb.sum(), -1), np.full(inb.sum(), 2)))
                        lo, hi, width, sens, ub = lo[~inb], hi[~inb], width[~inb], sens[~inb], ub[~inb]
                        if len(lo) == 0:
                            continue
            tiny = np.all(width <= min_width * scale, axis=1)
            if tiny.any():
                unresolved.append((lo[tiny], hi[tiny], ub[tiny]))
                lo, hi, width, sens, ub = lo[~tiny], hi[~tiny], width[~tiny], sens[~tiny], ub[~tiny]
                if len(lo) == 0:
                    continue
            pri = width * (sens + 0.2 * sens.max(axis=1, keepdims=True) + 1e-3)
            pri = np.where(width > 0, pri, -1)
            dim = np.argmax(pri, axis=1)
            r = np.arange(len(lo))
            mid = 0.5 * (lo[r, dim] + hi[r, dim])
            lo2, hi1 = lo.copy(), hi.copy()
            hi1[r, dim] = mid
            lo2[r, dim] = mid
            stack.append((np.vstack([lo, lo2]), np.vstack([hi1, hi]), vidx))
            if verbose and processed % (200 * chunk) < len(lo):
                print("   B&B: %d boxes, gamma_low = %.15g, %.1fs" % (processed, self.gamma_low, time.time() - t0))
        un_ub = max([float(np.max(u)) for _, _, u in unresolved], default=-np.inf)
        self.unresolved = unresolved
        self.certificate = cert
        self.t_hi = t_hi
        self.stats = {"boxes": processed, "discarded_by_rule": counts.tolist() + [self.n_rule4], "time": time.time() - t0,
                      "unresolved": int(sum(len(u) for _, _, u in unresolved))}
        if exact is not None:
            self.stats["balls"] = len(self.balls)
        base = self.gamma_low * (1 + self.delta) * (1 + self.safety) if exact is None else float(exact[0])
        self.gamma_upper = max(base, un_ub * (1 + self.safety)) if not self.overflow else float("inf")
        self.stats["vectors"] = len(self.vecs)
        return self.gamma_upper


# ---------------------------------------------------------------------------
# independent re-check of a recorded certificate with outward rounding
# ---------------------------------------------------------------------------
def _rule4_float(bb, lo, hi, idx, lam, thr, exp_ulps=8, log_ulps=4):
    """Vectorised re-check of rule-4 boxes in double precision with rigorous
    error bounds (same model as rules 0-3: IEEE round-to-nearest, |fl(x op y) -
    x op y| <= u |x op y|, embeddings of the witness vectors within 8u, exp within
    exp_ulps ulp, and log within log_ulps ulp).  Every computed quantity carries an
    upper bound on its absolute error; a box is accepted only if the resulting
    upper bound for  (sum lam phi + lin) / sum lam + hterm  (the same bound as the
    interval check) is below a lower bound for log(thr).  Returns a boolean
    array; False only means "not confirmed here" (the caller then uses the
    interval check), never "rejected"."""
    u = 2.0 ** -53
    ee = 8 * u
    eu = exp_ulps * u
    F = bb.P.F
    k = idx.shape[1]
    uniq = np.unique(idx)
    E = np.array([F.emb(bb.vecs[int(i)][0]) + F.emb(bb.vecs[int(i)][1]) for i in uniq])  # a1, a2, b1, b2
    pos = np.searchsorted(uniq, idx)
    a1, a2, b1, b2 = E[pos, 0], E[pos, 1], E[pos, 2], E[pos, 3]            # (nb, k)
    lam = np.asarray(lam, dtype=float)
    c = 0.5 * (lo + hi)
    w = np.maximum(hi - c, c - lo) * (1 + 4 * u)                           # half-widths, rounded up
    x1, x2, sc, tc = c[:, 0:1], c[:, 1:2], c[:, 2:3], c[:, 3:4]
    arg1, arg2 = tc - sc, tc + sc                                          # log y1, log y2 (err <= u|.|)
    y1, y2 = np.exp(arg1), np.exp(arg2)
    ey1 = eu + 1.01 * u * np.abs(arg1) + 2 * u                             # relative errors of y1, y2
    ey2 = eu + 1.01 * u * np.abs(arg2) + 2 * u
    U = arg1 + arg2
    eU = u * (np.abs(arg1) + np.abs(arg2)) + u * np.abs(U)

    def piece(a, b, x, y, ey):
        bx = b * x
        L = a + bx
        eL = 1.01 * (ee + 3 * u) * (np.abs(a) + np.abs(bx))
        by = b * y
        T = by * by
        eTr = 1.01 * (2 * (ee + ey + u) + 2 * u)                            # relative error of T
        Q = L * L + T
        eQ = 1.01 * (2 * np.abs(L) * eL + eL * eL + 2 * u * L * L + T * eTr + 2 * u * Q)
        Qlo = Q - eQ
        with np.errstate(divide="ignore", invalid="ignore"):
            logQ = np.log(Q)
            elog = eQ / Qlo + log_ulps * u * np.abs(logQ) + 1e-300
            gL = 2 * b * L / Q
            egL = 1.01 * (2 * np.abs(b) * (1 + ee) * (eL / Qlo + (np.abs(L) + eL) * eQ / (Q * Qlo))
                          + np.abs(gL) * (ee + 4 * u))
            r = 2 * T / Q
            gT = r - 1
            egT = 1.01 * (2 * (T * eTr / Qlo + T * (1 + eTr) * eQ / (Q * Qlo)) + 4 * u * (np.abs(r) + 1))
        return logQ, elog, gL, egL, gT, egT, Qlo

    lq1, el1, g1, eg1, g3, eg3, Q1lo = piece(a1, b1, x1, y1, ey1)
    lq2, el2, g2, eg2, g4, eg4, Q2lo = piece(a2, b2, x2, y2, ey2)
    phi = lq1 + lq2 - U
    ephi = el1 + el2 + eU + 3 * u * (np.abs(lq1) + np.abs(lq2) + np.abs(U))
    fk = (k + 2) * u
    S = lam.sum(1)
    S_lo, S_hi = S * (1 - fk), S * (1 + fk)
    Phi = (lam * phi).sum(1)
    ePhi = (lam * ephi).sum(1) + fk * (lam * np.abs(phi)).sum(1)
    G, eG = [], []
    for g, eg in ((g1, eg1), (g2, eg2), (g3, eg3), (g4, eg4)):
        G.append((lam * g).sum(1))
        eG.append((lam * eg).sum(1) + fk * (lam * np.abs(g)).sum(1))
    Gp, Gm = G[2] + G[3], G[3] - G[2]
    lin = ((np.abs(G[0]) + eG[0]) * w[:, 0] + (np.abs(G[1]) + eG[1]) * w[:, 1]
           + (np.abs(Gp) + eG[2] + eG[3] + 2 * u * np.abs(Gp)) * w[:, 3]
           + (np.abs(Gm) + eG[2] + eG[3] + 2 * u * np.abs(Gm)) * w[:, 2]) * (1 + 8 * u)
    num = Phi + ePhi + lin
    num = num + 4 * u * (np.abs(Phi) + lin)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(num >= 0, num / S_lo, num / S_hi)
    term = term + 4 * u * np.abs(term)
    am1 = lo[:, 3] - hi[:, 2]
    am2 = lo[:, 3] + lo[:, 2]
    ymin1 = np.exp(am1) * (1 - eu - 1.01 * u * np.abs(am1) - 2 * u)       # lower bounds
    ymin2 = np.exp(am2) * (1 - eu - 1.01 * u * np.abs(am2) - 2 * u)
    kap = 8.0 / (3.0 * math.sqrt(3.0)) * (1 + 8 * u)
    wu = (w[:, 2] + w[:, 3]) * (1 + 2 * u)
    with np.errstate(divide="ignore", invalid="ignore"):
        hterm = (2 * w[:, 0] ** 2 / ymin1 ** 2 + 2 * kap * w[:, 0] * wu / ymin1 + wu ** 2
                 + 2 * w[:, 1] ** 2 / ymin2 ** 2 + 2 * kap * w[:, 1] * wu / ymin2 + wu ** 2) / 2 * (1 + 32 * u)
    bound = term + hterm
    lt = math.log(thr)
    logthr_lo = lt - 4 * u * abs(lt) - 1e-300
    ok = ((S_lo > 0) & np.all(Q1lo > 0, axis=1) & np.all(Q2lo > 0, axis=1)
          & (ymin1 > 0) & (ymin2 > 0) & np.isfinite(bound) & (bound <= logthr_lo))
    return ok


def verify_certificate(bb, threshold, exp_ulps=8, float_rule4=True):
    """Re-check every discarded box of bb.certificate with outward rounding,
    independently of the B&B code path:
      - embeddings a_j, b_j of the witness vectors are recomputed from the
        exact integers and enclosed with relative error 8u (u = 2^-53);
      - every elementary operation is enclosed via |fl(x op y) - x op y| <= u |x op y|
        (IEEE round-to-nearest); exp() with exp_ulps ulp;
      - rule 0 needs sup f_v <= threshold, rule 1 sup N_v < 1, rule 3 the
        r-intervals to miss the parallelogram (margin 1e-14), rule 2 all
        corners inside a certified ball (radii taken from analysis.ball_certificate).
    Also returns the volume of discarded + unresolved boxes relative to the
    initial region (boxes are disjoint by construction, so 1.0 means a tiling)."""
    P = bb.P
    F = P.F
    u = 2.0 ** -53
    ee = 8 * u
    res = {r: [0, 0] for r in range(4)}
    if bb.certificate is None:
        raise ValueError("run the B&B with keep_certificate=True")
    thr = float(threshold)
    emb = {}

    def up(x, k=1):
        return x * (1 + (k + 1) * u)

    def Lmax(a, b, xl, xh):
        out = np.zeros_like(a)
        for x in (xl, xh):
            v = a + b * x
            err = (3 * u + 2 * ee) * (np.abs(a) + np.abs(b * x)) * (1 + 4 * u)
            out = np.maximum(out, np.abs(v) + err)
        return out * (1 + u)

    vol = 0.0
    for lo, hi, wit, rule in bb.certificate:
        vol += float(np.prod(hi - lo, axis=1).sum())
        for r in range(4):
            sel = rule == r
            if not sel.any():
                continue
            L, H, W = lo[sel], hi[sel], wit[sel]
            if r == 3:
                r1l, r1h, r2l, r2h = bb.r_ranges(L, H)
                sl = 1e-14
                ok = (r1h + sl < -0.5) | (r1l - sl > 0.5) | (r2h + sl < 0) | (r2l - sl > 0.5)
            elif r == 2:
                ok = bb.in_balls(L, H)
            else:
                rows = []
                for i in W:
                    i = int(i)
                    if i not in emb:
                        a, b = bb.vecs[i]
                        emb[i] = F.emb(a) + F.emb(b)      # (a1, a2, b1, b2)
                    rows.append(emb[i])
                E = np.array(rows)
                a1, a2, b1, b2 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
                eu = exp_ulps * u
                y1l = np.exp(L[:, 3] - H[:, 2]) * (1 - eu) * (1 - 2 * u)
                y1h = np.exp(H[:, 3] - L[:, 2]) * (1 + eu) * (1 + 2 * u)
                y2l = np.exp(L[:, 3] + L[:, 2]) * (1 - eu) * (1 - 2 * u)
                y2h = np.exp(H[:, 3] + H[:, 2]) * (1 + eu) * (1 + 2 * u)
                M1 = up(Lmax(a1, b1, L[:, 0], H[:, 0]) ** 2)
                M2 = up(Lmax(a2, b2, L[:, 1], H[:, 1]) ** 2)
                c1 = up((np.abs(b1) * (1 + ee)) ** 2)
                c2 = up((np.abs(b2) * (1 + ee)) ** 2)
                if r == 0:
                    g1 = np.maximum(up(up(M1 / y1l) + up(c1 * y1l)), up(up(M1 / y1h) + up(c1 * y1h)))
                    g2 = np.maximum(up(up(M2 / y2l) + up(c2 * y2l)), up(up(M2 / y2h) + up(c2 * y2h)))
                    ok = up(g1 * g2) <= thr
                else:
                    q1 = up(M1 + up(c1 * up(y1h * y1h)))
                    q2 = up(M2 + up(c2 * up(y2h * y2h)))
                    ok = up(q1 * q2) < 1.0
            res[r][0] += int(ok.sum())
            res[r][1] += int((~ok).sum())
    # rule 4 (corner bound): a vectorised float check with rigorous error bounds
    # (_rule4_float) first; boxes it cannot confirm are re-checked in interval
    # arithmetic (mpmath.iv), so every box gets at least the interval check's rigour
    res[4] = [0, 0]
    if getattr(bb, "cert4", None):
        from mpmath import iv, mp
        old_dps = mp.dps
        mp.dps = 30
        iv.dps = 30
        embi = {}
        kap = iv.mpf(8) / (3 * iv.sqrt(3))
        logthr = iv.log(iv.mpf(thr)).a
        for lo, hi, idx, lam in bb.cert4:
            vol += float(np.prod(hi - lo, axis=1).sum())
            okf = _rule4_float(bb, lo, hi, idx, lam, thr, exp_ulps) if float_rule4 else np.zeros(len(lo), bool)
            res[4][0] += int(okf.sum())
            for b in np.where(~okf)[0]:
                c = 0.5 * (lo[b] + hi[b])                    # any expansion point in the box
                # half-widths rounded up (the float subtraction may round down)
                w = [iv.mpf(max(hi[b][j] - c[j], c[j] - lo[b][j])) * (1 + iv.mpf(2) ** -51) for j in range(4)]
                x1, x2, sc, tc = [iv.mpf(float(t)) for t in c]
                y1, y2 = iv.exp(tc - sc), iv.exp(tc + sc)
                ymin1 = iv.exp(iv.mpf(lo[b][3]) - iv.mpf(hi[b][2]))
                ymin2 = iv.exp(iv.mpf(lo[b][3]) + iv.mpf(lo[b][2]))
                S = iv.mpf(0); Phi = iv.mpf(0); G = [iv.mpf(0)] * 4
                for i, l in zip(idx[b], lam[b]):
                    i = int(i)
                    if i not in embi:
                        a, bb_ = bb.vecs[i]
                        e = F.emb_mp(a) + F.emb_mp(bb_)          # (a1, a2, b1, b2) at 30 digits
                        embi[i] = [iv.mpf([t - mp.mpf(10) ** -25, t + mp.mpf(10) ** -25]) for t in e]
                    a1, a2, b1, b2 = embi[i]
                    l = iv.mpf(float(l))
                    L1, L2 = a1 + b1 * x1, a2 + b2 * x2
                    T1, T2 = (b1 * y1) ** 2, (b2 * y2) ** 2
                    Q1, Q2 = L1 * L1 + T1, L2 * L2 + T2
                    phi = iv.log(Q1) + iv.log(Q2) - iv.log(y1) - iv.log(y2)
                    g = [2 * b1 * L1 / Q1, 2 * b2 * L2 / Q2, 2 * T1 / Q1 - 1, 2 * T2 / Q2 - 1]
                    S += l
                    Phi += l * phi
                    G = [G[j] + l * g[j] for j in range(4)]
                lin = abs(G[0]) * w[0] + abs(G[1]) * w[1] + abs(G[2] + G[3]) * w[3] + abs(G[3] - G[2]) * w[2]
                wu = w[2] + w[3]
                hterm = (2 * w[0] ** 2 / ymin1 ** 2 + 2 * kap * w[0] * wu / ymin1 + wu ** 2
                         + 2 * w[1] ** 2 / ymin2 ** 2 + 2 * kap * w[1] * wu / ymin2 + wu ** 2) / 2
                bound = (Phi + lin) / S + hterm                 # valid for lambda / sum(lambda)
                ok4 = S.a > 0 and bound.b <= logthr
                res[4][0 if ok4 else 1] += 1
        mp.dps = old_dps
    region_vol = (bb.X1[1] - bb.X1[0]) * (bb.X2[1] - bb.X2[0]) * (P.s_max - P.s_min) * (bb.t_hi - P.t_lo)
    un_vol = sum(float(np.prod(h - l, axis=1).sum()) for l, h, _ in bb.unresolved)
    return {"verified": {r: res[r][0] for r in range(5)}, "failed": {r: res[r][1] for r in range(5)},
            "volume_ratio": (vol + un_vol) / region_vol}


# ---------------------------------------------------------------------------
# parallel branch-and-bound over the initial grid (fork-based)
# ---------------------------------------------------------------------------
_PAR = {}


def _par_task(args):
    """Worker: run the B&B of _PAR['bb'] on one block of initial boxes.  With
    _PAR['verify_thr'] set, the block's certificate is re-checked right here
    (verify_certificate, with the balls this block used) and only the counts and
    the volume fraction are returned, not the boxes: memory and checkpoints stay
    O(one block)."""
    i, lo, hi, balls = args
    bb = _PAR["bb"]
    bb.gamma_low = _PAR["gamma_low"]
    bb.overflow = False
    bb.run(init=(lo, hi), balls=balls, **_PAR["kw"])
    keep = _PAR["kw"].get("keep_certificate", False)
    thr = _PAR.get("verify_thr")
    out = {"index": i, "balls": bb.balls, "unresolved": bb.unresolved, "stats": bb.stats,
           "counterexample": bb.counterexample, "overflow": bb.overflow,
           "gamma_low": bb.gamma_low, "best_point": bb.best_point, "t_hi": bb.t_hi,
           "n_rule4": bb.n_rule4, "certificate": None, "cert4": [], "verification": None}
    if thr is not None:
        if bb.counterexample is None and not bb.overflow:
            out["verification"] = verify_certificate(bb, thr)
    elif keep:
        out["certificate"], out["cert4"] = bb.certificate, bb.cert4
    return out


def _par_child(args, conn):
    """Forked process for one block: send the result (or the error) and exit."""
    import os
    try:
        conn.send(_par_task(args))
    except BaseException as e:          # noqa: report instead of dying silently
        try:
            conn.send({"error": repr(e)})
        except Exception:
            pass
    finally:
        conn.close()
        os._exit(0)


def _ckpt_path(ckpt, i):
    import os
    d, prefix = ckpt
    key = "%03d" % i if isinstance(i, int) else str(i)      # follow-up blocks: "199.0", ...
    return os.path.join(d, "%s_b%s.pkl" % (prefix, key))


def _ckpt_save(ckpt, i, r):
    """Atomic write of one finished block's result."""
    import os
    import pickle
    path = _ckpt_path(ckpt, i)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(r, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _ckpt_load(ckpt, i):
    import os
    import pickle
    path = _ckpt_path(ckpt, i)
    if not os.path.exists(path):
        return None
    try:
        import mpmath
        with mpmath.workprec(1 << 16), open(path, "rb") as fh:   # see core._ck_load
            return pickle.load(fh)
    except Exception:                    # noqa: truncated / unreadable -> recompute
        return None


def run_parallel(bb, jobs, grid=(8, 8, 2, 2), ckpt=None, verify_thr=None, **kw):
    """bb.run(**kw) split over `jobs` processes: the initial grid (finer than the
    serial one) is dealt out block by block, one forked process per block;
    certified balls found by finished blocks are handed to the blocks started
    later.  The results (balls, unresolved boxes, statistics; the certificate, or
    with verify_thr its re-check) are merged into bb as a serial run would leave
    them (the union of the blocks tiles the region).

    verify_thr: re-check every block's certificate inside its own process
    (verify_certificate with that threshold) and keep only the counts and volume
    fractions; their sums are bb.verification (same totals as re-checking the
    merged certificate; boxes are never collected in the parent).
    ckpt = (directory, prefix): every finished block's result is saved there, and
    blocks already saved are loaded instead of recomputed, so an interrupted run
    resumes.  The prefix must identify the problem, gamma* and the options.

    A block whose process dies is retried once and otherwise left unresolved, so
    the certificate fails rather than the run hanging.  Stops early at the first
    counterexample / overflow.  Needs the 'fork' start method."""
    import multiprocessing as mp
    t0 = time.time()
    exact = kw.get("exact")
    thr_exact = _below(exact[0]) if exact is not None else None
    t_hi = 0.5 * math.log(1.0 / (bb.gamma_low if exact is None else thr_exact))
    lo, hi = bb.initial_boxes(t_hi, n=grid)
    blocks = [(lo[i:i + 1], hi[i:i + 1]) for i in range(len(lo))]
    if verify_thr is not None:
        kw = dict(kw, keep_certificate=True)
    _PAR.clear()
    _PAR.update(bb=bb, gamma_low=bb.gamma_low, kw=kw, verify_thr=verify_thr)
    balls = []
    keep = kw.get("keep_certificate") and verify_thr is None
    cert, cert4, unresolved = ([] if keep else None), [], []
    counts = np.zeros(5, dtype=np.int64)
    boxes, n_rule4, stop, lost, resumed = 0, 0, None, 0, 0
    best = (bb.gamma_low, bb.best_point)
    ver = {"verified": {r: 0 for r in range(5)}, "failed": {r: 0 for r in range(5)}, "volume_ratio": 0.0}
    ctx = mp.get_context("fork")

    def merge(r):
        nonlocal boxes, n_rule4, best, counts
        for qb in r["balls"]:
            if not any(np.linalg.norm(qb[0] - b[0]) < 1e-9 for b in balls):
                balls.append(qb)
        if cert is not None and r["certificate"]:
            cert.extend(r["certificate"])
        cert4.extend(r["cert4"])
        unresolved.extend(r["unresolved"])
        counts += np.array(r["stats"]["discarded_by_rule"], dtype=np.int64)
        boxes += r["stats"]["boxes"]
        n_rule4 += r["n_rule4"]
        if r["gamma_low"] > best[0] and r["best_point"] is not None:
            best = (r["gamma_low"], r["best_point"])
        v = r.get("verification")
        if v is not None:
            for key in ("verified", "failed"):
                for rr, c in v[key].items():
                    ver[key][rr] = ver[key].get(rr, 0) + c
            ver["volume_ratio"] += v["volume_ratio"]

    def is_stop(r):
        return r["counterexample"] is not None or r["overflow"] or (
            exact is not None and r["gamma_low"] > exact[0] * (1 + 1e-12))

    P = bb.P
    region_vol = (bb.X1[1] - bb.X1[0]) * (bb.X2[1] - bb.X2[0]) * (P.s_max - P.s_min) * (t_hi - P.t_lo)
    queue = []
    follow_ups = 0

    def accept(key, r):
        """Merge a block result (fresh or loaded).  Boxes the block left undecided
        because it ran out of its box budget (upper bound inf; not a stop) become
        follow-up blocks key.0 .. key.3, so a block can never fail for lack of
        budget; their volume is taken out of this block's re-check, since the
        follow-ups account for it.  Deterministic, so a resume finds the same keys."""
        nonlocal resumed, follow_ups
        budget = [(l, h) for l, h, u in r["unresolved"] if len(u) and np.all(np.isinf(u))]
        if budget and not is_stop(r):
            r["unresolved"] = [(l, h, u) for l, h, u in r["unresolved"]
                               if not (len(u) and np.all(np.isinf(u)))]
            BL = np.vstack([l for l, _ in budget])
            BH = np.vstack([h for _, h in budget])
            if r.get("verification") is not None:
                r["verification"] = dict(r["verification"])
                r["verification"]["volume_ratio"] -= float(np.prod(BH - BL, axis=1).sum()) / region_vol
            merge(r)
            for part, idx in enumerate(np.array_split(np.arange(len(BL)), 4)):
                if len(idx) == 0:
                    continue
                ck_key = "%s.%d" % (key, part)
                follow_ups += 1
                rr = _ckpt_load(ckpt, ck_key) if ckpt is not None else None
                if rr is not None and not is_stop(rr):
                    resumed += 1
                    accept(ck_key, rr)
                else:
                    queue.append((ck_key, (BL[idx], BH[idx]), 0))
        else:
            merge(r)

    for i, b in enumerate(blocks):
        r = _ckpt_load(ckpt, i) if ckpt is not None else None
        if r is not None and not is_stop(r):
            resumed += 1
            accept(i, r)
        else:
            queue.append((i, b, 0))
    running = []                              # (process, connection, index, block, attempts)
    try:
        while (queue or running) and stop is None:
            while queue and len(running) < jobs:
                i, (blo, bhi), att = queue.pop(0)
                if jobs <= 1:
                    # one process: run the block here (no fork overhead); errors are
                    # handled like a dead block process
                    try:
                        r = _par_task((i, blo, bhi, list(balls)))
                    except Exception as e:          # noqa
                        r = {"error": repr(e)}
                    running.append((None, r, i, (blo, bhi), att))
                    continue
                recv_end, send_end = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_par_child, args=((i, blo, bhi, list(balls)), send_end))
                proc.start()
                send_end.close()
                running.append((proc, recv_end, i, (blo, bhi), att))
            still, progressed = [], False
            for proc, conn, i, blk, att in running:
                r = None
                if proc is None:                    # in-process block: conn is its result
                    r = conn
                elif conn.poll():
                    try:
                        r = conn.recv()
                    except EOFError:            # died without a result
                        r = None
                    proc.join()
                elif proc.is_alive():
                    still.append((proc, conn, i, blk, att))
                    continue
                else:
                    proc.join()
                if proc is not None:
                    conn.close()
                progressed = True
                if r is None or "error" in r:
                    # the block's process died (e.g. killed) or raised: retry once, then
                    # leave the block unresolved (the certificate then fails, it never hangs)
                    if att == 0:
                        queue.append((i, blk, 1))
                    else:
                        lost += 1
                        unresolved.append((blk[0], blk[1], np.full(len(blk[0]), np.inf)))
                    continue
                if is_stop(r):
                    merge(r)
                    stop = r
                else:
                    if ckpt is not None:
                        _ckpt_save(ckpt, i, r)          # the raw result; accept() is deterministic
                    accept(i, r)
            running = still
            if not progressed:
                time.sleep(0.05)
    finally:
        for proc, conn, _, _, _ in running:    # early stop (or an exception): clean up
            if proc is None:
                continue
            proc.terminate()
            proc.join()
            conn.close()
    bb.balls = balls
    bb.certificate = cert
    bb.cert4 = cert4
    bb.unresolved = unresolved
    bb.t_hi = t_hi
    bb.n_rule4 = n_rule4
    bb.gamma_low, bb.best_point = best
    bb.counterexample = stop["counterexample"] if stop else None
    bb.overflow = bool(stop and stop["overflow"])
    bb.verification = ver if verify_thr is not None else None
    if stop is not None:            # the remaining blocks count as unresolved
        unresolved.append((lo, hi, np.full(len(lo), np.inf)))
    un_ub = max([float(np.max(u)) for _, _, u in unresolved], default=-np.inf)
    bb.stats = {"boxes": boxes, "discarded_by_rule": counts[:4].tolist() + [n_rule4],
                "time": time.time() - t0, "unresolved": int(sum(len(u) for _, _, u in unresolved)),
                "vectors": len(bb.vecs), "jobs": jobs, "lost_blocks": lost, "resumed_blocks": resumed,
                "follow_up_blocks": follow_ups}
    if exact is not None:
        bb.stats["balls"] = len(balls)
    base = bb.gamma_low * (1 + bb.delta) * (1 + bb.safety) if exact is None else float(exact[0])
    bb.gamma_upper = max(base, un_ub * (1 + bb.safety)) if not bb.overflow else float("inf")
    _PAR.clear()
    return bb.gamma_upper


def _verify_child(args, conn):
    import os
    try:
        view, thr = args
        conn.send(verify_certificate(view, thr))
    except BaseException as e:          # noqa
        try:
            conn.send({"error": repr(e)})
        except Exception:
            pass
    finally:
        conn.close()
        os._exit(0)


def verify_parallel(bb, threshold, jobs, chunks_per_job=6):
    """verify_certificate(bb, threshold) with the recorded boxes split into chunks
    that are re-checked in forked processes (the rule-4 interval re-check is the
    slow part).  Every chunk is checked by the same routine; the counts and the
    volume fractions add up to the serial result.  A chunk whose process dies is
    retried once, then counted as failed (never silently skipped)."""
    import copy
    import multiprocessing as mp
    if jobs <= 1 or bb.certificate is None:
        return verify_certificate(bb, threshold)
    n = max(1, jobs * chunks_per_job)
    cert, cert4 = list(bb.certificate), list(getattr(bb, "cert4", []) or [])
    views = []
    for k in range(n):
        v = copy.copy(bb)
        v.certificate, v.cert4, v.unresolved = cert[k::n], cert4[k::n], []
        views.append(v)
    rest = copy.copy(bb)                       # the unresolved boxes' volume only
    rest.certificate, rest.cert4 = [], []
    total = verify_certificate(rest, threshold)
    ctx = mp.get_context("fork")
    queue = [(v, 0) for v in views]
    running = []
    while queue or running:
        while queue and len(running) < jobs:
            v, att = queue.pop(0)
            r_end, s_end = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_verify_child, args=((v, threshold), s_end))
            p.start()
            s_end.close()
            running.append((p, r_end, v, att))
        still, progressed = [], False
        for p, conn, v, att in running:
            r = None
            if conn.poll():
                try:
                    r = conn.recv()
                except EOFError:
                    r = None
                p.join()
            elif p.is_alive():
                still.append((p, conn, v, att))
                continue
            else:
                p.join()
            conn.close()
            progressed = True
            if r is None or "error" in r:
                if att == 0:
                    queue.append((v, 1))
                else:                          # count the whole chunk as failed
                    nb = sum(len(c[0]) for c in v.certificate) + sum(len(c[0]) for c in v.cert4)
                    total["failed"][0] = total["failed"].get(0, 0) + max(1, nb)
                continue
            for key in ("verified", "failed"):
                for rr, c in r[key].items():
                    total[key][rr] = total[key].get(rr, 0) + c
            total["volume_ratio"] += r["volume_ratio"]
        running = still
        if not progressed:
            time.sleep(0.05)
    return total
