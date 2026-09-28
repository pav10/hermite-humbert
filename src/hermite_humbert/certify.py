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
        self.vecs, self.A1, self.B1, self.A2, self.B2 = P.candidate_vectors(self.C)
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
def verify_certificate(bb, threshold, exp_ulps=8):
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
    # rule 4 (corner bound), re-checked in interval arithmetic (mpmath.iv)
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
            for b in range(len(lo)):
                c = 0.5 * (lo[b] + hi[b])                    # any expansion point in the box
                w = [iv.mpf(max(hi[b][j] - c[j], c[j] - lo[b][j])) for j in range(4)]
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
    """Worker: run the B&B of _PAR['bb'] on one block of initial boxes."""
    lo, hi, balls = args
    bb = _PAR["bb"]
    bb.gamma_low = _PAR["gamma_low"]
    bb.overflow = False
    bb.run(init=(lo, hi), balls=balls, **_PAR["kw"])
    keep = _PAR["kw"].get("keep_certificate", False)
    return {"certificate": bb.certificate if keep else None, "cert4": bb.cert4 if keep else [],
            "balls": bb.balls, "unresolved": bb.unresolved, "stats": bb.stats,
            "counterexample": bb.counterexample, "overflow": bb.overflow,
            "gamma_low": bb.gamma_low, "best_point": bb.best_point, "t_hi": bb.t_hi,
            "n_rule4": bb.n_rule4}


def run_parallel(bb, jobs, grid=(8, 8, 2, 2), **kw):
    """bb.run(**kw) split over `jobs` processes: the initial grid (finer than the
    serial one) is dealt out block by block to a process pool; certified balls
    found by finished blocks are handed to the blocks started later.  The
    results (certificate, balls, unresolved boxes, statistics) are merged into
    bb exactly as a serial run would leave them, so verify_certificate(bb, ...)
    applies unchanged (the union of the blocks tiles the region).  Stops early
    at the first counterexample / overflow.  Needs the 'fork' start method."""
    import multiprocessing as mp
    t0 = time.time()
    exact = kw.get("exact")
    thr_exact = _below(exact[0]) if exact is not None else None
    t_hi = 0.5 * math.log(1.0 / (bb.gamma_low if exact is None else thr_exact))
    lo, hi = bb.initial_boxes(t_hi, n=grid)
    blocks = [(lo[i:i + 1], hi[i:i + 1]) for i in range(len(lo))]
    _PAR.clear()
    _PAR.update(bb=bb, gamma_low=bb.gamma_low, kw=kw)
    balls = []
    cert, cert4, unresolved = ([] if kw.get("keep_certificate") else None), [], []
    counts = np.zeros(5, dtype=np.int64)
    boxes, n_rule4, stop = 0, 0, None
    best = (bb.gamma_low, bb.best_point)
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        pending = []
        it = iter(blocks)

        def submit():
            b = next(it, None)
            if b is not None:
                pending.append(pool.apply_async(_par_task, ((b[0], b[1], list(balls)),)))

        for _ in range(2 * jobs):
            submit()
        while pending:
            done = [p for p in pending if p.ready()]
            if not done:
                time.sleep(0.02)
                continue
            pending.remove(done[0])
            r = done[0].get()
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
            if r["counterexample"] is not None or r["overflow"] or (
                    exact is not None and r["gamma_low"] > exact[0] * (1 + 1e-12)):
                stop = r
                pool.terminate()
                break
            submit()
    bb.balls = balls
    bb.certificate = cert
    bb.cert4 = cert4
    bb.unresolved = unresolved
    bb.t_hi = t_hi
    bb.n_rule4 = n_rule4
    bb.gamma_low, bb.best_point = best
    bb.counterexample = stop["counterexample"] if stop else None
    bb.overflow = bool(stop and stop["overflow"])
    if stop is not None:            # the remaining blocks count as unresolved
        unresolved.append((lo, hi, np.full(len(lo), np.inf)))
    un_ub = max([float(np.max(u)) for _, _, u in unresolved], default=-np.inf)
    bb.stats = {"boxes": boxes, "discarded_by_rule": counts[:4].tolist() + [n_rule4],
                "time": time.time() - t0, "unresolved": int(sum(len(u) for _, _, u in unresolved)),
                "vectors": len(bb.vecs), "jobs": jobs}
    if exact is not None:
        bb.stats["balls"] = len(balls)
    base = bb.gamma_low * (1 + bb.delta) * (1 + bb.safety) if exact is None else float(exact[0])
    bb.gamma_upper = max(base, un_ub * (1 + bb.safety)) if not bb.overflow else float("inf")
    _PAR.clear()
    return bb.gamma_upper
