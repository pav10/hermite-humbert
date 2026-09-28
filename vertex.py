"""
Extreme binary Humbert forms as vertices of the function G on H^2.

phi_v = log f_v.  In coordinates q = (x1, x2, u1, u2), u_j = log y_j,
    phi_v = log Q1 + log Q2 - u1 - u2,   Q_j = (a_j + b_j x_j)^2 + b_j^2 e^{2 u_j}.

Coulangeon's Voronoi theorem (extreme <=> perfect and eutactic) in these
coordinates: with M = minimal classes at z,
  perfect   <=> the differences grad phi_v - grad phi_w (v, w in M) span R^4
            <=> dim span{X_v} = 5, X_v = (v v^t / S1[v], v' v'^t / S2[v'])  [PW05 Sec. 2]
  eutactic  <=> S^{-1} = sum_v rho_v X_v with all rho_v > 0
            <=> sum_v rho_v grad phi_v = 0 with all rho_v > 0.
(The gradient of log S[v] - (1/2) log det S on the form space is X_v - S^{-1}/2;
the chart z -> S(z) is transversal to the 2-dim scaling orbits, along which all
these gradients vanish.)  Necessarily sum rho_v = 2 (take traces against S).
"""
import itertools
import math

import numpy as np
import mpmath
from scipy.optimize import linprog


class VertexFinder:
    def __init__(self, problem):
        self.P = problem
        if problem.V is None:
            problem.build_V(problem.d / 2.0)

    # ----------------------------------------------------------------- float
    def phi_grad(self, p, idx=None):
        P = self.P
        A1, B1, A2, B2 = P.VA1, P.VB1, P.VA2, P.VB2
        if idx is not None:
            A1, B1, A2, B2 = A1[idx], B1[idx], A2[idx], B2[idx]
        x1, x2, y1, y2 = p
        L1, L2 = A1 + B1 * x1, A2 + B2 * x2
        T1, T2 = (B1 * y1) ** 2, (B2 * y2) ** 2
        Q1, Q2 = L1 * L1 + T1, L2 * L2 + T2
        phi = np.log(Q1) + np.log(Q2) - math.log(y1) - math.log(y2)
        g = np.stack([2 * B1 * L1 / Q1, 2 * B2 * L2 / Q2, 2 * T1 / Q1 - 1, 2 * T2 / Q2 - 1], axis=1)
        return phi, g

    def _eval(self, p):
        """log G at p after moving p into the exact region (G is invariant)."""
        P = self.P
        p = P.to_region(p)
        if not P.in_exact_region(p, tol=0.0):
            if not P.use_primitive:          # h_K > 1: stay inside this chart's region
                return p, -np.inf
            p = P.reduce_to_top(p)
        phi, _ = self.phi_grad(p)
        return p, float(np.min(phi))

    def ascend(self, p0, r0=0.05, max_iter=2000):
        """Trust-region SLP ascent of min_v phi_v.  Returns the final point
        (in the exact region) and the near-active index set."""
        p, m = self._eval(np.array(p0, dtype=float))
        r = r0
        for _ in range(max_iter):
            phi, g = self.phi_grad(p)
            m = float(np.min(phi))
            near = phi <= m + 1.0
            gmax = float(np.max(np.abs(g[near]).sum(axis=1)))
            tau = max(1e-12, 2.0 * r * gmax)
            A = np.where(phi <= m + tau)[0]
            # LP: max eta s.t. eta <= phi_v + g_v . delta,  |delta_i| <= r
            c = np.zeros(5)
            c[4] = -1.0
            A_ub = np.hstack([-g[A], np.ones((len(A), 1))])
            b_ub = phi[A]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(-r, r)] * 4 + [(None, None)], method="highs")
            if res.status != 0:
                r *= 0.25
                if r < 1e-15:
                    break
                continue
            delta, eta = res.x[:4], res.x[4]
            pred = eta - m
            if pred < 1e-15:
                break
            q = np.array([p[0] + delta[0], p[1] + delta[1], p[2] * math.exp(delta[2]), p[3] * math.exp(delta[3])])
            qn, mn = self._eval(q)
            rho = (mn - m) / pred
            if rho > 0.1:
                p = qn
                if rho > 0.75 and np.max(np.abs(delta)) > 0.9 * r:
                    r = min(2 * r, 0.5)
            else:
                r *= 0.25
                if r < 1e-15:
                    break
        phi, g = self.phi_grad(p)
        m = float(np.min(phi))
        A = np.where(phi <= m + max(1e-9, 1e3 * r))[0]
        return p, A

    def newton_float(self, p, A, iters=30, max_subsets=120):
        """Newton on phi_vi - phi_v0 = 0 (4 equations) for 5-subsets of the
        near-active set, tried in order of activity (then conditioning); a
        subset is accepted only if at the Newton solution its 5 vectors are
        minimal among all of V (otherwise it converged to an intersection that
        is not a vertex of G)."""
        phi, g = self.phi_grad(p, A)
        if len(A) < 5:
            return p, None
        m = float(np.min(phi))
        order = np.argsort(phi)[:min(len(A), 14)]
        subs = []
        for sub in itertools.combinations(order, 5):
            J = g[list(sub[1:])] - g[sub[0]]
            smin = np.linalg.svd(J, compute_uv=False)[-1]
            if smin > 1e-8:
                subs.append((float(np.max(phi[list(sub)]) - m), -smin, sub))
        subs.sort()
        for _, _, sub in subs[:max_subsets]:
            idx = A[list(sub)]
            q = np.array([p[0], p[1], math.log(p[2]), math.log(p[3])])
            ok = False
            for _ in range(iters):
                pp = np.array([q[0], q[1], math.exp(q[2]), math.exp(q[3])])
                ph, gg = self.phi_grad(pp, idx)
                Fv = ph[1:] - ph[0]
                J = gg[1:] - gg[0]
                try:
                    dq = np.linalg.solve(J, -Fv)
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(dq)) or np.max(np.abs(dq)) > 0.2:
                    break                    # left the basin
                q = q + dq
                if np.max(np.abs(dq)) < 1e-15:
                    ok = True
                    break
            else:
                ok = np.max(np.abs(dq)) < 1e-10
            if not ok:
                continue
            pp = np.array([q[0], q[1], math.exp(q[2]), math.exp(q[3])])
            ph_all, _ = self.phi_grad(pp)
            if np.max(ph_all[idx]) - np.min(ph_all) < 1e-10:
                return pp, idx
        return p, None

    # ------------------------------------------------------ high precision
    def _emb_mp(self, v):
        F = self.P.F
        a1, a2 = F.emb_mp(v[0])
        b1, b2 = F.emb_mp(v[1])
        return a1, b1, a2, b2

    def phi_grad_mp(self, q, v):
        x1, x2, u1, u2 = q
        a1, b1, a2, b2 = self._emb_mp(v)
        y1, y2 = mpmath.exp(u1), mpmath.exp(u2)
        L1, L2 = a1 + b1 * x1, a2 + b2 * x2
        T1, T2 = (b1 * y1) ** 2, (b2 * y2) ** 2
        Q1, Q2 = L1 ** 2 + T1, L2 ** 2 + T2
        phi = mpmath.log(Q1) + mpmath.log(Q2) - u1 - u2
        g = [2 * b1 * L1 / Q1, 2 * b2 * L2 / Q2, 2 * T1 / Q1 - 1, 2 * T2 / Q2 - 1]
        return phi, g

    def polish(self, p, idx, dps=60):
        """Newton at dps digits on the 5 chosen active vectors, then collect the
        full minimal set at that precision.  Returns a vertex dict or None."""
        P = self.P
        vecs = [P.V[i] for i in idx]
        old = mpmath.mp.dps
        mpmath.mp.dps = dps + 10
        try:
            q = mpmath.matrix([p[0], p[1], math.log(p[2]), math.log(p[3])])
            for _ in range(60):
                vals = [self.phi_grad_mp(q, v) for v in vecs]
                Fv = mpmath.matrix([vals[i][0] - vals[0][0] for i in range(1, 5)])
                J = mpmath.matrix([[vals[i][1][k] - vals[0][1][k] for k in range(4)] for i in range(1, 5)])
                dq = mpmath.lu_solve(J, -Fv)
                q = q + dq
                if mpmath.norm(dq, mpmath.inf) < mpmath.mpf(10) ** (-(dps + 5)):
                    break
            else:
                return None
            # shortlist with floats, confirm with mp
            pf = np.array([float(q[0]), float(q[1]), float(mpmath.exp(q[2])), float(mpmath.exp(q[3]))])
            phi, _ = self.phi_grad(pf)
            m = float(np.min(phi))
            short = np.where(phi <= m + 1e-6)[0]
            vals = {int(i): self.phi_grad_mp(q, P.V[int(i)]) for i in short}
            mmin = min(v[0] for v in vals.values())
            tol = mpmath.mpf(10) ** (-(dps - 10))
            active = sorted(i for i, v in vals.items() if v[0] - mmin < tol)
            if any(i not in active for i in idx):
                return None
            rest = np.delete(phi, active) if len(active) < len(phi) else np.array([np.inf])
            gap = float(np.min(rest) - m)
            vert = {
                "q": q, "dps": dps, "p": pf, "gamma": mpmath.exp(mmin), "active": active,
                "vectors": [P.V[i] for i in active],
                "grads": [vals[i][1] for i in active], "gap": gap,
            }
            self.analyze(vert)
            return vert
        finally:
            mpmath.mp.dps = old

    def analyze(self, vert):
        """Perfection rank and eutaxy (Coulangeon), in the form space."""
        P = self.P
        q = vert["q"]
        x1, x2, y1, y2 = q[0], q[1], mpmath.exp(q[2]), mpmath.exp(q[3])
        S = [mpmath.matrix([[1, x], [x, x * x + y * y]]) / y for x, y in ((x1, y1), (x2, y2))]
        Sinv = [mpmath.inverse(s) for s in S]
        rows = []
        for v in vert["vectors"]:
            a1, b1, a2, b2 = self._emb_mp(v)
            row = []
            for (a, b), s in zip(((a1, b1), (a2, b2)), S):
                val = s[0, 0] * a * a + 2 * s[0, 1] * a * b + s[1, 1] * b * b
                row += [a * a / val, a * b / val, b * b / val]
            rows.append(row)
        X = mpmath.matrix(rows)                      # |M| x 6
        sv = mpmath.svd_r(X, compute_uv=False)
        svf = sorted([float(abs(s)) for s in sv], reverse=True)
        rank = sum(1 for s in svf if s > 1e-25 * svf[0])
        target = [Sinv[0][0, 0], Sinv[0][0, 1], Sinv[0][1, 1], Sinv[1][0, 0], Sinv[1][0, 1], Sinv[1][1, 1]]
        # eutaxy: max tau s.t. X^T rho = target, rho >= tau
        n = len(rows)
        Xf = np.array([[float(c) for c in r] for r in rows])
        tf = np.array([float(c) for c in target])
        c = np.zeros(n + 1)
        c[-1] = -1.0
        A_eq = np.hstack([Xf.T, np.zeros((6, 1))])
        A_ub = np.hstack([-np.eye(n), np.ones((n, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=tf,
                      bounds=[(0, None)] * n + [(None, None)], method="highs")
        eut_margin = float(res.x[-1]) if res.status == 0 else float("-inf")
        rho = None
        if rank == 5 and n == 5:
            # unique coefficients (X_v a basis of their 5-dim span): least squares at full precision
            A = X.T                                   # 6 x 5, columns X_v
            t = mpmath.matrix(target)
            sol = mpmath.lu_solve(A.T * A, A.T * t)
            resid = mpmath.norm(A * sol - t)
            rho = [sol[i] for i in range(n)]
            eut_margin = float(min(rho)) if resid < mpmath.mpf(10) ** (-(vert["dps"] - 15)) else float("-inf")
        elif res.status == 0:
            rho = [mpmath.mpf(float(t)) for t in res.x[:n]]
        vert["perf_rank"] = rank
        vert["perfect"] = rank == 5
        vert["eutaxy_margin"] = eut_margin
        vert["eutactic"] = eut_margin > 1e-9
        vert["extreme"] = vert["perfect"] and vert["eutactic"]
        vert["rho"] = rho
        return vert

    # ----------------------------------------------------------- multistart
    def halton(self, n, dim=4, skip=20):
        primes = [2, 3, 5, 7, 11, 13][:dim]
        out = np.zeros((n, dim))
        for j, b in enumerate(primes):
            for i in range(n):
                k, f, r = i + skip, 1.0, 0.0
                while k > 0:
                    f /= b
                    r += f * (k % b)
                    k //= b
                out[i, j] = r
        return out

    def sample_region(self, n, t_hi=0.0):
        P = self.P
        H = self.halton(n)
        t_hi = max(t_hi, P.t_lo + 1e-3)
        r1 = -0.5 + H[:, 0]
        r2 = 0.5 * H[:, 1]
        s = P.s_min + (P.s_max - P.s_min) * H[:, 2]
        t = P.t_lo + (t_hi - P.t_lo) * H[:, 3]
        g1, g2 = P.g1, P.g2
        pts = np.stack([r1 * g1[0] + r2 * g2[0], r1 * g1[1] + r2 * g2[1], np.exp(t - s), np.exp(t + s)], axis=1)
        return pts

    def find_vertex(self, p0, dps=60):
        p, A = self.ascend(p0)
        for attempt in range(3):
            pn, idx = self.newton_float(p, A)
            if idx is not None:
                vert = self.polish(pn, idx, dps=dps)
                if vert is not None and vert["extreme"]:
                    return vert
            # restart from a small perturbation (stationary but not extreme)
            p, A = self.ascend(p * (1 + 1e-3 * np.random.default_rng(attempt).normal(size=4)), r0=1e-3)
        return None

    def multistart(self, n_samples=3000, n_seeds=30, dps=60, verbose=False):
        P = self.P
        pts = self.sample_region(n_samples)
        vals, _ = P.G_V(pts)
        order = np.argsort(-vals)
        seeds = []
        for i in order:
            rst = np.array(P.p_to_rst(pts[i]))
            if all(np.max(np.abs(rst - s[1]) / np.array([1, .5, P.s_max - P.s_min + 1e-9, 1])) > 0.08 for s in seeds):
                seeds.append((pts[i], rst))
            if len(seeds) >= n_seeds:
                break
        verts = []
        for k, (p0, _) in enumerate(seeds):
            v = self.find_vertex(p0, dps=dps)
            if v is None:
                continue
            if not any(abs(v["gamma"] - w["gamma"]) < 1e-30 * v["gamma"] and
                       np.allclose(v["p"], w["p"], atol=1e-9) for w in verts):
                verts.append(v)
            if verbose:
                print("  seed %2d: gamma = %s" % (k, mpmath.nstr(v["gamma"], 15)))
        verts.sort(key=lambda v: -v["gamma"])
        return verts


