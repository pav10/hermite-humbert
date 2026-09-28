"""Structural features of the critical forms, for pattern hunting."""
import sys, json, math, time
import mpmath
from .field import RealQuadraticField
from .core import _problems
from .vertex import VertexFinder
from .analysis import classify_global, identify, normal_form

def kron(d, n):
    """Kronecker symbol (d/n), d a fundamental discriminant, n >= 1."""
    from sympy import jacobi_symbol
    r, m = 1, n
    while m % 2 == 0:
        m //= 2
        r *= 0 if d % 2 == 0 else (1 if d % 8 in (1, 7) else -1)
    return r * (jacobi_symbol(d % m, m) if m > 1 else 1)

def zetaK2(d):
    mpmath.mp.dps = 20
    chi = [kron(d, n) if math.gcd(n, d) == 1 else 0 for n in range(d)]
    chi[0] = 0
    return mpmath.zeta(2) * mpmath.dirichlet(2, chi)

def features(D, gamma_cert):
    F = RealQuadraticField(D)
    probs, h = _problems(D)
    VFs = [VertexFinder(P) for P in probs]
    verts = []
    for ns, nd in ((2000, 16), (4000, 32), (8000, 64)):
        for k, VF in enumerate(VFs):
            for v in VF.multistart(n_samples=ns, n_seeds=nd, dps=60):
                v["chart"] = k
                verts.append(v)
        if any(abs(v["gamma"] - gamma_cert) < 1e-15 * gamma_cert for v in verts):
            break
    cl = classify_global(probs, verts)
    cl.sort(key=lambda c: -c["rep"]["gamma"])
    top = [c for c in cl if abs(c["rep"]["gamma"] - gamma_cert) < 1e-15 * gamma_cert]
    if not top:
        raise RuntimeError("critical form not found for D=%d" % D)
    c0 = top[0]["rep"]
    # norms of the chart representatives, all classes (inverse pairs counted twice)
    sN = 0.0
    for P in probs:
        n = 1 if P.chart is None else float(P.chart.c.norm())
        mult = 1 if (P.chart is None or P.chart.inverse_index == P.chart.index) else 2
        sN += mult / n ** 2
    z2 = float(zetaK2(F.disc))
    le = math.log(F.eps0())
    vb = F.disc * z2 / (math.pi ** 2 * le * sN)
    with mpmath.workdps(60):
        g = c0["gamma"]
        rat = identify(g, max_degree=1, dps=60)[0] is not None
        rat2 = identify(g ** 2, max_degree=1, dps=60)[0] is not None
    # unimodular pair of minimal vectors in O_K^2 coordinates (h = 1 meaning)
    Mok = c0["vectors_OK"]
    unimod = any(F.is_unit(F.sub(F.mul(u[0], w[1]), F.mul(u[1], w[0])))
                 for i, u in enumerate(Mok) for w in Mok[i + 1:])
    return {"D": D, "d": F.disc, "h": h, "Neps": F.unit_norm(), "log_eps": le,
            "gamma": float(gamma_cert), "gamma_rational": rat, "gamma2_rational": rat2,
            "n_min": len(c0["vectors"]), "n_crit_classes": len(top),
            "crit_self_conj": [c["self_conjugate"] for c in top],
            "n_extreme_classes": len(cl), "unimodular_pair": unimod,
            "gamma_over_half_d": float(gamma_cert) / (F.disc / 2), "vol_bound": vb,
            "gamma_over_vol_bound": float(gamma_cert) / vb, "zetaK2": z2, "sumN": sN}

if __name__ == "__main__":
    # python -m hermite_humbert.features OUT.jsonl RESULTS.tsv D [D ...]
    import csv
    vals = {}
    with open(sys.argv[2]) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["status"].startswith("certified"):
                vals[int(row["D"])] = mpmath.mpf(row["gamma"])
    out = []
    for D in [int(x) for x in sys.argv[3:]]:
        t0 = time.time()
        f = features(D, vals[D])
        f["seconds"] = round(time.time() - t0, 1)
        out.append(f)
        print(json.dumps(f), flush=True)
        with open(sys.argv[1], "a") as fh:
            fh.write(json.dumps(f) + "\n")
