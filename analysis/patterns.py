"""Patterns in the perfect binary Humbert forms of results/perfect/ (multiplicative
normalisation) and in the certified gamma_{K,2} of results/gamma_K2.tsv.

    python analysis/patterns.py            # prints every number quoted in paper/

Read-only on the result files.  zeta_K(-1) by Siegel's formula
zeta_K(-1) = (1/60) sum_{b^2 < d, b = d mod 2} sigma_1((d - b^2)/4), d = disc(K).
"""
import csv
import glob
import json
import math
import os
from collections import Counter
from fractions import Fraction

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def sigma1(n):
    s, k = 0, 1
    while k * k <= n:
        if n % k == 0:
            s += k + (n // k if k * k != n else 0)
        k += 1
    return s


def zeta_m1(d):
    r = math.isqrt(d)
    return Fraction(sum(sigma1((d - b * b) // 4) for b in range(-r, r + 1)
                        if (b - d) % 2 == 0 and b * b < d), 60)


def unit(s):
    """fundamental unit from the closed form of gamma_K2.tsv, e.g. '(1 + sqrt(5))/2'"""
    s = s.replace(" ", "")
    half = s.startswith("(")
    if half:
        s = s[1:s.index(")")]
    a, b = s.split("+")
    c, r = b.split("sqrt(")
    v = float(a) + float(c or 1) * math.sqrt(float(r.rstrip(")")))
    return v / 2 if half else v


def fit(y, *xs):
    """least squares log y = sum c_i log x_i + c_0; returns (coefficients, R^2)"""
    A = np.vstack([np.log(x) for x in xs] + [np.ones(len(y))]).T
    c, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    r = np.log(y) - A @ c
    return c, 1 - r.var() / np.log(y).var()


def is_norm(k, form):
    m = math.isqrt(k) + 2
    return any(form(a, b) == k for a in range(m) for b in range(m))


def main():
    T = {int(r["D"]): r for r in csv.DictReader(open(os.path.join(ROOT, "results/gamma_K2.tsv")), delimiter="\t")}
    P = {}
    for f in glob.glob(os.path.join(ROOT, "results/perfect/D*.json")):
        p = json.load(open(f))
        P[p["D"]] = p
    Ds = sorted(P)
    assert all(P[D]["status"] == "COMPLETE" for D in Ds)
    d = np.array([int(T[D]["d_K"]) for D in Ds], float)
    h = np.array([int(T[D]["h_K"]) for D in Ds], float)
    R = np.array([math.log(unit(T[D]["eps0"])) for D in Ds])
    z = np.array([float(zeta_m1(int(T[D]["d_K"]))) for D in Ds])
    g = np.array([float(T[D]["gamma"]) for D in Ds])
    n = np.array([P[D]["n_perfect"] for D in Ds], float)
    e = np.array([P[D]["n_extreme"] for D in Ds], float)
    w = np.array([P[D]["n_weakly_eutactic"] for D in Ds], float)
    cls = [c for D in Ds for c in P[D]["classes"]]

    print("== totals")
    print("fields %d, perfect %d, extreme %d, weakly eutactic %d (in %d fields), not eutactic %d"
          % (len(Ds), n.sum(), e.sum(), w.sum(), (w > 0).sum(), sum(P[D]["n_not_eutactic"] for D in Ds)))
    print("gamma_max = certified gamma in all fields:", all(P[D]["gamma_max_matches_certified"] for D in Ds))
    print("single perfect class:", [D for D in Ds if P[D]["n_perfect"] == 1],
          "; most:", max((P[D]["n_perfect"], D) for D in Ds))
    print("extreme/perfect: median %.3f" % np.median(e / n))
    print("zeta_K(-1) check:", [str(zeta_m1(x)) for x in (5, 8, 12, 13)], "(1/30, 1/12, 1/6, 1/6)")

    print("\n== counts against field invariants (log-log least squares, d >= 100: %d fields)" % (d >= 100).sum())
    m = d >= 100
    for lab, x in [("zeta_K(-1)", z), ("zeta_K(-1) Reg", z * R), ("d", d), ("Reg", R), ("h Reg", h * R)]:
        c, r2 = fit(n[m], x[m])
        c2, r22 = fit(e[m], x[m])
        print("  n_perfect ~ %-15s slope %.3f R2 %.3f   n_extreme slope %.3f R2 %.3f" % (lab, c[0], r2, c2[0], r22))
    big = d > 200
    q = n / z
    print("  n/zeta_K(-1), d > 200 (%d fields): min %.2f  quartiles %.2f %.2f %.2f  max %.2f"
          % ((big.sum(),) + tuple(np.percentile(q[big], [0, 25, 50, 75, 100]))))
    mass = np.array([sum(1 / c["aut"] for c in P[D]["classes"]) for D in Ds])
    print("  mass sum 1/|Aut| / zeta_K(-1), d > 200: median %.2f range [%.2f, %.2f]"
          % (np.median((mass / z)[big]), (mass / z)[big].min(), (mass / z)[big].max()))

    print("\n== gamma_{K,2}")
    for lab, xs in [("d", (d,)), ("d, Reg", (d, R)), ("d, Reg, h", (d, R, h)), ("d, h Reg", (d, h * R))]:
        c, r2 = fit(g, *xs)
        print("  log gamma ~ %-10s coefficients %s  R2 %.3f" % (lab, np.round(c, 3), r2))
    print("  gamma/d: min %.4f (D=%d)  max %.4f (D=%d); Cohn: gamma <= d/2"
          % ((g / d).min(), Ds[int(np.argmin(g / d))], (g / d).max(), Ds[int(np.argmax(g / d))]))
    c, _ = fit(g, d, R)
    res = np.log(g) - (c[0] * np.log(d) + c[1] * np.log(R) + c[2])
    D = np.array(Ds)
    for lab, mm in [("2 split (D=1 mod 8)", D % 8 == 1), ("2 inert (D=5 mod 8)", D % 8 == 5), ("2 ramified", D % 4 != 1)]:
        print("  residual of (d, Reg) fit, %-20s n=%3d  mean %+.3f" % (lab, mm.sum(), res[mm].mean()))

    print("\n== classes")
    print("  #minimal classes:", sorted(Counter(c["n_min"] for c in cls).items()))
    print("  |Aut|:", sorted(Counter(c["aut"] for c in cls).items()))
    cp = Counter(c["copies_found"] for c in cls)
    print("  copies found per class: mean %.2f, found once %.3f" % (sum(k * v for k, v in cp.items()) / len(cls), cp[1] / len(cls)))
    loe = lambda a, b: a * a + a * b + b * b
    two = lambda a, b: a * a + b * b
    for orders, form, lab, scale in [((3, 6, 24), loe, "a^2+ab+b^2", Fraction(3, 4)), ((4,), two, "a^2+b^2", Fraction(1))]:
        ok = bad = 0
        for c in cls:
            if c["aut"] not in orders:
                continue
            try:
                k = Fraction(c["closed_form"]) * scale
            except (TypeError, ValueError):
                k = None
            if k is not None and k.denominator == 1 and is_norm(int(k), form):
                ok += 1
            else:
                bad += 1
        print("  |Aut| in %s: %d classes with %s*gamma = %s (integer), %d others" % (orders, ok, scale, lab, bad))

    print("\n== search statistics")
    st = [s for D in Ds for s in P[D]["stats"]]
    tot = {k: sum(s[k] for s in st) for k in ("boxes", "leaves", "newton", "vertices")}
    print("  boxes %.3g, leaves %.3g, Newton runs %.3g, polished vertices %.3g" % tuple(tot[k] for k in ("boxes", "leaves", "newton", "vertices")))
    boxes = np.array([sum(s["boxes"] for s in P[D]["stats"]) for D in Ds], float)
    c, r2 = fit(boxes[m], z[m])
    print("  boxes ~ zeta_K(-1)^%.2f (R2 %.2f)" % (c[0], r2))
    print("  longest field: %ds (D = %d)" % max((P[D]["seconds"], D) for D in Ds))


if __name__ == "__main__":
    main()
