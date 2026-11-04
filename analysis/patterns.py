"""Patterns in the perfect binary Humbert forms of results/perfect/ (multiplicative
normalisation) and in the certified gamma_{K,2} of results/gamma_K2.tsv.

    python analysis/patterns.py            # prints every number quoted in paper/

Read-only on the result files.  zeta_K(-1) by Siegel's formula (Hirzebruch, Enseign. Math. 1973, section 1.4)
zeta_K(-1) = (1/60) sum_{b^2 < d, b = d mod 2} sigma_1((d - b^2)/4), d = disc(K).
"""
import csv
import glob
import json
import math
import os
from collections import Counter
from fractions import Fraction

import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hermite_humbert.field import RealQuadraticField      # noqa: E402
from hermite_humbert.ideals import class_group_charts    # noqa: E402


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


def h_imag(D):
    """class number of the imaginary quadratic discriminant D < 0 (reduced primitive forms)"""
    D, n, a = -D, 0, 1
    while 3 * a * a <= D:
        for b in range(-a + 1, a + 1):
            if (b * b + D) % (4 * a):
                continue
            c = (b * b + D) // (4 * a)
            if c >= a and not (c == a and b < 0) and math.gcd(math.gcd(a, abs(b)), c) == 1:
                n += 1
        a += 1
    return n


def kronecker(D, n):
    """Kronecker symbol (D/n), D a fundamental discriminant, n >= 1"""
    r = 1
    while n % 2 == 0:
        n //= 2
        if D % 2 == 0:
            return 0
        r *= 1 if D % 8 in (1, 7) else -1
    a = D % n if n > 1 else 0
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                r = -r
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            r = -r
        a %= n
    return r if n == 1 else 0


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
    zeta2 = 4 * math.pi ** 4 * z / d ** 1.5          # functional equation: zeta_K(-1) = d^{3/2} zeta_K(2) / (4 pi^4)
    th = g / (zeta2 * d / (math.pi ** 2 * h * R))    # Thunder 1998, Thm 2 (n = 2, d = 1): gamma >= zeta_K(2) d / (pi^2 h Reg)
    th1 = g / (zeta2 * d / (math.pi ** 2 * R))       # the same without the factor 1/h
    print("  lower bound zeta_K(2) d/(pi^2 h Reg) [Thunder]: gamma/bound min %.3f (D=%d); h > 1: min %.3f;"
          " without 1/h, h > 1: min %.3f" % (th.min(), Ds[int(np.argmin(th))], th[h > 1].min(), th1[h > 1].min()))

    def least_ideal_norm(D, dK):                    # B(K): least norm of a nontrivial integral ideal
        best = 4 if D % 8 == 5 else 2               # 2 inert iff D = 5 mod 8
        for p in (3, 5, 7):
            l = pow(D % p, (p - 1) // 2, p) if D % p else 0
            best = min(best, p if l != p - 1 else p * p)
        return best
    Bk = np.array([least_ideal_norm(D, int(T[D]["d_K"])) for D in Ds], float)
    m5 = np.array(Ds) != 5
    print("  gamma >= B(K) [Coulangeon-Icaza-O'Ryan 2007, Prop. 4.1 when L(K) < 4]: min gamma/B(K), D != 5: %.3f (D=%d)"
          % ((g / Bk)[m5].min(), np.array(Ds)[m5][int(np.argmin((g / Bk)[m5]))]))
    NK = {D: max(c.c.norm() for c in class_group_charts(RealQuadraticField(D))[0]) for D in Ds if T[D]["h_K"] != "1"}
    r = sorted((float(T[D]["gamma"]) / float(NK[D]) ** 2, D) for D in NK)
    print("  gamma >= N(I)N(J) = N(K)^2 for h > 1 [Baeza-Icaza 2004, proof of Cor. 3.2]: %d fields, min gamma/N(K)^2 %.3f (D=%d),"
          " N(K) up to %d" % (len(r), r[0][0], r[0][1], max(NK.values())))
    # imaginary quadratic, h = 1: gamma_K^2 >= w zeta_K(2) |D| / (2 pi^2) (cusp volume + Grunewald-Kuehnlein;
    # Thunder r = 0, s = 1), against the values of Chan-Icaza-Lauret 2015, Table 1
    cil = {3: 1.5, 4: 2, 7: 7 / 3, 8: 4, 11: 5.5, 19: 9.5, 43: 21.5, 67: 33.5, 163: 81.5}   # gamma_K^2
    rat = []
    for a_, g2 in cil.items():
        Dn = -a_
        L2 = sum(kronecker(Dn, k) / k ** 2 for k in range(1, 200000))
        w = {3: 6, 4: 4}.get(a_, 2)
        rat.append((math.sqrt(g2 / (w * math.pi ** 2 / 6 * L2 * a_ / (2 * math.pi ** 2))), Dn))
    print("  imaginary quadratic, h = 1 [CIL Table 1]: gamma/volume bound from %.3f (D=%d) to %.3f (D=%d)"
          % (min(rat) + max(rat)))
    print("  upper bounds: max gamma/(d/2) = %.3f [Cohn], max gamma/(16 d/pi^2) = %.3f [Icaza 1997, Thm 1]"
          % ((g / (d / 2)).max(), (g / (16 * d / math.pi ** 2)).max()))
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

    n3 = a3 = nf = 0
    for D in Ds:                                    # Prestel, as quoted in Hirzebruch 1973, p. 198: order-3 elliptic points of PSL_2(O_K)
        if D < 7 or math.gcd(D, 6) != 1:
            continue
        nf += 1
        a3 += h_imag(-3 * D) if D % 4 == 1 else h_imag(-12 * D)
        n3 += sum(1 for c in P[D]["classes"] if c["aut"] % 3 == 0)
    print("  elliptic points of order 3, %d fields with (D,6)=1, D>=7: %d perfect classes with 3 | |Aut|, Prestel's a_3 sum %d"
          % (nf, n3, a3))

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
