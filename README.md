# Hermite–Humbert constants of real quadratic fields (binary forms)

Computes and certifies the Hermite–Humbert constant γ_{K,2} of a real quadratic
field K = Q(√D) of any class number, together with the extreme forms, following
and extending Baeza–Coulangeon–Icaza–O'Ryan (2001) and Pohst–Wagner (2005, 2009).

## Installation

Requires Python ≥ 3.9 with numpy, scipy and mpmath (pure Python, no compiler needed).

```bash
git clone https://github.com/pav10/hermite-humbert.git
cd hermite-humbert
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e ".[test]"
```

Optional extras: `pip install -e ".[pari]"` (cypari2, used by one cross-check test),
`".[features]"` (sympy, for `hermite_humbert.features`).

## Quick start

```bash
hermite-humbert 13                            # full report for Q(sqrt 13)
hermite-humbert 7 11 17 --verify              # several fields + summary table
python -m hermite_humbert 13                  # same, without the console script

hermite-humbert-sweep out.tsv 2-100 -j 4      # parallel batch run over all squarefree D in [2, 100]
hermite-humbert-sweep out.tsv 46 91 --timeout 0   # no time limit; re-running resumes

pytest -m "not slow"                          # unit tests (~2 min)
pytest                                        # + full pipeline on the published fields
```

```python
from hermite_humbert import compute, gamma_of_form
r = compute(13)                     # value, closed form, extreme forms, certificate status
r["gamma"], r["classes"][0]["closed_form"][1], r["exact"]["proved"]
g, vecs, _ = gamma_of_form(5, S1, S2)   # gamma(S) and minimal vectors of any given form
```

CLI options: `--samples`, `--seeds` (multistart), `--dps` (working precision,
default 100), `--delta` (tolerance-mode accuracy), `--no-exact`, `--verify`
(record the certificate and re-check it with outward rounding), `--quiet`.

`hermite-humbert-sweep` writes one row per field to the TSV table as soon as the
field finishes (sorted by D; columns `D, d_K, h_K, eps0, gamma, closed_form, #M,
#classes, status, seconds, notes`) and the complete result of each field
(every extreme class with its normal form, minimal vectors, perfection rank,
eutactic coefficients, certificate statistics) to `fields/D<D>.json` next to it.
Certified fields are skipped when the command is repeated, so an interrupted
sweep can simply be restarted; `--timeout` (default 3600 s per field),
`--jobs`, `--field-jobs`, `--no-verify`, `--redo`.

**Checkpoints.**  Each field is checkpointed in `checkpoints/D<D>/` next to the
table (`--checkpoints DIR`, or `--no-checkpoint`): the multistart vertices and
every finished block of the branch-and-bound, each block's certificate already
re-checked in its own process.  A field that is interrupted (time limit, killed
process, machine restart) resumes from its finished blocks, so even repeated short
runs make progress (a field with a checkpoint is retried although it timed out
before); the directory is removed once the field is certified.  Checkpoints are
keyed by γ* (exact) and the options, so a changed γ* never reuses stale blocks.
`hermite-humbert D --checkpoint DIR` does the same for a single run.

## Normalisation

As in [BCIO], [PW05], [PW09] (Icaza's multiplicative constant): for
S = (S1, S2) with S_j positive definite 2×2 real,

    S[v] = S1[v^(1)] · S2[v^(2)],   m(S) = min_{0≠v∈O_K^2} S[v],
    γ(S) = m(S) / sqrt(det S1 · det S2),   γ_{K,2} = sup_S γ(S).

Elements are written in the basis 1, ω with ω = (1+√D)/2 (D ≡ 1 mod 4) or √D
(as in PARI); S1 belongs to the embedding with ω ↦ ω₁ > 0.  Forms are reported
in the Pohst–Wagner normal form ((1, b1; b1, c1), (1, b2; b2, c2)) with m(S) = 1,
e1 minimal, and e2 also minimal whenever a unimodular pair of minimal vectors
exists (then c1·c2 = 1).

## Method

1. **Forms as points of ℍ².** S_j ∝ M(z_j) = (1/y_j)[[1, x_j], [x_j, |z_j|²]], so
   γ(S) = G(z) := min_v f_v(z), f_v(z) = |a₁+b₁z₁|²|a₂+b₂z₂|²/(y₁y₂). G is invariant
   under GL₂(O_K) and the Galois swap; units act trivially on f_v. At a highest
   point of an orbit (e1 minimal) G = 1/(y₁y₂), so γ_{K,2} is the reciprocal height
   of the lowest point of the Hilbert modular fundamental domain ([BCIO, Prop. 1.1];
   [PW09, (2)]; Cohn's "floor").
2. **Search region (h_K = 1; see below for h_K > 1).** Every orbit has a highest point with
   x = r₁ + r₂ω (r₁ ∈ [−½, ½], r₂ ∈ [0, ½]), y₂/y₁ ∈ [1, ε₀], y₁y₂ ≥ 2/d:
   translations, z ↦ ε₀ᵏz, x ↦ −x and the Galois swap preserve G and y₁y₂, and
   Cohn's bound γ_{K,2} ≤ d/2 (= (γ₄/2)²·d by AM–GM on the trace form) bounds the height.
3. **Candidate vectors.** For a given C, an explicit finite set V containing
   every primitive v with f_v ≤ C somewhere on the region (|N(β)| ≤ √(Cd/2) and
   interval bounds for α/β); min_V f_v = G there whenever it is ≤ C.
   An independent routine computes m(S) and M(S) for any form (unit-balanced
   Fincke–Pohst on ℤ⁴, LLL-preconditioned).
4. **Extreme forms** = perfect + eutactic (Coulangeon) = vertices of G at which 0
   is a strictly positive combination of the gradients.  Found by trust-region LP
   ascent from quasi-random seeds, polished by Newton at 60 digits; perfection rank
   (dim span X_v = 5) and eutactic coefficients (S⁻¹ = Σρ_v X_v, ρ_v > 0, Σρ = 2) are
   computed.  Equivalence classes via [PW05, Alg. 11] in exact K-arithmetic.
5. **Certificate (branch-and-bound over the region).** A box B is discarded by
   - rule 0: min_v sup_B f_v ≤ threshold (exact product bound, convexity in y);
   - rule 1: some v has N|α+βz|² < 1 on B, so B contains no highest point;
   - rule 2 (exact mode): B lies in a certified ball around a maximiser, on which
     min_{v∈M} φ_v ≤ φ* − κh + Hh²/2 < φ*.  The ball lives in the scaled coordinates
     (x₁/y₁*, x₂/y₂*, log y₁, log y₂) of the vertex, where |φ_xx| ≤ 2/y²,
     |φ_xu| ≤ 8/(3√3·y), |φ_uu| ≤ 1 become y-independent: H ≤ (4e^{4ρ} + (128/27)e^{2ρ} + 1)^{1/2};
     κ = distance from 0 to the boundary of the convex hull of the scaled active gradients;
   - rule 3: B misses the translation parallelogram;
   - rule 4 (second-order "corner" bound): for λ in the simplex, min_i φ_i ≤ Σλ_iφ_i, and
     by Taylor with the same universal Hessian bounds,
     sup_B Σλ_iφ_i ≤ Σλ_iφ_i(c) + |G₁|w_{x1} + |G₂|w_{x2} + |G₃+G₄|w_t + |G₄−G₃|w_s + ½Σ_j(2w_{xj}²/y_j² + 2κw_{xj}w_u/y_j + w_u²),
     G = Σλ_i∇φ_i(c), κ = 8/(3√3), w_u = w_s + w_t.  λ is chosen by projected
     subgradient descent over the 6 smallest pieces at the centre.  Near crossings of
     several f_v this is second-order in the box width, where rules 0–1 are only
     first-order: e.g. 30M+ boxes (unfinished) → 56k boxes for the hardest chart of Q(√95).

   Exact mode (threshold γ*, run first) proves γ_{K,2} = γ*; tolerance mode
   (γ_{K,2} ≤ γ*(1+δ)) is the fallback.  A run stops at once if a box centre shows
   G > γ* (or G above the range its vector set is exact for), and the vertex search
   restarts from that point — so the branch-and-bound, not the multistart, is what
   finally locates the maximum (for Q(√227) the multistart missed it twice).  With `--verify` every discarded box is re-checked independently
   with outward rounding, and the boxes are checked to tile the region.
6. **Closed forms** by PSLQ at 60 digits (relation required to hold to ≥ 45 digits).

## What is and is not rigorous

- *Exact:* field arithmetic, fundamental unit, the h_K = 1 test (Minkowski bound),
  primitivity, SL₂ completion, Steinitz frames, equivalence of extreme forms.
- *Exact up to the float slack of a lattice enumeration:* the class group
  (ideals up to the Minkowski bound; principal-ideal test by exhaustive search for
  a balanced generator); cross-checked with PARI for all D < 300.
- *Rigorous up to floating point, independently re-checked with outward rounding*
  (IEEE round-to-nearest, exp accurate to 8 ulp): the branch-and-bound covering,
  rules 0–3, the tiling.  Rule 4 is re-checked by a vectorised double-precision
  evaluation that carries an explicit bound on the accumulated rounding error of
  every quantity (same model, plus log accurate to 4 ulp; `certify._rule4_float`),
  and every box it cannot confirm is re-checked in interval arithmetic (mpmath.iv,
  30 digits), with λ normalised by its interval sum.  A test checks, at thresholds
  down to the tight one, that the float check never accepts a box the interval
  check rejects; it confirms essentially all boxes and is ~200× faster, which
  matters for fields with many rule-4 boxes.  (An earlier version of rule 4
  lost Σλ = 1 at the 10⁻⁸ level through cancellation, which made it unsound on tiny
  boxes; it was caught by a disagreement with an exact-mode run, fixed, and the
  interval re-check would reject such a certificate.  All results in
  `results/results_h1_D_lt_50.tsv` and `results/results_h_gt_1_D_lt_100.tsv` predate rule 4.)
- *Numerical, not formally verified:* the maximising vertex (Newton at 60 digits,
  no interval Newton/Kantorovich), the ball radius (κ via qhull in double precision
  from 60-digit gradients, safety factor 0.9), and the closed forms (PSLQ, not
  symbolic).  The candidate-vector enumeration uses floats with 1e-9 slack.
- *Not certified:* the lists of non-critical extreme forms (multistart only).
- *Scope:* any class number; ε₀ ≤ 10¹² (double precision); the cost grows
  roughly like √ε₀ (D = 46 and D = 91, ε₀ ≈ 4.9·10⁴ and 3.1·10³, did not finish
  within the 5-minute limit of the development sandbox).

## Performance and reach

Measured on one core (Python/numpy):

- Candidate vectors: |V| ≈ 0.86·d^1.41·ε₀^0.41 (137 fields, D < 300).
- Typical run time ≈ 0.06·d^0.9·(log ε₀)^0.7 s within a factor ≈ 1.6 (without rule 4;
  faster now), e.g. D = 101, 197, 257 (h = 3) in 4, 6, 16 s; Q(√227) (d = 908) in 80 s.
- The cost is heavy-tailed and not predicted by (d, ε₀): near-degenerate maxima
  (now handled by rule 4 and scaled balls) and extreme unit skew (Q(√46), ε₀ ≈ 4.9·10⁴:
  x₁ must be resolved at scale y₁ ≈ 10⁻³; certified with 9.4·10⁷ boxes in 26 min on
  4 cores, before the split-priority fix of "Open fields" below).
- Hard limit: ε₀ ≤ 10¹² (double precision).

Estimate (not a guarantee): the current code handles most D up to a few hundred with
ε₀ ≲ 10³ in minutes; large-unit fields need hours.  Remaining improvements, in order
of expected payoff: a parallel vertex search over slabs, a compiled kernel for the
per-box work (numba), extended precision for ε₀ > 10¹² (parallel blocks and
checkpoint/resume are implemented).

## Class number > 1 (Steinitz charts)

If v ∈ O_K² has content 𝔠′ = αO_K + βO_K, then O_K² = 𝔠′⁻¹v ⊕ 𝔠′w with
w = (γ, δ), γ, δ ∈ 𝔠′⁻¹, αδ − βγ = 1.  The content of a minimal vector has
minimal norm in its ideal class (otherwise λv, λ ∈ 𝔠′⁻¹, |N(λ)| < 1, is
shorter).  Fixing an integral ideal 𝔠 of minimal norm in each class, 𝔠′ = μ𝔠 with
|N(μ)| = 1, and in the basis (μ⁻¹v, μw) the lattice becomes Λ_𝔠 = 𝔠⁻¹ ⊕ 𝔠 (change of
basis of norm ±1, so γ is unchanged) with e1 minimal.  So every form has a
highest point in the chart of its minimal vector's class: translations by 𝔠⁻²,
the same unit action, the same height bound y₁y₂ ≥ 2/d and the same "no highest
point" rule.  Galois maps the chart of a class C onto that of C⁻¹, so only one of
each pair C ≠ C⁻¹ is searched (these are the cusps of SL₂(O_K)).  Extreme forms
from different charts are compared after mapping their minimal vectors back to
O_K²; `chart_point(D, p)` maps any form to its chart.  Tests check the covering in
both directions against the chart-independent O_K² minimum.

## Recomputation of all squarefree D ≤ 300 (current code)

**156 of the 182 squarefree D ≤ 300 are certified** (exact mode, every certificate
re-checked with outward rounding).  The table is `results/gamma_K2.tsv` (one row per
field; rendered in `results/README.md`), and `results/fields/D<D>.json` holds the
full result of each field (all extreme classes found, normal forms, minimal
vectors, perfection rank, eutactic coefficients, certificate statistics).  It was
produced by

    hermite-humbert-sweep results/gamma_K2.tsv 2-300 -j 4 --timeout 600              # all fields, 10 min each
    hermite-humbert-sweep results/gamma_K2.tsv <D> -j 1 --field-jobs 4 --timeout 0   # then each timed-out field
    hermite-humbert-sweep results/gamma_K2.tsv <D> -j 1 --field-jobs 2 --timeout 0 --redo   # memory-killed fields

with checkpoints (`--checkpoints`, the default), so runs interrupted by restarts of
the machine resumed where they stopped.  The long fields took 0.2–15 h each on 4
cores (D = 206: 14.6 h, before the split-priority fix below).

Where both exist, the values agree to 20 digits with the original runs (the
tables further down, kept for reference: 51 fields).

### Open fields (D ≤ 300)

The remaining 26 fields are open: none was disproved, they were not finished.
Cost and memory grow with ε₀ (the region is long in the unit direction), so they
sort by the fundamental unit:

| reason | fields (ε₀) |
|---|---|
| ε₀ < 10⁶, still running: 21 slabs in s with up to 6·10⁶ candidate vectors each; the (serial) vertex search alone takes about 1 h per slab | 298 (8.2·10⁵) |
| ε₀ between 10⁶ and 10⁸: timed out at 10 min, not retried (likely a day or more each) | 201, 286, 259, 281, 193, 295, 94, 251, 217, 179, 127, 239, 249, 191 |
| ε₀ > 10⁸: killed for memory (16 GB machine) or not run; candidate sets of 10⁷–10⁸ vectors per field (D = 139: 3.2·10⁷ vectors, 13.6 GB when all slabs were held at once) | 163, 241, 139, 262, 283, 166, 151, 199, 271, 211 |
| ε₀ > 10¹² (≈ 1.4·10¹²): outside the double-precision range of the code | 214 |

For four certified fields the closed form was not identified (PSLQ up to degree 4
at the precision used); the table gives the certified value with `?`: D = 118, 134,
238, 247.

What made the last 15 fields with ε₀ ≤ 10⁶ finish:

- checkpoint/resume of the vertex search and of every branch-and-bound block, with
  follow-up blocks for blocks that exhaust their box budget (see "Checkpoints");
- the split priority: boxes are bisected mostly along the direction in which the
  bound is loose.  For large units that is x₁ (y₁ = e^(t−s) ≈ 10⁻⁴), and the old
  mixing weight kept bisecting the other directions; with the new weight D = 43 needs
  2.0·10⁴ boxes instead of 1.7·10⁶, and the rest of D = 247 (stuck for days in
  follow-up blocks) finished in 1.5 h;
- one slab's candidate set in memory at a time (D = 238 and 298 were killed for memory);
- skipping vertex-search starts whose float Gram matrix is numerically indefinite
  (D = 134), and a race in the parallel scheduler that could lose a finished block.

What would extend the reach further: a parallel vertex search over slabs, cutting
the high s-slabs into strips in x₁ so the candidate sets stay small, a compiled
per-box kernel, and extended precision for ε₀ > 10¹².

## Results (h_K = 1, D < 50; all certified in exact mode with `--verify`)

| D | ε₀ | γ_{K,2} | closed form | source |
|---|---|---|---|---|
| 2 | 1+√2 | 2.10639452948436 | (12+8√6)/15 = 4/(2√6−3) | BCIO01 Thm 3.4 |
| 3 | 2+√3 | 4 | 4 | BCIO01 Thm 3.6 |
| 5 | (1+√5)/2 | 1.78885438199983 | 4√5/5 | BCIO01 Thm 3.2 |
| 6 | 5+2√6 | 5 | 5 | PW09 |
| 7 | 8+3√7 | 4.25210189537186 | (100+16√70)/55 | PW09 conjecture — certified |
| 11 | 10+3√11 | 9.33333333333333 | 28/3 | PW09 conjecture — certified |
| 13 | (3+√13)/2 | 4.03532434528619 | (84+6√91)/35 | PW05, PW09 |
| 14 | 15+4√14 | 11.1111111111111 | 100/9 | not in the project papers |
| 17 | 4+√17 | 2.60798930002934 | √(784+128√34)/15 | PW05 local max — certified global |
| 19 | 170+39√19 | 8.05597465311096 | (1944+528√19)/527 | not in the project papers |
| 21 | (5+√21)/2 | 5.33333333333333 | 16/3 | PW09 |
| 22 | 197+42√22 | 10.8 | 54/5 | not in the project papers |
| 23 | 24+5√23 | 17.3333333333333 | 52/3 | not in the project papers |
| 29 | (5+√29)/2 | 5.79208192156049 | (255+25√145)/96 | not in the project papers |
| 31 | 1520+273√31 | 11.3142857142857 | 396/35 | not in the project papers |
| 33 | 23+4√33 | 4.57142857142857 | 32/7 | not in the project papers |
| 37 | 6+√37 | 5.73587564568333 | (405+63√185)/220 | not in the project papers |
| 38 | 37+6√38 | 30.16 | 754/25 | not in the project papers |
| 41 | 32+5√41 | 5.32110736028619 | (235+5√7585)/126 | not in the project papers |
| 43 | 3482+531√43 | 21.1253140668909 | (2028+104√1634)/295 | not in the project papers |
| 46 | 24335+3588√46 | — | — | not finished (see below) |
| 47 | 48+7√47 | 33.4126699130802 | (807+35√1833)/69 | not in the project papers |

"Not in the project papers": PW09 mentions a table for further fields at
math.tu-berlin.de/~wagner/HF.pdf, which was not checked.  Per-field details
(number of minimal vectors, extreme classes, timings) are in `results/results_h1_D_lt_50.tsv`
(computed with the previous version of the code; the certificates remain valid, and
the test suite re-derives the nine published values with the current version).

Remarks.  PW's radicands are squares: √((1476+144√91)/175) = (84+6√91)/35 and
√((5584+640√70)/605) = (100+16√70)/55; PW05's second Q(√17) form has
√((1408+128√85)/405) = (40+8√85)/45.  Q(√7) has an extreme (non-critical) form whose
invariant is exactly (84+6√91)/35 = γ_{Q(√13),2}.  Ten of the 21 values are rational.

**D = 46** (and D = 91 below) exceed the 5-minute limit of the sandbox used for
development: the vector sets are large (|V| ≈ 1.4·10⁵ for D = 46).  In a partial
run no point with G > 25/2 was encountered for D = 46, which proves nothing.
They should finish on a local machine without a time limit.

### Class number > 1 (D < 100; certified in exact mode with `--verify`)

None of these is in the project papers (BCIO01, PW05, PW09 treat h_K = 1 only).

| D | h | ε₀ | γ_{K,2} | closed form |
|---|---|---|---|---|
| 10 | 2 | 3+√10 | 8.70642950417960 | (80+16√10)/15 |
| 15 | 2 | 4+√15 | 12.5500929362678 | (112+16√159)/25 |
| 26 | 2 | 5+√26 | 23.5542168674699 | 1955/83 |
| 30 | 2 | 11+2√30 | 28 | 28 |
| 34 | 2 | 35+6√34 | 34.0072978122793 | (236+8√1174)/15 |
| 35 | 2 | 6+√35 | 33.1217374457386 | root of 6117075x⁴ − 169953840x³ − 1147606740x² + 2216077920x − 952092736 |
| 39 | 2 | 25+4√39 | 25 | 25 |
| 42 | 2 | 13+2√42 | 36.6263258075943 | √((16404497408+35995648√207130)/24440529) |
| 51 | 2 | 50+7√51 | 36.6019806576382 | (51300+360√37689)/3311 |
| 55 | 2 | 89+12√55 | 40.1142857142857 | 1404/35 |
| 58 | 2 | 99+13√58 | 40.1001981028670 | not found by PSLQ (degree ≤ 4 for γ, ≤ 8 for γ²) |
| 65 | 2 | 8+√65 | 17.0666666666667 | 256/15 |
| 66 | 2 | 65+8√66 | 52 | 52 |
| 70 | 2 | 251+30√70 | 52.2286544552382 | (1582+91√814)/80 |
| 74 | 2 | 43+5√74 | 47.8500949541327 | not found by PSLQ |
| 78 | 2 | 53+6√78 | 57.4843103699420 | root of 27025x⁴ − 1527305x³ − 7792967x² + 396887205x − 2041750854 |
| 79 | 3 | 80+9√79 | 52.1377082669887 | root of 5975347x⁴ − 339051976x³ + 1445360588x² − 559500864x − 719956224 |
| 82 | 4 | 9+√82 | 75.7456326366365 | not found by PSLQ |
| 85 | 2 | (9+√85)/2 | 22.0979739947391 | not found by PSLQ |
| 87 | 2 | 28+3√87 | 76 | 76 |
| 91 | 2 | 1574+165√91 | — | not finished (5-minute limit) |
| 95 | 2 | 39+4√95 | 101.347191421516 | not found by PSLQ |

Per-field details are in `results/results_h_gt_1_D_lt_100.tsv`.  The quartic/"not found"
entries are certified numerically to 100 digits; only the closed form is missing.

Q(√10) shows why several charts are needed: its critical form has minimal vectors
of principal content, e.g. (1, 0), and of non-principal content, e.g. (−√10, 2) with
content (2, √10); its Galois conjugate is a different GL₂(O_K)-class with the same γ,
and a further extreme form has γ ≈ 7.93430.  Seven of the 21 values above are rational (D = 26, 30, 39, 55, 65, 66, 87).

### Further fields (certified in exact mode with `--verify`, current code)

| D | h | γ_{K,2} | closed form | #M |
|---|---|---|---|---|
| 53 | 1 | 12.4163141720261 | root of 40320x⁴ − 294177x³ − 2304708x² − 3117576x − 1162084 | 6 |
| 61 | 1 | 6.45599184841325 | (1405+35√5185)/608 | 5 |
| 69 | 1 | 13.1649091645627 | (100+14√115)/19 | 8 |
| 77 | 1 | 17.6262208528866 | root of 98040x⁴ − 1586893x³ − 2947868x² + 8324316x − 4008004 | 6 |
| 83 | 1 | 62.2962962962963 | 1682/27 | 6 |
| 93 | 1 | 20.2666666666667 | 304/15 | 8 |
| 101 | 1 | 20.4327913832483 | root of 14008896x⁴ − 263485440x³ − 454196937x² − 218654280x − 24403600 | 6 |
| 102 | 2 | 65.7813937141254 | not found by PSLQ | 5 |
| 123 | 2 | 81.7282285456556 | (333764+3080√90118)/15397 | 6 |
| 197 | 1 | 38.5610602671244 | (9512+29√525793)/792 | 6 |
| 227 | 1 | 169.333333333333 | 508/3 | 12 |
| 257 | 3 | 52.2095910882804 | not found by PSLQ | 6 |

(`results/results_more.tsv`; D = 102, 123, 227 belong to the family D = n² + 2, whose h = 1
members D = 3, 6, 11, 38, 83, 227 all have rational γ.  `features.py` extracts
structural features of the critical forms; `results/features.jsonl` has them for D < 100.)

## Files

Package `src/hermite_humbert/`:

| file | contents |
|---|---|
| `field.py` | exact arithmetic in O_K = Z[ω], fundamental unit (continued fractions), h_K = 1 test, primitivity, SL₂ completion, elements of bounded norm modulo units |
| `ideals.py` | ℤ-lattices and ideals in K, principal-ideal test, class group with minimal-norm representatives, Steinitz frames and charts |
| `humbert.py` | forms ↔ points of ℍ², group action, reduction to the highest point, search region, candidate vectors, independent minimum (Fincke–Pohst) |
| `vertex.py` | LP trust-region ascent, Newton at high precision, perfection rank, eutaxy, multistart |
| `certify.py` | branch-and-bound (tolerance and exact modes), recorded certificate, outward-rounded re-check |
| `analysis.py` | equivalence ([PW05, Alg. 11]), normal forms, PSLQ identification, local ball certificate |
| `core.py` | driver / CLI (`hermite-humbert`), `compute`, `gamma_of_form`, `chart_point` |
| `sweep.py` | parallel, resumable batch runs (`hermite-humbert-sweep`) |
| `features.py` | feature extraction for the critical forms (patterns) |

Elsewhere:

| path | contents |
|---|---|
| `tests/test_hermite_humbert.py` | 17 pytest tests (incl. soundness of rule 4 and its interval re-check): PARI cross-checks (units, class numbers), enumeration vs brute force, published forms, exactness of V, invariance, chart covering in both directions, pipeline vs published values (40 digits), h = 2 regression, verifier soundness |
| `results/` | result tables, per-field JSON (`results/fields/`), `features.jsonl` |
| `.github/workflows/tests.yml` | CI: unit tests on Python 3.9 and 3.12, pipeline tests on 3.12 |

## Extensions (not implemented)

- Implemented since: parallel sweeps over fields (`hermite-humbert-sweep`), a parallel
  branch-and-bound and certificate re-check within a field (`--jobs` / `--field-jobs`),
  slabs in s for large units, a vectorised rule-4 re-check.
- Implemented since: checkpoint/resume of a field (see "Checkpoints").
- Not yet: a compiled per-box kernel, strips in x₁ for the high s-slabs (see
  "Open fields").
- Totally real fields of higher degree (points of ℍⁿ) and n ≥ 3 (γ_{K,n}).

## References (project files)

- [BCIO01] R. Baeza, R. Coulangeon, M. I. Icaza, M. O'Ryan, Hermite's constant for
  quadratic number fields, Experiment. Math. 10 (2001) 543–551.
- [PW05] M. E. Pohst, M. Wagner, On the computation of Hermite–Humbert constants for
  real quadratic number fields, J. Théor. Nombres Bordeaux 17 (2005) 905–920.
- [PW09] M. E. Pohst, M. Wagner, On the computation of Hermite–Humbert constants:
  the algorithm of Cohn revisited, J. Algebra 322 (2009) 936–947.
- R. Coulangeon, Voronoï theory over algebraic number fields, Monogr. Enseign. Math. 37 (2001);
  H. Cohn, A numerical survey of the floors of various Hilbert fundamental domains,
  Math. Comp. 19 (1965) — both as cited in [BCIO01], [PW09].
