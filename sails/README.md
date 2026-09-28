# sails — sails and indecomposable integers of totally real fields

Computes, for a totally real number field *K* (degree 2–5 tested, any degree
supported), the **sails** of every inequivalent sign class and the
**indecomposable integers** of each class, modulo totally positive units, and writes
them in the output format of the "Simple hybrid" notebook (Siu Hang Man's code,
`Simple_hybrid_signed_Version_260803.ipynb`).  It is a faster, exact
re-implementation of the pipeline of that notebook and of the `trsail` package
(the decision problem of `trsail` is not included).

Pure Python + [cypari2](https://github.com/sagemath/cypari2) (PARI/GP); runs with
plain `python3` or inside Sage (`sage -python`).

```bash
pip install -r requirements.txt                        # cypari2, numpy, mpmath
python -m sails field 1,-2,-1,1 --label 3.3.49.1 --header
python -m sails fetch-visser --degree 3 --out data/deg3.txt      # field list (from the LMFDB)
python -m sails run --source data/deg3.txt --degree 3 --disc-max 100000 \
                    --out results/deg3.txt --jobs 8 --timeout 3600 --json
python -m sails compare results/deg3.txt --reference ../indecomposables/data/deg3.txt
python tests/test_sails.py
```

## Output

One line per field, `|`-separated, exactly the columns of the notebook's
`data_exporter`:

| # | column | example (3.3.229.1) |
|---|---|---|
| 1 | LMFDB label | `3.3.229.1` |
| 2 | Polynomial coefficients (ascending) | `[-1,-4,0,1]` |
| 3 | Sail signatures (GF(2) sign vectors, 1 = negative) | `[(0, 0, 0),(1, 0, 0)]` |
| 4 | TPFUs (totally positive fundamental units) | `[a + 2,a^2]` |
| 5 | # Indecomposables per signature | `[4,6]` |
| 6 | # Facets per signature | `[2,2]` |
| 7 | Max norm (for indecomposables) | `[16,28]` |
| 8 | Max integer distance | `[2,3]` |
| 9 | Max integer volume | `[2,7]` |
| 10 | List of indecomposables | `[[1, a^2 + 2*a + 1, ...],[a + 1, ...]]` |
| 11 | List of sail faces `[vertices, int distance, int volume]` | `[[[[(1, 0, 0), (1, 4, 2), (2, 1, 0)], 1, 2], ...],[...]]` |

Elements are polynomials in the root `a` of the defining polynomial; facet vertices
are vectors of power basis coordinates (as in the notebook).  The signature column
lists representatives of `GF(2)^n / sgn(O_K^×)` chosen exactly as the notebook does
(Sage's `Tuples` order), the first one being totally positive; the embeddings are
ordered by the value of the root.  `--json` additionally writes `OUT.jsonl` with the
full data (facet normals, integral-basis coordinates, class number, regulator,
timings, counters).

Differences to the notebook's output that do not affect any invariant:
representatives of indecomposables are canonical (the element of smallest
`T2 = Tr(x^2)` in its orbit under totally positive units, ties broken by
coordinates) and sorted by norm; the TPFU basis is LLL-reduced in log space
(`--gordon-units` keeps the notebook's basis); each facet orbit is represented by a
canonical translate, facets are sorted by (distance, volume).

## Algorithm

Notation: `P_σ = {x ∈ O_K \ 0 : sgn σ_i(x) = σ_i}` for a sign vector σ; `U⁺` the
totally positive units.  A facet is `(c, d)`: `c ∈ Z^n` primitive (dual-basis
coordinates of its normal δ, so `Tr(δx) = c·x` in integral-basis coordinates) and
`d = min{c·x : x ∈ P_σ}` — `d` **is** the integer distance.  Integer volume = sum of
`|det|` over a triangulation of the facet, divided by `d`.

1. **Seeds.** Generators of the right sign vector of principal ideals of norm
   `≤ disc^{1/n}` (`bnfisintnorm`), as in the notebook (bound doubled until every
   class has one).
2. **Orbit hull by gift wrapping modulo U⁺.**  For a finite `S ⊂ P_σ` the facets of
   `conv(U⁺S)` are found by rotating a supporting hyperplane about every ridge of
   every facet orbit (start: `Tr(s₀x)`, rank raised by rotations).  The points
   `q = u·s` with `c·q ≤ D` are enumerated in the unit lattice: `Tr(δ u s) ≤ D` forces
   `log σ_i(δus) ≤ log D`, a small simplex in `Z^{n-1}` (vectorised in numpy); every
   candidate is then decided with exact integers.  New facets are reduced to a
   canonical translate; a ridge through which a facet was reached is not rotated
   again.  The traversal is closed by construction, i.e. the sail is complete
   (no separate completeness check as in the notebook).
3. **Refinement to the sail.**  The half-open parallelepipeds over a triangulation of
   the orbit hull contain every lattice point strictly below it; their points are
   computed exactly from the Smith normal form of the simplex matrix.  They are added
   to `S` and step 2 repeated until nothing lies below: then the orbit hull *is* the
   sail of `P_σ`.
4. **Indecomposables.**  Each facet is re-triangulated through all its lattice
   points (exact stellar subdivisions).  Candidates are the facet lattice points and
   the parallelepiped points of these simplices; their height `s = c·x/d` satisfies
   `s < n−1`, and
   * `s < 2` ⟹ indecomposable (a decomposition needs two heights ≥ 1) — in degree 3
     this covers every candidate, so **cubic fields need no test at all**;
   * `s = 2` ⟹ decomposable iff `x ∈ W_F + W_F` (hashed sumset);
   * `s > 2` ⟹ decomposable iff `x − ι·u ∈ P_σ` for an already found indecomposable
     `ι` with `N(ι) ≤ N(x)/2^n` (superadditivity of `N^{1/n}`) and `u ∈ U⁺` (a small
     box in the unit lattice).
   In degree 3 the count is checked against the Kala–Man identity
   `#Ind = Σ_F (d_F − ½)·vol_F` at run time.

**Exactness.** Signs, norms, traces, facet functionals, hulls, triangulations,
Smith forms and all comparisons are exact (Python integers, PARI).  Floating point
only bounds enumeration boxes (with outward margins) and orders candidates; logs of
embeddings are computed from a 100-digit embedding matrix whenever double precision
would lose more than 10⁻¹⁰.  Units are certified with `bnfcertify` (disable with
`--no-certify`).

## Validation

| check | fields | result |
|---|---|---|
| rvisser7/indecomposables tables (TP count and max norm) | all 1238 cubic fields, disc ≤ 28669 | all agree |
| same | all 352 quartic fields, disc ≤ 46400 | all agree |
| same | all 143 quintic fields of the table (151 fields computed), disc ≤ 495317 | all agree |
| trsail's orthant table (count, max norm, #facet orbits, S), every sign class | 27 cubic fields, disc ≤ 1000 | all agree |
| trsail's reference values (incl. 4.4.1125.1, 5.5.14641.1) | 13 fields | all agree |
| Kala–Tinková: simplest cubics have `2 + (a+1)(a+2)/2` indecomposables | a = −1…7 (monogenic) | all agree |
| Dress–Scharlau (continued fractions) | 22 real quadratic fields | all agree |
| brute force: all TP elements of norm ≤ disc, box search for summands | 6 cubic / quartic fields | identical lists |
| Kala–Man identity (degree 3) | every cubic run | checked at run time |

## Performance

PERFORMANCE_TABLE

## Field sources

* `python -m sails fetch-lmfdb --degree 3 --disc-max 100000 --out data/lmfdb_deg3.txt`
  pages through the LMFDB API (`nf_fields`, `degree=i3`, `r2=i0`), writing
  `label|coeffs` lines.
* `python -m sails fetch-visser --degree N` downloads
  `totally_real_fields/degN.txt` from rvisser7/indecomposables (all fields taken
  from the LMFDB: 193467 cubic, 13073 quartic, 9494 quintic); labels are rebuilt as
  `n.n.disc.index`.
* `run --source` also accepts LMFDB search downloads (`columns = [...]`,
  `data = [...]`), JSON, and plain `label|c0,...,cn` lines.

`run` resumes (labels already in `OUT` or `OUT.errors` are skipped), writes failures
and timeouts to `OUT.errors`, and supports `--shard i/m` to split a list over
machines.

## Files

| file | contents |
|---|---|
| `sails/field.py` | PARI setup: integral basis, multiplication table, embeddings, units, the notebook's TPFU and sail-signature procedures, canonical forms, Sage-style printing |
| `sails/exact.py` | exact integer linear algebra, convex hulls (brute force / gift wrapping), pulling triangulations |
| `sails/sail.py` | orbit hulls by gift wrapping modulo U⁺, refinement, parallelepipeds (Smith form), fine triangulations |
| `sails/indecomposables.py` | seeds, candidate classification |
| `sails/compute.py` | per-field driver, output line, JSON record |
| `sails/sources.py` | LMFDB API, rvisser7 lists, LMFDB downloads |
| `sails/cli.py` | command line, parallel batch runner |
| `tests/test_sails.py` | the checks listed above |
