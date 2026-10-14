# MATH.md — admissibility of py-Pando's acceleration rewrites

Every rewrite kept in the port is listed with its admissibility class (omicverse-rebuildr
ACCELERATION_PLAYBOOK): **E** exact identity, **B** bounded ε-approximation, **C** class containment.

## 1. QR memoisation across IRLS iterations (E)

`glm.fit` decomposes `x[good, ] * w` with `dqrdc2` at every IRLS iteration. For
`family = gaussian` (identity link) with unit prior weights, `w = sqrt(1 * 1^2 / 1) = 1` exactly
and `good` is all rows, so `x * w` is bit-identical to `x` at every iteration, and `dqrdc2` — a
deterministic function of its input and `tol` — returns the same `qr`, `qraux`, `pivot` and rank.
Only the solve against the new working response `z` changes. `pypando.rstats._qrls(cache=...)`
keeps the first decomposition and re-runs only `dqrsl` (identical calls to identical BLAS kernels).

Evidence: on BACH2 (2,000 x 2,173 design, 25 non-converging IRLS iterations) the outputs with and
without the cache are bit-identical; 284 s → 16 s. On the fixture, the glm gates are unchanged.

## 2. Compiled coordinate descent for glmnet (B)

`pypando.glmnet._elnet_path_nb` is the same algorithm as the interpreted `_elnet_path` (same
coordinate order, same active-set and convergence logic), compiled with numba and with the inner
products written as sequential loops instead of BLAS `ddot` calls. Only the summation order of
inner products changes. For an inner product of length `n` computed in a different order, the
standard bound is

    |fl(x·y)_A − fl(x·y)_B| ≤ 2 γ_n Σ_i |x_i y_i|,   γ_n = n u / (1 − n u),  u = 2^-53.

With standardised columns (`Σ_i x_ij^2 = 1` under weights `1/n`), Cauchy–Schwarz gives
`Σ_i |x_i y_i| ≤ 1`, so each gradient / covariance entry moves by at most `2 γ_n ≈ 4.4e-13` for
`n = 2,000`. These perturbations are six orders of magnitude below glmnet's convergence threshold
(`thresh = 1e-7` on `max_j x_j'x_j Δβ_j^2`), so they can only alter where the iteration stops within
that tolerance, never the lambda sequence beyond rounding. The ordinal gate (`pearson ≥ 0.999`,
`support_jaccard ≥ 0.95`, `lambda_rel ≤ 1e-4`) is unaffected; measured against R on the probe
design, coefficients agree to 2e-13 with and without compilation.

## 3. Near-target motif scanning (E) — helper outside the R API

`find_motifs(bg = "subject")` uses the A/C/G/T composition of all candidate regions as the MOODS
background; thresholds depend only on the matrices and that background, and every region is scanned
independently. Regions are assigned to genes one region at a time (`find_peaks_near_genes`, both
`Signac` windows and `GREAT` domains), and `infer_grn` only reads regions assigned to a target gene.
Hence, with `bg = region_background(all regions)`, scanning only `regions_near_genes(...)` yields
exactly the rows of the full motif matrix that `infer_grn` reads, and the same network. Verified
on the fixture: 4,623 of 8,127 regions scanned, coefficients, fit and modules bit-identical.
