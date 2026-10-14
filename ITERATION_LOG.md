# Acceleration Iteration Log — py-Pando

> One block per Acceleration step (omicverse-rebuildr PROTOCOL 3b). Benchmark: `tests/bench_infer_grn.py` on the canonical fixture —
> `infer_grn(method="glm")` + `infer_grn(method="glmnet")` (Pando's default model and its elastic-net variant), one discarded warm-up,
> three timed runs, one BLAS thread, same node type. R reference for the same two calls: 140.6 s + 48.8 s = 189.4 s.
> The equivalence iterations that preceded the baseline are narrated in `examples/evolution.ipynb` (iterations 0-4).

---

## Baseline — 2026-10-13 19:00:00

```yaml
iter: 0
status: baseline
action: null
admissibility: null
playbook_section: null
wall_clock_mean_s: 1292.7
wall_clock_stddev_s: 246.6
wall_clock_runs_s: [1545.6, 1053.0, 1279.6]
warmup_run_s: 2181.2
parity_metric: 1.0
parity_class: mixed (fraction of pre-registered gates against the params-honoured reference)
parity_threshold: 1.0
parity_passes: true
notes: |
  Equivalence result: BLAS-exact dqrdc2 recomputed at every IRLS step (glm), interpreted coordinate descent (glmnet).
  All deterministic gates pass (glm bit-identical to R with one BLAS thread).
```

## iter 1 — 2026-10-13 19:10:00

```yaml
iter: 1
status: ACCEPT
action: memoise_qr_across_irls
playbook_section: "§1.1"
admissibility: exact
admissibility_evidence: |
  glm.fit decomposes x*w each IRLS step; Gaussian/identity with unit weights gives w = 1 exactly, so x*w == x bit for bit
  and dqrdc2 (deterministic) returns the same factorisation. Only dqrsl with the new working response is rerun. Outputs
  verified bit-identical (BACH2 2000x2173, 25 non-converging iterations: 284 s -> 16 s). See MATH.md §1.
wall_clock_mean_s: 903.3
wall_clock_stddev_s: 163.3
wall_clock_runs_s: [1091.6, 801.3, 816.9]
warmup_run_s: 1609.5
speedup_vs_previous: 1.43
speedup_vs_baseline: 1.43
parity_metric: 1.0
parity_class: mixed (fraction of pre-registered gates against the params-honoured reference)
parity_threshold: 1.0
parity_passes: true
parity_delta_vs_baseline: 0.0
```

## iter 2 — 2026-10-13 19:20:00

```yaml
iter: 2
status: ACCEPT
action: numba_elnet_path
playbook_section: "§4.4"
admissibility: bounded
admissibility_evidence: |
  Same coordinate-descent algorithm compiled with numba; inner products as sequential sums instead of BLAS ddot.
  Reordering bound |fl(x.y)_A - fl(x.y)_B| <= 2 gamma_n sum|x_i y_i| <= 2 gamma_n ~ 4.4e-13 (standardised columns, n = 2000),
  six orders below glmnet's thresh = 1e-7. See MATH.md §2.
wall_clock_mean_s: 306.1
wall_clock_stddev_s: 5.7
wall_clock_runs_s: [299.7, 308.2, 310.5]
warmup_run_s: 327.9
speedup_vs_previous: 2.95
speedup_vs_baseline: 4.22
parity_metric: 1.0
parity_class: mixed (fraction of pre-registered gates against the params-honoured reference)
parity_threshold: 1.0
parity_passes: true
parity_delta_vs_baseline: 0.0
perturbation_bound: |
  per covariance / gradient entry <= 2 gamma_n = 4.4e-13 (n = 2000); coefficients vs R on the probe design 2e-13 before and after.
```

### Decision

ACCEPT for iterations 1 and 2. No further admissible rewrite from the playbook was pursued: the remaining glm cost is the
BLAS-exact QR itself, whose call sequence must stay R's for bit-identity.

## Not in this log

* Near-target motif scanning (`region_background` / `regions_near_genes` / `subset_regions`, exact; MATH.md §3) is a user-facing
  helper outside Pando's API, not a change to `find_motifs`; on the fixture it cuts the scan from 8,127 to 4,623 regions with a
  bit-identical network.
