# Discovery — py-Pando

Run on 2026-10-13 with omicverse-rebuildr `engine.discover_omicverse_deps` (104 repos cached).
Upstream: quadbio/Pando @ e8abb30 (DESCRIPTION version 1.1.1), MIT.

## Direct check

```
## Discovery — `Pando`

**No existing omicverse port found.** Safe to start a new port.
```

## Dependency audit

### Imports

| R dep | omicverse match | Description |
|---|---|---|
| `dplyr` | — | (none) |
| `purrr` | — | (none) |
| `stringr` | — | (none) |
| `tibble` | — | (none) |
| `tidyr` | — | (none) |
| `Seurat` | [`py-cca`](https://github.com/omicverse/py-cca) / [`py-spatialecotyper`](https://github.com/omicverse/py-spatialecotyper) | A pure-Python re-implementation of Seurat's RunCCA (Stuart, Butler, Hoffman, Haf…; Pure-Python port of the R package SpatialEcoTyper (Zhang et al., Nature 2026): r… |
| `SeuratObject` | — | (none) |
| `Signac` | — | (none) |
| `Matrix` | [`anndata-oom`](https://github.com/omicverse/anndata-oom) | Out-of-memory AnnData powered by Rust — a drop-in replacement for anndata.AnnDat… |
| `methods` | [`py-monocle2`](https://github.com/omicverse/py-monocle2) / [`py-dada2`](https://github.com/omicverse/py-dada2) | A pure-Python re-implementation of Monocle 2 (Qiu et al., Nature Methods 2017) f…; A pure-Python re-implementation of DADA2 (Callahan et al., Nature Methods 2016) … |
| `MatrixGenerics` | — | (none) |
| `ggplot2` | — | (none) |
| `IRanges` | — | (none) |
| `GenomicRanges` | — | (none) |
| `S4Vectors` | — | (none) |
| `motifmatchr` | — | (none) |
| `Biostrings` | — | (none) |
| `BSgenome` | — | (none) |
| `GenomeInfoDb` | — | (none) |
| `glmnetUtils` | — | (none) |
| `bayestestR` | — | (none) |
| `ggpointdensity` | — | (none) |
| `ggraph` | — | (none) |
| `igraph` | — | (none) |
| `tidygraph` | — | (none) |
| `pals` | — | (none) |
| `patchwork` | — | (none) |
| `scales` | — | (none) |
| `pbapply` | — | (none) |
| `TFBSTools` | — | (none) |
| `foreach` | — | (none) |
| `sparseMatrixStats` | — | (none) |
| `irlba` | — | (none) |
| `uwot` | — | (none) |

## Decisions

The three automatic matches are spurious and are **not** reused:

- `Seurat` → `py-cca` / `py-spatialecotyper`: these port Seurat's RunCCA and SpatialEcoTyper. Pando uses Seurat only as a
  container (assay layers, variable features). py-Pando takes AnnData / NumPy / pandas inputs instead.
- `Matrix` → `anndata-oom`: description-substring hit. `scipy.sparse` covers what Pando uses (`dgCMatrix`, `crossprod`).
- `methods` → `py-monocle2` / `py-dada2`: description-substring hits on the word "methods". R S4 classes become Python dataclasses.

Native-Python equivalents for the deps without a mirror:

| R dep | Used for in Pando | py-Pando |
|---|---|---|
| `Seurat`, `SeuratObject`, `Signac` | object container; `Extend`, `GRangesToString`, `StringToGRanges`, `Annotation` | AnnData + pandas; `pypando.ranges` reimplements the GRanges operations used (closed 1-based intervals) |
| `IRanges`, `GenomicRanges`, `S4Vectors`, `GenomeInfoDb` | `findOverlaps`, `pintersect`, `subtract`, `intersect`, `resize` | `pypando.ranges` (NumPy) |
| `motifmatchr` (+ Pando's patched background conversion) | MOODS scanning | **MOODS-python** — the same MOODS C++ library motifmatchr compiles in |
| `TFBSTools`, `Biostrings`, `BSgenome` | `toPWM`, letter frequencies, `getSeq` | `pypando.motifs` (`to_pwm` reimplemented) + `pyfaidx` on a FASTA |
| `stats::glm` | default model | `pypando.models.fit_glm`: R's `glm.fit` (LINPACK `dqrdc2` QR with limited pivoting) reimplemented |
| `glmnetUtils` / `glmnet` | glmnet, cv.glmnet | `pypando.glmnet`: Gaussian elastic-net coordinate descent following glmnet's path rules |
| `xgboost` | xgb | Python `xgboost` (same C++ core) |
| `reticulate` → scikit-learn | bagging_ridge, bayesian_ridge | scikit-learn directly (Pando already calls it from R) |
| `bayestestR`, `brms` | brms models | **not ported** (Stan); see port.yaml |
| `sparseMatrixStats`, `Matrix` | colMaxs, aggregation | NumPy / scipy.sparse |
| `dplyr`, `purrr`, `stringr`, `tibble`, `tidyr` | data wrangling | pandas |
| `igraph`, `tidygraph`, `ggraph`, `uwot`, `irlba` | network graph layout (`get_network_graph`) | networkx for the graph; layout/plotting out of scope |
| `ggplot2`, `ggpointdensity`, `pals`, `patchwork`, `scales` | plotting | out of scope (plotting) |
| `pbapply`, `foreach` | progress bars / parallelism | plain loop / joblib |

## Amendment 2026-10-14 — `get_tf_network` / `plot_tf_network`

Written from `R/plots.R` (Pando 1.1.1 @ e8abb30, lines 445–668) before any Python code for these functions.

### What the function does

`get_tf_network(object, tf, graph = 'module_graph', features = NULL, order = 3, keep_all_edges = FALSE)` builds a
tree-like sub-network rooted at `tf` from the module graph of `get_network_graph`. The module graph has one
edge per TF → target pair in the module table, with `estimate`, `padj` and the other module columns as edge attributes.

1. `features` defaults to the network's target genes, intersected with the graph's nodes.
2. `igraph::all_shortest_paths(gene_graph, tf, features, mode = 'out')` lists every shortest path from `tf` to each feature.
   With fewer than 3 paths, the function stops.
3. For each path it computes:
   - `dir`, the product of the edge signs;
   - `order`, the number of edges;
   - `mean_estimate`, the mean edge estimate;
   - if the edges carry `padj`, `mean_log_padj`, the mean of `-log10(padj)` over the path's edges.
4. Paths are grouped by end node. A pruning step then keeps, per end node:
   - every path when `keep_all_edges = TRUE`;
   - otherwise, the direct edge (`order == 1`) plus the strongest path. The doc says "prune to the strongest overall
     connection", so strongest means the largest `mean_log_padj` (the most significant path) when p-values exist, and
     the largest `mean_estimate` otherwise.
5. The edges of the kept paths are joined back to the module graph's edges. The result carries `end_node` and
   `comb_dir`, and has one row per (edge, end node, combined sign). It is stored in `graphs$tf_graphs[[tf]]`.

`plot_tf_network` draws that graph with `ggraph(layout = 'tree', circular = TRUE)`:
- diagonal edges coloured by `dir`;
- labels on the network's TFs, on every node, or on the root only.

### The bugs, exactly

1. **The default pruning never uses p-values.** The test is `'padj' %in% colnames(grn_pruned)`, but `grn_pruned` is the
   per-path table, whose column is named `mean_log_padj`. It never has a `padj` column, so the test is always FALSE.
   The branch it guards filters on `mean_padj`, a column that exists nowhere. For a glm network, the default therefore
   falls through to `mean_estimate == max(mean_estimate)`. That keeps the path with the most positive mean
   coefficient, so a strongly significant repressive path always loses to a weak activating one. The p-value branch
   is dead code; if it were reached, it would fail on `mean_padj`.
   (Correction to the 0.1.0 report: R does not error on this path, it silently uses the estimate.)
2. **`order` is ignored.** `filter(order <= order)` sits inside dplyr's data mask, so both sides are the per-path column
   `order`. The test is always TRUE and the documented "maximal order of the graph" has no effect.

### Intended semantics implemented in py-Pando

- If the per-path table has `mean_log_padj`, keep per end node the paths with `order == 1` or
  `mean_log_padj == max(mean_log_padj)`. Otherwise keep `order == 1` or `mean_estimate == max(mean_estimate)`, as R does.
- Keep only paths with `order <= order` (the argument).
- Everything else follows R exactly:
  - the shortest-path set;
  - R's `mean` (long-double accumulation with the second refinement pass) for the path statistics, so ties resolve as in R;
  - the join back to the module edges;
  - `comb_dir`.

With `keep_all_edges = True` and `order = inf`, the result is the installed R function's output (gate
`tf_network_all_edges`). Every other setting is gated against R patched with exactly these two changes
(`tf_network_intent()` in `tests/r_reference_driver.R`; gate `tf_network_default_intent`).
