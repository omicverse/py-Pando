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
