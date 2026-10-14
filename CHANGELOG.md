# Changelog

## 0.1.2 — 2026-10-14

### Changed: plots restyled after R Pando's ggplot2 / ggraph figures
- **`plot_gof`.** Scatter of R² against model size with marginal histograms (top R², right model size, 20 bins).
  - Points are coloured by local density (ggpointdensity's neighbour count, reversed Brewer reds).
  - Models that fail `find_modules`' thresholds are faded, with dashed lines at the thresholds read from the object.
  - Pseudo-log y axis, so a model with 0 variables can be drawn. Pando's axis labels.
- **`plot_module_metrics`.** Violins with an inner boxplot, in Pando's panel order: target genes per TF on a pseudo-log
  axis, then regions per target gene, then TFs per target gene. Pando's titles and labels.
- **`plot_network_graph`.**
  - Diagonal edges: orange activating, dark grey repressing.
  - Nodes filled with magma by PageRank, sized 1–5 mm, grey outline.
  - Every gene labelled, with labels repelled by adjustText and leader lines.
- **`plot_tf_network`.** Same layout as ggraph's `layout = 'tree'`: igraph Reingold–Tilford with ggraph's radial
  transform (radius = depth / max depth, clockwise from the top).
  - Node coordinates equal ggraph's on the fixture (PAX5 sub-network) and on the toy graph in the smoke tests.
  - Diagonal edges as ggraph draws them; point size and label boxes as in Pando.
- New extra: `pip install 'pypando[plot]'` installs matplotlib and adjustText. Without adjustText, labels are not repelled.


## 0.1.1 — 2026-10-14

### Added
- `get_tf_network` and `plot_tf_network`.
  - With `keep_all_edges=True, order=np.inf`, `get_tf_network` returns exactly what R Pando 1.1.1 returns: identical
    nodes and edges on the fixture, attributes within about 5e-16 relative.
  - `plot_tf_network` draws the result as a matplotlib tree or circular tree, like ggraph's `layout = 'tree'`.

### Deviation from R (bug fix)
- **`get_tf_network` default (`keep_all_edges=False`).** For each leaf it keeps the direct edge plus the path with the
  largest mean `-log10(padj)`, as the function intends.
  - Pando 1.1.1 tests for a column named `padj` in its per-path table. The column there is `mean_log_padj`, so the
    test is always false and R falls back to the largest *mean estimate*. R's p-value branch also filters on a
    nonexistent `mean_padj`.
  - On the fixture this changes the kept paths for 2 of 5 TFs.
- **`get_tf_network(order=...)` is honoured.** Pando's `filter(order <= order)` compares the column with itself.
- Both fixes are gated against R patched with exactly these changes. See DISCOVERY.md and RECONSTRUCTION_REPORT.md §3.1b and §3.3.

### Fixed
- `get_network_graph` now drops edges of genes without a layout row (TFs that are never a target), as tidygraph's
  node `inner_join` does in R. In 0.1.0 those nodes were dropped but their edges kept. The module graph now matches R's
  on the fixture (112 edges).

## 0.1.0 — 2026-10-13

First release: R Pando 1.1.1 ported. 13 of 14 parity gates pass; see RECONSTRUCTION_REPORT.md.
