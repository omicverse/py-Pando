# py-Pando

**Pure-Python Pando**: gene regulatory network inference from paired single-cell RNA + ATAC data, AnnData-native, parity-tested against the R package.

[Pando](https://github.com/quadbio/Pando) (Fleck *et al.*, *Nature* 2023, [doi:10.1038/s41586-022-05459-w](https://doi.org/10.1038/s41586-022-05459-w)) fits, for every target gene, a regression of its expression on TF expression x accessibility of TF-binding peaks near the gene, and reads significant TF–peak terms as regulatory links. py-Pando ports it following the [omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr) protocol: the R source is the specification, and pre-registered parity gates (`data/manifest.yaml`) were committed before any algorithmic code.

## Install

```bash
pip install pypando            # core
pip install "pypando[xgb]"     # + xgboost models
pip install "pypando[umap]"    # + UMAP layout of the module graph
```

## Quick start

```python
import pypando as pp

# rna:  AnnData, cells x genes, log-normalised (Seurat LogNormalize)
# atac: AnnData, cells x peaks, TF-IDF (Signac RunTFIDF), var_names "chr-start-end"
annotation = pp.annotation_from_gtf("gencode.v41.basic.annotation.gff3.gz")

grn = pp.initiate_grn(rna, atac, annotation)                # peaks minus exons
grn = pp.find_motifs(grn, genome="hg38.fa")                 # Pando's 1,590 motifs, MOODS
grn = pp.infer_grn(grn, genes=targets, method="glm")        # one model per target gene
grn = pp.find_modules(grn)                                  # TF modules

pp.coef(grn)            # TF / target / region / estimate / std_err / pval / padj / corr
pp.gof(grn)             # R^2 and number of variables per target gene
pp.NetworkModules(grn)  # module table + genes_pos / genes_neg / regions / peaks
```

Within Epione: `epione.grn.pando(rna, atac, genome="hg38", genes=targets)`.

## Function map

| R (Pando 1.1.1) | Python | Parity on the fixture |
|---|---|---|
| `initiate_grn` | `pp.initiate_grn` | identical regions (with and without `regions=` phastCons) |
| `find_motifs` | `pp.find_motifs` | identical region x motif matrix (1,240,962 hits) |
| `find_peaks_near_genes` (Signac, GREAT) | `pp.find_peaks_near_genes` | identical |
| `infer_grn(method = "glm")` | `pp.infer_grn` | identical rows; estimates / SE / t / rsq to ~1e-14; -log10 p to 4e-13 |
| `infer_grn(method = "glmnet")` | | lambda identical; coefficients Pearson 0.9999 (see notes) |
| `infer_grn(method = "cv.glmnet")` | | R's folds reproduced; coefficients within 5e-14 |
| `infer_grn(method = "xgb")` | | **gate fails vs installed R** (xgboost >= 3 drops Pando's parameters); identical (2e-16) with them honoured |
| `infer_grn(method = "bagging_ridge" / "bayesian_ridge")` | | ~1e-14 (NumPy seeded as in R) |
| `find_modules` | `pp.find_modules` | identical modules |
| `fit_model`, `fit_glm`, `fit_glmnet`, `fit_cvglmnet`, `fit_xgb`, `fit_bagging_ridge`, `fit_bayesian_ridge` | same names | per-function, to ~1e-14 (notebook 3) |
| `format_coefs`, `sparse_cor`, `aggregate_matrix`, `aggregate_assay`, `GetAssaySummary` | same names | exact |
| `get_network_graph` | `pp.get_network_graph` | graph + PageRank; UMAP layout via umap-learn (not R's uwot stream) |
| `coef`, `gof`, `GetNetwork`, `NetworkModules`, `NetworkTFs`, `NetworkRegions`, `NetworkFeatures`, `NetworkParams`, `NetworkGraph`, `DefaultNetwork`, `GetGRN`, `Params`, `create_GRNData` | same names | accessors |
| `plot_gof`, `plot_module_metrics`, `plot_network_graph` | same names (matplotlib; `pip install 'pypando[plot]'` for repelled labels) | styled after Pando's ggplot2 figures: density-coloured gof scatter with marginal histograms and threshold lines, violins with boxplots, diagonal edges and magma PageRank nodes |
| `fit_brms` | — | not ported (Stan) |
| `get_tf_network` | `pp.get_tf_network` | **deliberate fix**: the default prunes each leaf to its most significant path (mean −log10 padj) and honours `order`; R 1.1.1 silently prunes on the mean estimate and ignores `order`. `keep_all_edges=True, order=np.inf` = R, identical edges; default = R with the intended fix, identical edges |
| `plot_tf_network` | `pp.plot_tf_network` | ggraph's `layout = 'tree'` reproduced (igraph Reingold-Tilford + ggraph's radial transform: node coordinates equal R's on the fixture) |

Full numbers: [RECONSTRUCTION_REPORT.md](RECONSTRUCTION_REPORT.md) and `examples/compare_R_vs_Python.ipynb`.

## What makes the glm network identical

R's `glm` uses LINPACK's `dqrdc2` QR with limited pivoting: a column whose remaining norm falls below `1e-11` of its original norm is moved to the end and its coefficient becomes `NA`. Pando's per-gene designs (hundreds to thousands of `peak x TF` products, often nearly collinear) sit right at that threshold, so the pivot decisions depend on the last bits of BLAS level-1 results. py-Pando's QR (`pypando._blas`, numba) calls the same BLAS routines (`dnrm2`, `ddot`, `daxpy`, `dscal`) in the same order as R's Fortran, multiplies with `dgemv` like R's `%*%`, sums in long double like R's `sum()`, and inverts with LAPACK `dpotri` like `chol2inv`. With the same BLAS build (here OpenBLAS 0.3.34) and one BLAS thread (`infer_grn(blas_threads=1)`, the default), the glm network is bit-identical to R's, including genes where R's IRLS does not converge.

## Notes

* **cv.glmnet** folds come from an emulation of R's `set.seed()` + `sample()` (`pypando.rrng`), so `infer_grn(method="cv.glmnet", seed=42)` uses R's folds after `set.seed(42)`. Fold fits follow glmnet 5.0: each fold on its own lambda path, predictions interpolated onto the full path (glmnet 4.x refitted folds on the full path; `cv_glmnet_gaussian(fold_lambda="full")`).
* **glmnet**: py-Pando's coordinate descent follows glmnet's path rules (standardisation, lambda sequence, `fdev` / `devmax` stopping) but not its strong-rule screening, so for genes with hundreds of variables both stop within glmnet's 1e-7 tolerance at slightly different coefficients. `fit_glmnet` ports `family = "gaussian"` given as a string, which is how `infer_grn` calls it; R's `fit_glmnet` called directly with its default `family = gaussian` (a function) goes to glmnet's generic `glmnet.path` solver, which is not ported.
* **xgb**: Pando calls `xgboost::xgboost(data, label, params = list(max_depth = 3, eta = 0.01), ...)`. xgboost >= 3 changed that function's interface and silently ignores `params`, training with depth 6 and learning rate 0.3. py-Pando uses Pando's parameters; with them honoured in R (`xgb.train`), importances match to 1e-15.
* **bagging_ridge** draws from NumPy's global RNG, as Pando's scikit-learn call does from R; `infer_grn(seed=...)` seeds it.
* **New in Python** (not in R): `region_background`, `regions_near_genes` and `subset_regions` scan only the regions a target gene can reach, against the background of all regions — the network is identical to a full scan.

## Reproducing the R comparison

```bash
python tests/make_fixture.py pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 data/fixture
Rscript tests/r_reference_driver.R data/fixture ref                     # R 4.4.3, Pando 1.1.1
python tests/_run_candidate.py ref cand hg38.fa
python tests/parity_check.py ref cand results.json
PYPANDO_REF_DIR=ref PYPANDO_FASTA=hg38.fa PYPANDO_CAND_DIR=cand pytest tests/test_exact_match.py
```

## Citation

Fleck, J.S., Jansen, S.M.J., Wollny, D. *et al.* Inferring and perturbing cell fate regulomes in human brain organoids. *Nature* 621, 365–372 (2023).

## License

MIT, as the upstream R package (see `LICENSE`).
