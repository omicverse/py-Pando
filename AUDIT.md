## R function coverage audit

### Coverage summary

| Category | Ported | Total | % |
|---|---|---|---|
| Exported R functions | 30 | 51 | 58.8% |
| Internal helpers (reachable) | 1 | 7 | 14.3% |

_Python package exposes 127 unique names._

### Exported R functions

| R function | Python equivalent | Status |
|---|---|---|
| `DefaultNetwork` | `DefaultNetwork` | ✅ ported |
| `GRNData` | `—` | ❌ MISSING |
| `GetAssay` | `—` | ❌ MISSING |
| `GetAssaySummary` | `—` | ❌ MISSING |
| `GetGRN` | `GetGRN` | ✅ ported |
| `GetNetwork` | `GetNetwork` | ✅ ported |
| `LayerData` | `—` | ❌ MISSING |
| `Network` | `—` | ❌ MISSING |
| `NetworkFeatures` | `NetworkFeatures` | ✅ ported |
| `NetworkGraph` | `NetworkGraph` | ✅ ported |
| `NetworkModules` | `NetworkModules` | ✅ ported |
| `NetworkParams` | `NetworkParams` | ✅ ported |
| `NetworkRegions` | `NetworkRegions` | ✅ ported |
| `NetworkTFs` | `NetworkTFs` | ✅ ported |
| `Params` | `Params` | ✅ ported |
| `Regions` | `—` | ❌ MISSING |
| `RegulatoryNetwork` | `—` | ❌ MISSING |
| `VariableFeatures` | `—` | ❌ MISSING |
| `aggregate_assay` | `—` | ❌ MISSING |
| `aggregate_matrix` | `aggregate_matrix` | ✅ ported |
| `coef` | `coef` | ✅ ported |
| `create_GRNData` | `—` | ❌ MISSING |
| `find_modules` | `find_modules` | ✅ ported |
| `find_motifs` | `find_motifs` | ✅ ported |
| `find_peaks_near_genes` | `find_peaks_near_genes` | ✅ ported |
| `fit_bagging_ridge` | `fit_bagging_ridge` | ✅ ported |
| `fit_bayesian_ridge` | `fit_bayesian_ridge` | ✅ ported |
| `fit_brms` | `fit_brms` | ✅ ported |
| `fit_cvglmnet` | `fit_cvglmnet` | ✅ ported |
| `fit_glm` | `fit_glm` | ✅ ported |
| `fit_glmnet` | `fit_glmnet` | ✅ ported |
| `fit_grn_models` | `fit_grn_models` | ✅ ported |
| `fit_model` | `fit_model` | ✅ ported |
| `fit_xgb` | `fit_xgb` | ✅ ported |
| `format_coefs` | `format_coefs` | ✅ ported |
| `get_network_graph` | `get_network_graph` | ✅ ported |
| `get_tf_network` | `—` | ❌ MISSING |
| `gof` | `gof` | ✅ ported |
| `infer_grn` | `infer_grn` | ✅ ported |
| `initiate_grn` | `initiate_grn` | ✅ ported |
| `map_par` | `—` | ❌ MISSING |
| `no_legend` | `—` | ❌ MISSING |
| `no_margin` | `—` | ❌ MISSING |
| `no_x_text` | `—` | ❌ MISSING |
| `no_y_text` | `—` | ❌ MISSING |
| `plot_gof` | `—` | ❌ MISSING |
| `plot_module_metrics` | `—` | ❌ MISSING |
| `plot_network_graph` | `—` | ❌ MISSING |
| `plot_tf_network` | `—` | ❌ MISSING |
| `print` | `—` | ❌ MISSING |
| `sparse_cor` | `sparse_cor` | ✅ ported |

### Internal helpers reachable from exports

| R helper | File | Python equivalent | Status |
|---|---|---|---|
| `dMcast` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `extract_rows` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `fast_aggregate` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `r2` | `models.R` | `r2` | ✅ ported |
| `rse` | `models.R` | `—` | 🔸 missing-or-inlined |
| `sparse_covcor` | `utils.R` | `—` | 🔸 missing-or-inlined |
| `sse` | `models.R` | `—` | 🔸 missing-or-inlined |
