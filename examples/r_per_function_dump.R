#!/usr/bin/env Rscript
# Per-function R outputs for examples/function_by_function_R_parity.ipynb.
#
#   Rscript examples/r_per_function_dump.R <out_dir>
#
# Writes a small deterministic input (design data frame, ranges, matrices) as CSV and
# Pando's output of each exported function on it, so Python can call the same function
# on the same numbers. RETICULATE_PYTHON must point at a Python with scikit-learn for
# the two ridge models.
suppressPackageStartupMessages({
  library(Pando); library(GenomicRanges); library(Matrix); library(jsonlite); library(data.table)
})
out <- commandArgs(trailingOnly = TRUE)[[1]]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
SEED <- 42L

# ---- shared input: a per-gene design like the ones infer_grn builds ----------
set.seed(1)
n <- 300
d <- data.frame(
  chr1_100_600 = rexp(n), chr1_900_1400 = rexp(n), chr2_50_550 = rexp(n),
  TFA = rnorm(n, 2), TFB = rnorm(n, 1), NKX2_1 = rnorm(n, 1.5)
)
d$GENE <- 0.8 * d$chr1_100_600 * d$TFA - 0.5 * d$chr2_50_550 * d$NKX2_1 + rnorm(n)
fwrite(d, file.path(out, "design.csv"))
frm <- GENE ~ chr1_100_600:TFA + chr1_100_600:TFB + chr1_900_1400:TFA + chr2_50_550:NKX2_1 + chr2_50_550:TFB

dump <- function(res, name) {
  fwrite(as.data.frame(res$coefs), file.path(out, paste0(name, "_coefs.csv")))
  fwrite(as.data.frame(res$gof), file.path(out, paste0(name, "_gof.csv")))
}
dump(fit_glm(frm, d), "fit_glm")
# family as infer_grn passes it (a character string -> glmnet's elnet path); fit_glmnet's own
# default, the function `gaussian`, sends glmnet to its generic glmnet.path solver instead
dump(fit_glmnet(frm, d, family = "gaussian", alpha = 0.5), "fit_glmnet")
set.seed(SEED); dump(fit_cvglmnet(frm, d, family = "gaussian", alpha = 0.5), "fit_cvglmnet")
dump(fit_xgb(frm, d), "fit_xgb")
# diagnostic: xgboost >= 3's xgboost() drops Pando's `params`; the same model with them honoured
mm <- stats::model.matrix(frm, data = d)
bst <- xgboost::xgb.train(params = list(max_depth = 3, eta = 0.01, objective = "reg:squarederror", nthread = 1),
                          data = xgboost::xgb.DMatrix(mm, label = d$GENE), nrounds = 1000, verbose = 0)
imp <- as.data.frame(xgboost::xgb.importance(model = bst)); colnames(imp) <- c("term", "gain", "cover", "frequency")
yp <- predict(bst, newdata = mm)
dump(list(coefs = imp, gof = data.frame(rsq = 1 - sum((d$GENE - yp)^2) / sum((d$GENE - mean(d$GENE))^2))), "fit_xgb_params_honoured")
np <- reticulate::import("numpy", convert = FALSE); np$random$seed(SEED)
dump(fit_bagging_ridge(frm, d, alpha = 0.5), "fit_bagging_ridge")
dump(fit_bayesian_ridge(frm, d), "fit_bayesian_ridge")
set.seed(SEED); dump(fit_model(frm, d, method = "glm"), "fit_model")

# format_coefs on the glm table with a target column, as fit_grn_models calls it
cf <- fit_glm(frm, d)$coefs; cf <- cbind(target = "GENE", cf)
fwrite(as.data.frame(format_coefs(cf, term = ":", adjust_method = "fdr")), file.path(out, "format_coefs.csv"))

# sparse_cor and aggregate_matrix
X <- Matrix(as.matrix(d[, c("TFA", "TFB", "NKX2_1")]), sparse = TRUE)
Y <- Matrix(as.matrix(d[, "GENE", drop = FALSE]), sparse = TRUE)
fwrite(as.data.frame(as.matrix(sparse_cor(X, Y))), file.path(out, "sparse_cor.csv"))
fwrite(as.data.frame(as.matrix(sparse_cor(X))), file.path(out, "sparse_cor_self.csv"))
groups <- rep(c("g2", "g1", "g3"), length.out = n)
writeLines(groups, file.path(out, "groups.txt"))
for (f in c("mean", "sum", "max", "count")) {
  a <- as.matrix(aggregate_matrix(as.matrix(d[, 1:6]), groups = groups, fun = f))
  fwrite(cbind(group = rownames(a), as.data.frame(a)), file.path(out, paste0("aggregate_matrix_", f, ".csv")))
}

# find_peaks_near_genes on small ranges
peaks <- Signac::StringToGRanges(c("chr1-1000-1500", "chr1-90000-90500", "chr1-200500-201000",
                           "chr1-300100-300300", "chr2-5000-5400"))
genes <- GRanges(c("chr1", "chr1", "chr1", "chr2"),
                 IRanges(c(100000, 150000, 300000, 9000), c(120000, 160000, 305000, 9500)),
                 strand = c("+", "-", "+", "-"), gene_name = c("G1", "G2", "G3", "G4"))
fwrite(data.table(seqnames = as.character(seqnames(genes)), start = start(genes), end = end(genes),
                  strand = as.character(strand(genes)), gene_name = genes$gene_name),
       file.path(out, "genes.csv"))
writeLines(Signac::GRangesToString(peaks), file.path(out, "peaks.txt"))
for (only in c(FALSE, TRUE)) {
  h <- Pando:::find_peaks_near_genes(peaks, genes, method = "Signac", upstream = 100000,
                                     downstream = 1000, only_tss = only)
  s <- summary(h)
  fwrite(data.table(i = s$i, j = s$j, x = s$x), file.path(out, paste0("find_peaks_near_genes_tss", only, ".csv")))
}
cat("done\n")
