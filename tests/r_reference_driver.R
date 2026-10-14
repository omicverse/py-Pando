#!/usr/bin/env Rscript
# R reference for py-Pando's parity gates (data/manifest.yaml).
#
#   Rscript tests/r_reference_driver.R <fixture_dir> <out_dir> [methods]
#
# <fixture_dir> holds the raw subset written by tests/make_fixture.py. This script
#   1. builds the Seurat/Signac object (LogNormalize, vst variable genes, peak window, TF-IDF),
#   2. writes the shared inputs Python reads (so both sides start from identical numbers),
#   3. runs Pando and writes every output the manifest gates.
# Requires RETICULATE_PYTHON to point at a Python with scikit-learn for the ridge methods.
suppressPackageStartupMessages({
  library(Matrix); library(data.table); library(jsonlite)
  library(Seurat); library(Signac); library(GenomicRanges); library(IRanges)
  library(EnsDb.Hsapiens.v86); library(BSgenome.Hsapiens.UCSC.hg38)
  library(Pando)
})
args <- commandArgs(trailingOnly = TRUE)
fx <- args[[1]]; out <- args[[2]]
methods <- if (length(args) >= 3) strsplit(args[[3]], ",")[[1]] else
  c("glm", "glmnet", "cv.glmnet", "xgb", "bagging_ridge", "bayesian_ridge")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
shared <- file.path(out, "shared"); dir.create(shared, showWarnings = FALSE)
timing <- list()
tic <- function() proc.time()[["elapsed"]]
SEED <- 42L
N_TARGETS <- 150L
WINDOW <- 250000L

# ---- 1. fixture object -------------------------------------------------------
rna <- readMM(file.path(fx, "rna_counts.mtx")); peaks <- readMM(file.path(fx, "peak_counts.mtx"))
genes <- fread(file.path(fx, "genes.tsv"), header = FALSE)
cells <- readLines(file.path(fx, "barcodes.txt")); peak_names <- readLines(file.path(fx, "peaks.txt"))
rna <- as(rna, "CsparseMatrix"); peaks <- as(peaks, "CsparseMatrix")
rownames(rna) <- make.unique(genes$V2); colnames(rna) <- cells
rownames(peaks) <- peak_names; colnames(peaks) <- cells

obj <- CreateSeuratObject(counts = rna, assay = "RNA")
obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)

annot <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86, verbose = FALSE)
seqlevels(annot) <- paste0("chr", seqlevels(annot))
genome(annot) <- "hg38"

vf <- Seurat::VariableFeatures(obj)
targets <- head(vf[vf %in% annot$gene_name], N_TARGETS)

# peaks on standard chromosomes within WINDOW of the target genes' annotation
pr <- StringToGRanges(peak_names, sep = c("-", "-"))
std <- as.character(seqnames(pr)) %in% paste0("chr", c(1:22, "X", "Y"))
tg <- annot[annot$gene_name %in% targets]
tg_win <- reduce(Extend(tg, upstream = WINDOW, downstream = WINDOW), ignore.strand = TRUE)
near <- overlapsAny(pr, tg_win)
keep <- std & near
peaks <- peaks[keep, ]
obj[["peaks"]] <- CreateChromatinAssay(counts = peaks, sep = c("-", "-"), annotation = annot)
DefaultAssay(obj) <- "peaks"
obj <- RunTFIDF(obj, verbose = FALSE)
DefaultAssay(obj) <- "RNA"

# ---- 2. shared inputs ----------------------------------------------------------
rna_data <- SeuratObject::LayerData(obj, assay = "RNA", layer = "data")
peak_data <- SeuratObject::LayerData(obj, assay = "peaks", layer = "data")
writeMM(rna_data, file.path(shared, "rna_data.mtx"))
writeMM(peak_data, file.path(shared, "peak_data.mtx"))
writeLines(rownames(rna_data), file.path(shared, "rna_genes.txt"))
writeLines(rownames(peak_data), file.path(shared, "peaks.txt"))
writeLines(colnames(rna_data), file.path(shared, "cells.txt"))
writeLines(vf, file.path(shared, "variable_features.txt"))
writeLines(targets, file.path(shared, "targets.txt"))
ad_gr <- annot; names(ad_gr) <- NULL; ad <- as.data.frame(ad_gr)
ad$seqnames <- as.character(ad$seqnames); ad$strand <- as.character(ad$strand); ad$width <- NULL
fwrite(ad, file.path(shared, "annotation.tsv"), sep = "\t", na = "")
# binary dumps keep every bit of the doubles (writeMM prints 15 significant digits)
writeBin(as.numeric(rna_data@x), file.path(shared, "rna_data_x.f64"))
writeBin(as.numeric(peak_data@x), file.path(shared, "peak_data_x.f64"))

# ---- 3. Pando ----------------------------------------------------------------------
data(motifs, package = "Pando"); data(motif2tf, package = "Pando")

t0 <- tic(); grn <- initiate_grn(obj, rna_assay = "RNA", peak_assay = "peaks"); timing$initiate_grn <- tic() - t0
reg <- NetworkRegions(grn)
fwrite(data.table(region = GRangesToString(reg@ranges), peak = reg@peaks),
       file.path(out, "candidate_regions.tsv"), sep = "\t")

data(phastConsElements20Mammals.UCSC.hg38, package = "Pando")
t0 <- tic()
grn_pc <- initiate_grn(obj, regions = phastConsElements20Mammals.UCSC.hg38, rna_assay = "RNA", peak_assay = "peaks")
timing$initiate_grn_phastcons <- tic() - t0
reg_pc <- NetworkRegions(grn_pc)
fwrite(data.table(region = GRangesToString(reg_pc@ranges), peak = reg_pc@peaks),
       file.path(out, "candidate_regions_phastcons.tsv"), sep = "\t")

t0 <- tic(); grn <- find_motifs(grn, pfm = motifs, genome = BSgenome.Hsapiens.UCSC.hg38, verbose = FALSE)
timing$find_motifs <- tic() - t0
reg <- NetworkRegions(grn)
mm <- summary(as(reg@motifs@data, "CsparseMatrix"))
fwrite(data.table(i = mm$i, j = mm$j), file.path(out, "motif_matrix_ij.tsv"), sep = "\t")
writeLines(rownames(reg@motifs@data), file.path(out, "motif_matrix_rows.txt"))
writeLines(colnames(reg@motifs@data), file.path(out, "motif_matrix_cols.txt"))
writeLines(colnames(reg@tfs), file.path(out, "tfs_use.txt"))

# find_peaks_near_genes on the candidate regions, both methods
ga <- Annotation(obj[["peaks"]]); ga <- ga[ga$type != "gap"]; ga <- ga[ga$gene_name %in% targets]
for (m in c("Signac", "GREAT")) {
  t0 <- tic()
  h <- Pando:::find_peaks_near_genes(peaks = reg@ranges, genes = ga, method = m,
                                     upstream = 100000, downstream = 0, only_tss = FALSE)
  timing[[paste0("find_peaks_near_genes_", m)]] <- tic() - t0
  s <- summary(h)
  fwrite(data.table(i = s$i, j = s$j, x = s$x), file.path(out, paste0("peaks_near_genes_", m, "_ij.tsv")), sep = "\t")
  writeLines(colnames(h), file.path(out, paste0("peaks_near_genes_", m, "_cols.txt")))
}

# Diagnostic only (PANDO_XGB_HONOUR_PARAMS=1): xgboost >= 3's xgboost() ignores the `params` list
# Pando passes, so the reference trains with xgboost defaults. This swaps in the same model with
# Pando's parameters honoured (xgb.train), to separate that from any porting difference.
if (identical(Sys.getenv("PANDO_XGB_HONOUR_PARAMS"), "1")) {
  fit_xgb_honoured <- function(formula, data, params = list(max_depth = 3, eta = 0.01,
                               objective = "reg:squarederror"), nrounds = 1000, nthread = 1, ...) {
    model_mat <- stats::model.matrix(formula, data = data)
    response <- data[[formula[[2]]]]
    fit <- xgboost::xgb.train(params = c(params, nthread = nthread),
                              data = xgboost::xgb.DMatrix(model_mat, label = response),
                              nrounds = nrounds, verbose = 0)
    y_pred <- predict(fit, newdata = model_mat)
    gof <- tibble::tibble(rsq = 1 - sum((response - y_pred)^2) / sum((response - mean(response))^2))
    coefs <- tibble::as_tibble(as.data.frame(xgboost::xgb.importance(model = fit)))
    colnames(coefs) <- c("term", "gain", "cover", "frequency")
    list(gof = gof, coefs = coefs)
  }
  assignInNamespace("fit_xgb", fit_xgb_honoured, ns = "Pando")
  message("fit_xgb patched: Pando's params honoured via xgb.train")
}

# ---- TF sub-networks (PANDO_TF_NETWORK=1) -------------------------------------------
# get_tf_network as installed, plus a reference implementing what its default path intends
# (see DISCOVERY.md, "get_tf_network"): prune on the per-path mean -log10(padj) it computes
# (`mean_log_padj`), and honour the `order` argument. The patched copy is a reference only.
tf_network_intent <- function() {
  f <- Pando:::get_tf_network.GRNData
  src <- paste(deparse(f, width.cutoff = 500L), collapse = "\n")
  subs <- c('"padj" %in% colnames(grn_pruned)' = '"mean_log_padj" %in% colnames(grn_pruned)',
            "mean_padj == max(mean_padj)" = "mean_log_padj == max(mean_log_padj)",
            "filter(order <= order)" = "filter(order <= .env$order)")
  for (k in names(subs)) {
    stopifnot(lengths(regmatches(src, gregexpr(k, src, fixed = TRUE))) == 1L)
    src <- sub(k, subs[[k]], src, fixed = TRUE)
  }
  g <- eval(parse(text = src)); environment(g) <- asNamespace("Pando"); g
}
export_tf_networks <- function(g, out) {
  saveRDS(g, file.path(out, "grn_glm_modules.rds"))
  # Pando 1.1.1's get_network_graph calls tidyr::pivot_wider without importing it; attach tidyr.
  suppressPackageStartupMessages(library(tidyr))
  g <- get_network_graph(g, verbose = FALSE)
  mg <- NetworkGraph(g, graph = "module_graph")
  ed <- as.data.frame(tidygraph::as_tibble(tidygraph::activate(mg, "edges")))
  deg <- table(ed$from_node)
  deg <- deg[order(-as.integer(deg), names(deg), method = "radix")]
  tfs <- head(names(deg), 12L)
  writeLines(tfs, file.path(out, "tf_network_tfs.txt"))
  fwrite(ed[, setdiff(names(ed), c("from", "to"))], file.path(out, "module_graph_edges.tsv"), sep = "\t", na = "NA")
  intent <- tf_network_intent()
  variants <- list(
    R_all = function(tf) get_tf_network(g, tf = tf, keep_all_edges = TRUE, verbose = FALSE),
    R_default = function(tf) get_tf_network(g, tf = tf, verbose = FALSE),
    intent_default = function(tf) intent(g, tf = tf, verbose = FALSE),
    intent_all = function(tf) intent(g, tf = tf, keep_all_edges = TRUE, verbose = FALSE),
    intent_default_o2 = function(tf) intent(g, tf = tf, order = 2, verbose = FALSE))
  rows <- list(); status <- list()
  for (v in names(variants)) for (tf in tfs) {
    t0 <- tic()
    res <- tryCatch(variants[[v]](tf), error = function(e) conditionMessage(e))
    key <- paste(v, tf, sep = "/")
    if (is.character(res)) { status[[key]] <- paste("error:", res); next }
    tg <- NetworkGraph(res, graph = "tf_graphs")[[tf]]
    e <- as.data.frame(tidygraph::as_tibble(tidygraph::activate(tg, "edges")))
    n <- tidygraph::as_tibble(tidygraph::activate(tg, "nodes"))$name
    e$from_name <- n[e$from]; e$to_name <- n[e$to]
    e$variant <- v; e$root <- tf; e$edge_index <- seq_len(nrow(e))
    rows[[key]] <- e[, setdiff(names(e), c("from", "to"))]
    status[[key]] <- sprintf("ok: %d edges, %d nodes, %.1f s", nrow(e), length(n), tic() - t0)
  }
  fwrite(rbindlist(rows, fill = TRUE), file.path(out, "tf_networks.tsv"), sep = "\t", na = "NA")
  writeBin(as.numeric(rbindlist(rows, fill = TRUE)$estimate), file.path(out, "tf_networks_estimate.f64"))
  write(toJSON(status, auto_unbox = TRUE, pretty = TRUE), file.path(out, "tf_networks_status.json"))
}

np_seed <- function() {
  np <- reticulate::import("numpy", convert = FALSE); np$random$seed(SEED)
}
for (m in methods) {
  message("infer_grn: ", m)
  set.seed(SEED)
  if (m == "bagging_ridge") np_seed()
  t0 <- tic()
  g <- infer_grn(grn, genes = targets, method = m, parallel = FALSE, verbose = FALSE)
  timing[[paste0("infer_grn_", m)]] <- tic() - t0
  cf <- as.data.frame(coef(g)); gf <- as.data.frame(gof(g))
  fwrite(cf, file.path(out, paste0("coefs_", m, ".tsv")), sep = "\t", na = "NA")
  fwrite(gf, file.path(out, paste0("gof_", m, ".tsv")), sep = "\t", na = "NA")
  # full-precision copies of the numeric columns
  for (col in intersect(names(cf), c("estimate", "std_err", "statistic", "pval", "padj", "corr", "gain", "cover", "frequency", "est_variance", "neglog10p"))) {
    writeBin(as.numeric(cf[[col]]), file.path(out, paste0("coefs_", m, "_", col, ".f64")))
  }
  for (col in intersect(names(gf), c("rsq", "lambda", "nvariables", "alpha"))) {
    writeBin(as.numeric(gf[[col]]), file.path(out, paste0("gof_", m, "_", col, ".f64")))
  }
  if (m == "glm") {
    t0 <- tic(); g <- find_modules(g); timing$find_modules <- tic() - t0
    mods <- NetworkModules(g)
    fwrite(as.data.frame(mods@meta), file.path(out, "modules_meta.tsv"), sep = "\t", na = "NA")
    write(toJSON(mods@features, auto_unbox = FALSE), file.path(out, "modules_features.json"))
    if (identical(Sys.getenv("PANDO_TF_NETWORK"), "1")) export_tf_networks(g, out)
  }
}
si <- sessionInfo()
write(toJSON(list(timing = timing, R = R.version.string,
                  packages = lapply(c(Pando = "Pando", Seurat = "Seurat", Signac = "Signac",
                                      motifmatchr = "motifmatchr", glmnet = "glmnet",
                                      glmnetUtils = "glmnetUtils", xgboost = "xgboost"),
                                    function(p) as.character(packageVersion(p))),
                  blas = si$BLAS, seed = SEED, n_targets = length(targets),
                  n_peaks = nrow(peak_data), n_cells = ncol(peak_data)),
             auto_unbox = TRUE, pretty = TRUE), file.path(out, "run.json"))
cat("done\n")
