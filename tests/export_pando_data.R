#!/usr/bin/env Rscript
# Export Pando's bundled data objects to plain files for py-Pando.
#
#   Rscript tests/export_pando_data.R <pkg_data_dir> <asset_dir>
#
# pkg_data_dir (shipped in the wheel): motifs.json.gz, motif2tf.tsv.gz, ensdb_v93_hg38.tsv.gz
# asset_dir    (GitHub release assets, fetched with pooch): phastcons20_hg38.tsv.gz, screen_ccre_hg38.tsv.gz
suppressPackageStartupMessages({
  library(Pando); library(TFBSTools); library(GenomicRanges); library(jsonlite); library(data.table)
})
args <- commandArgs(trailingOnly = TRUE)
pkg <- args[[1]]; assets <- args[[2]]
dir.create(pkg, showWarnings = FALSE, recursive = TRUE)
dir.create(assets, showWarnings = FALSE, recursive = TRUE)

data(motifs, package = "Pando")
mlist <- lapply(seq_along(motifs), function(i) {
  m <- motifs[[i]]
  mat <- Matrix(m)[c("A", "C", "G", "T"), , drop = FALSE]
  list(ID = names(motifs)[[i]], name = name(m), bg = unname(as.numeric(bg(m)[c("A", "C", "G", "T")])),
       counts = unname(lapply(seq_len(4), function(r) as.integer(mat[r, ]))))
})
con <- gzfile(file.path(pkg, "motifs.json.gz"), "w")
writeLines(toJSON(mlist, auto_unbox = TRUE, digits = NA), con); close(con)

data(motif2tf, package = "Pando")
fwrite(as.data.frame(motif2tf), file.path(pkg, "motif2tf.tsv.gz"), sep = "\t", na = "")

gr_df <- function(gr) {
  d <- as.data.frame(gr)
  names(d)[names(d) == "width"] <- "width_"
  d$width_ <- NULL
  d$seqnames <- as.character(d$seqnames); d$strand <- as.character(d$strand)
  d
}
data(EnsDb.Hsapiens.v93.annot.UCSC.hg38, package = "Pando")
fwrite(gr_df(EnsDb.Hsapiens.v93.annot.UCSC.hg38), file.path(pkg, "ensdb_v93_hg38.tsv.gz"), sep = "\t", na = "")

data(phastConsElements20Mammals.UCSC.hg38, package = "Pando")
fwrite(gr_df(phastConsElements20Mammals.UCSC.hg38), file.path(assets, "phastcons20_hg38.tsv.gz"), sep = "\t", na = "")
data(SCREEN.ccRE.UCSC.hg38, package = "Pando")
fwrite(gr_df(SCREEN.ccRE.UCSC.hg38), file.path(assets, "screen_ccre_hg38.tsv.gz"), sep = "\t", na = "")
cat("exported", length(motifs), "motifs\n")
