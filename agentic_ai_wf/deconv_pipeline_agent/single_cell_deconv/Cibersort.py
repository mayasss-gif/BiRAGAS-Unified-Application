#!/usr/bin/env python3
"""
cibersort_Cibersort.py — Run Sheryar's CIBERSORT (LM22) pipeline and QA via rpy2.

Usage examples:
  python cibersort_runner.py \
    --counts Bulk-data/exampleForLUAD.txt \
    --lm22 inst/extdata/LM22.txt \
    --out CIBERSORT_outputs-v-11-2-1-0 \
    --perm 100 --qn false --chunk-size 60 --install

  # with metadata
  python cibersort_runner.py \
    --counts Bulk-data/exampleForLUAD.txt \
    --meta Bulk-data/sample_metadata_patientAKI.tsv \
    --lm22 inst/extdata/LM22.txt \
    --out outputs --perm 1000 --qn false --install
"""
from __future__ import annotations
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


LM22_PATH = Path(__file__).parent / "inst" / "extdata" / "LM22.txt"


# ---------------------------------------------------------------------
# Lazy rpy2 import for Celery fork-safety
# ---------------------------------------------------------------------
def _lazy_init_rpy2():
    """
    Lazy initialize rpy2 to avoid fork-safety issues in Celery.
    
    CRITICAL: This must be called INSIDE the task/function that uses R,
    NOT at module import time. Otherwise, R gets initialized in the parent
    Celery worker process before forking, which causes deadlocks.
    
    Returns:
        tuple: (ro, rpackages, StrVector) - rpy2 objects ready to use
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import packages as rpackages
        from rpy2.robjects.vectors import StrVector
        from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
        
        # Quiet down rpy2's own verbose logging
        rpy2_logger.setLevel(logging.ERROR)
        
        return ro, rpackages, StrVector
    except Exception as e:
        logging.error(
            "ERROR: rpy2 is required. Install with:\n"
            "  pip install rpy2\n\n"
            "Also ensure system R is installed and in PATH (R >= 4.x)."
        )
        raise RuntimeError("rpy2 initialization failed") from e

# ---------------------------------------------------------------------
# R code blocks (minimally edited to accept variables from Python)
# ---------------------------------------------------------------------
R_MAIN_PIPELINE = r'''
options(stringsAsFactors = FALSE)
set.seed(123)

counts_path <- Sys.getenv("PY_COUNTS_PATH")
meta_path   <- Sys.getenv("PY_META_PATH")
lm22_path   <- Sys.getenv("PY_LM22_PATH")
out_dir     <- Sys.getenv("PY_OUT_DIR")
perm_val    <- as.integer(Sys.getenv("PY_PERM"))
qn_flag     <- as.logical(tolower(Sys.getenv("PY_QN")))
chunk_size  <- as.integer(Sys.getenv("PY_CHUNK_SIZE"))

if (!dir.exists(out_dir)) dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ipkgs <- c("data.table","readr","readxl","dplyr","tibble","stringr",
           "tools","ggplot2","pheatmap","reshape2","tidyr","ggrepel","scales")
missing <- ipkgs[!sapply(ipkgs, requireNamespace, quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = "https://cloud.r-project.org")

# Install Bioconductor dependencies required by CIBERSORT
bioc_pkgs <- c("edgeR", "preprocessCore")
for (pkg in bioc_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

if (!requireNamespace("CIBERSORT", quietly = TRUE)) {
  local_path <- "CIBERSORT-main"
  if (dir.exists(local_path)) {
    if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools", repos = "https://cloud.r-project.org")
    devtools::install_local(local_path, upgrade = "never", dependencies = TRUE)
  }
}
if (!requireNamespace("CIBERSORT", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools", repos = "https://cloud.r-project.org")
  devtools::install_github("Moonerss/CIBERSORT", upgrade = "never", dependencies = TRUE)
}

library(CIBERSORT)
library(edgeR)
library(data.table); library(readr); library(readxl); library(dplyr); library(tibble)
library(stringr); library(ggplot2); library(pheatmap); library(reshape2); library(tidyr)
library(ggrepel); library(scales)

ensure_df <- function(x) {
  if (is.null(x)) stop("Object is NULL; earlier read/compute step failed.")
  if (is.matrix(x)) return(as.data.frame(x, stringsAsFactors = FALSE))
  if (inherits(x, "tbl_df")) return(as.data.frame(x))
  if (is.data.frame(x)) return(x)
  stop(sprintf("Expected a data.frame; got class: %s", paste(class(x), collapse = ", ")))
}
smart_read <- function(path) {
  if (!file.exists(path)) return(NULL)
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("tsv","txt")) readr::read_tsv(path, guess_max = 1e6, show_col_types = FALSE)
  else if (ext %in% c("csv","csvl")) readr::read_csv(path, guess_max = 1e6, show_col_types = FALSE)
  else if (ext %in% c("xls","xlsx")) readxl::read_excel(path)
  else stop(paste("Unsupported file extension:", ext))
}
detect_meta_columns <- function(df) {
  # Flexible column detection (trust upstream validation)
  names(df) <- tolower(gsub("[^A-Za-z0-9]+","_", names(df)))
  
  # Find sample column
  sample_col <- if("sample_id" %in% names(df)) "sample_id"
  else if("sample" %in% names(df)) "sample"
  else if("sampleid" %in% names(df)) "sampleid"
  else if("sample_name" %in% names(df)) "sample_name"
  else names(df)[1]  # Fallback to first column
  
  # Find condition column
  condition_col <- if("condition" %in% names(df)) "condition"
  else if("group" %in% names(df)) "group"
  else if("treatment" %in% names(df)) "treatment"
  else if("class" %in% names(df)) "class"
  else if(length(names(df)) > 1) names(df)[2]  # Fallback to second column
  else NULL
  
  if(is.null(condition_col)) return(list(ok=FALSE, sample=NULL, condition=NULL))
  
  # Standardize to expected names
  df_std <- df
  df_std$sample_id <- trimws(as.character(df[[sample_col]]))
  df_std$condition <- trimws(as.character(df[[condition_col]]))
  
  list(ok=TRUE, data=df_std, sample_col=sample_col, condition_col=condition_col)
}

message("Loading counts and (optional) metadata ...")
counts_raw <- smart_read(counts_path)
if (is.null(counts_raw)) stop("Counts file not found: ", counts_path)
counts_raw <- as.data.frame(counts_raw)

meta <- NULL; meta_present <- FALSE
if (!is.na(meta_path) && nzchar(meta_path) && file.exists(meta_path)) {
  meta_raw <- smart_read(meta_path)
  if (!is.null(meta_raw)) {
    meta_result <- detect_meta_columns(as.data.frame(meta_raw))
    if (meta_result$ok) {
      meta <- meta_result$data
      meta_present <- TRUE
      message("✅ Metadata loaded: ", nrow(meta), " samples, columns detected: ",
              meta_result$sample_col, " → sample_id, ", meta_result$condition_col, " → condition")
    } else {
      message("Metadata found but could not detect sample/condition columns → proceeding WITHOUT metadata graphics.")
    }
  } else {
    message("Metadata file path given but could not be read → proceeding WITHOUT metadata graphics.")
  }
} else {
  message("No metadata file used → proceeding WITHOUT metadata graphics.")
}

gene_col_candidates <- c("Gene","gene","GeneSymbol","Gene_Symbol","Symbol","SYMBOL","GeneSymbolID","Name","id","ID")
gene_col <- intersect(gene_col_candidates, colnames(counts_raw))
gene_col <- if (length(gene_col) == 0) colnames(counts_raw)[1] else gene_col[1]
counts_raw <- counts_raw %>% dplyr::rename(GeneSymbol = dplyr::all_of(gene_col))

# CRITICAL: Remove rows with missing/empty gene symbols BEFORE setting row names
# R does not allow NA/empty values in row names
counts_raw$GeneSymbol <- as.character(counts_raw$GeneSymbol)
counts_raw <- counts_raw %>% dplyr::filter(!is.na(GeneSymbol), trimws(GeneSymbol) != "")
if (nrow(counts_raw) == 0) stop("No valid gene symbols found after removing NA/empty values")

# Make unique and set as row names
counts_raw$GeneSymbol <- make.unique(trimws(counts_raw$GeneSymbol))
rownames(counts_raw) <- counts_raw$GeneSymbol
counts_raw$GeneSymbol <- NULL

count_mat <- as.matrix(counts_raw); mode(count_mat) <- "numeric"

if (meta_present) {
  common_samples <- intersect(colnames(count_mat), meta$sample_id)
  if (length(common_samples) == 0) {
    warning("No overlapping sample IDs between counts and metadata → proceeding WITHOUT metadata.")
    meta_present <- FALSE
  } else {
    meta <- meta %>% dplyr::filter(sample_id %in% common_samples)
    count_mat <- count_mat[, meta$sample_id, drop = FALSE]
  }
}

message("QC: library sizes ...")
lib_sizes <- colSums(count_mat, na.rm = TRUE)
qc_df <- data.frame(sample = names(lib_sizes), library_size = lib_sizes, stringsAsFactors = FALSE)
if (meta_present) qc_df <- dplyr::left_join(qc_df, meta, by = c("sample" = "sample_id"))
data.table::fwrite(qc_df, file.path(out_dir, "QC_library_sizes.csv"))

count_mat <- count_mat[rowSums(count_mat, na.rm = TRUE) > 0, , drop = FALSE]

message("Checking scale (counts vs TPM/RPKM) ...")
med_sum <- median(colSums(count_mat, na.rm = TRUE))
use_direct <- is.finite(med_sum) && med_sum > 5e5 && med_sum < 2e6
if (use_direct) {
  message("Input looks like TPM/RPKM (sample sums ≈ 1e6). Using matrix as-is for CIBERSORT (QN=FALSE).")
  expr_mat <- count_mat
} else {
  message("Input looks like counts. Converting to CPM (TMM-normalized).")
  dge <- edgeR::DGEList(counts = count_mat)
  dge <- edgeR::calcNormFactors(dge, method = "TMM")
  expr_mat <- edgeR::cpm(dge, normalized.lib.sizes = TRUE, log = FALSE)
}

keep <- rowSums(expr_mat > 1) >= max(1, floor(ncol(expr_mat) * 0.1))
expr_mat <- expr_mat[keep, , drop = FALSE]

mixture_df <- ensure_df(expr_mat) %>% tibble::rownames_to_column("GeneSymbol")
mixture_out <- file.path(out_dir, "mixture_for_CIBERSORT.txt")
write.table(mixture_df, file = mixture_out, sep = "\t", row.names = FALSE, quote = FALSE)
stopifnot(file.exists(lm22_path))

lm22_df <- readr::read_tsv(lm22_path, show_col_types = FALSE)
colnames(lm22_df)[1] <- "GeneSymbol"
lm22_df$GeneSymbol <- toupper(as.character(lm22_df$GeneSymbol))

lm22_long <- lm22_df %>%
  tidyr::pivot_longer(
    cols = -GeneSymbol,
    names_to = "lm22_cell_type",
    values_to = "weight"
  ) %>% dplyr::filter(!is.na(weight), weight != 0)

bulk_genes <- toupper(rownames(expr_mat))
overlap_genes <- intersect(bulk_genes, lm22_long$GeneSymbol)

expr_overlap_long <- expr_mat[match(overlap_genes, bulk_genes), , drop = FALSE] %>%
  as.data.frame() %>%
  tibble::rownames_to_column("GeneSymbol") %>%
  tidyr::pivot_longer(-GeneSymbol, names_to = "sample", values_to = "expr_value")

overlap_annotated <- expr_overlap_long %>%
  dplyr::mutate(GeneSymbol = toupper(GeneSymbol)) %>%
  dplyr::left_join(lm22_long, by = "GeneSymbol") %>%
  dplyr::select(GeneSymbol, lm22_cell_type, weight, sample, expr_value)

gene_cell_map <- lm22_long %>%
  dplyr::filter(GeneSymbol %in% overlap_genes) %>%
  dplyr::group_by(GeneSymbol) %>%
  dplyr::summarise(lm22_cell_types = paste(unique(lm22_cell_type), collapse = "; "), .groups = "drop")

gene_mean_expr <- expr_overlap_long %>%
  dplyr::group_by(GeneSymbol) %>%
  dplyr::summarise(mean_expr = mean(expr_value, na.rm = TRUE), .groups = "drop")

overlap_gene_summary <- gene_mean_expr %>%
  dplyr::left_join(gene_cell_map, by = "GeneSymbol") %>%
  dplyr::arrange(dplyr::desc(mean_expr))

readr::write_csv(overlap_annotated, file.path(out_dir, "LM22_overlap_gene_values_by_sample.csv"))
readr::write_csv(overlap_gene_summary, file.path(out_dir, "LM22_overlap_gene_summary.csv"))

pct_present <- length(unique(overlap_genes)) / length(unique(lm22_df$GeneSymbol))
writeLines(c(
  sprintf("LM22 overlapping genes: %d / %d (%.1f%%)",
          length(unique(overlap_genes)), length(unique(lm22_df$GeneSymbol)), 100 * pct_present),
  "Files written:",
  paste0(" - ", file.path(out_dir, "LM22_overlap_gene_values_by_sample.csv")),
  paste0(" - ", file.path(out_dir, "LM22_overlap_gene_summary.csv"))
), file.path(out_dir, "LM22_overlap_report.txt"))

message("Running CIBERSORT (LM22) ...")
results <- cibersort(sig_matrix = lm22_path,
                     mixture_file = mixture_out,
                     perm = perm_val,
                     QN = qn_flag)

results <- ensure_df(results)
res_csv <- file.path(out_dir, "CIBERSORT_results.csv")
write.csv(results, res_csv, row.names = TRUE)

non_cell_cols <- intersect(colnames(results),
                           c("P-value","Correlation","RMSE","Absolute score","AbsoluteScore","Absolute.score"))
cell_cols <- setdiff(colnames(results), non_cell_cols)

frac_long <- results[, cell_cols, drop = FALSE] %>%
  ensure_df() %>%
  tibble::rownames_to_column("sample") %>%
  tidyr::pivot_longer(-sample, names_to = "cell_type", values_to = "fraction")

plot_horizontal_stacked_bars <- function(frac_long, out_dir, chunk_size = 60) {
  frac_long <- frac_long %>%
    dplyr::mutate(sample = factor(sample, levels = sort(unique(sample))))
  sample_levels <- levels(frac_long$sample)
  n_total <- length(sample_levels)
  n_chunks <- ceiling(n_total / chunk_size)
  for (i in seq_len(n_chunks)) {
    idx_start <- (i - 1) * chunk_size + 1
    idx_end   <- min(i * chunk_size, n_total)
    keep_samp <- sample_levels[idx_start:idx_end]
    sub_df <- frac_long %>% dplyr::filter(sample %in% keep_samp)
    n_here <- length(keep_samp)
    w <- 12
    h <- max(10, n_here * 0.25)
    p <- ggplot(sub_df, aes(y = sample, x = fraction, fill = cell_type)) +
      geom_col(width = 0.8) +
      labs(title = if (n_chunks == 1)
             "CIBERSORT (LM22) - Relative Fractions per Sample"
           else
             paste0("CIBERSORT (LM22) - Fractions (Samples ", idx_start, "–", idx_end, " of ", n_total, ")"),
           x = "Fraction (≈ sum to 1)", y = NULL) +
      theme_bw(base_size = 11) +
      theme(
        axis.text.y = element_text(size = 8),
        axis.text.x = element_text(size = 9),
        legend.position = "right",
        panel.grid.major.y = element_blank()
      ) +
      coord_cartesian(xlim = c(0, 1), expand = FALSE)
    if (n_chunks == 1) {
      ggsave(file.path(out_dir, "01_stacked_bar_fractions.png"), p, width = w, height = h, dpi = 150)
    } else {
      fn_png <- sprintf("01_stacked_bar_fractions_page_%02d_of_%02d.png", i, n_chunks)
      ggsave(file.path(out_dir, fn_png), p, width = w, height = h, dpi = 150)
    }
  }
}
plot_horizontal_stacked_bars(frac_long, out_dir, chunk_size = chunk_size)

frac_mat <- as.matrix(results[, cell_cols, drop = FALSE])
.calc_heat_w <- function(k) max(9, k * 0.30)
.calc_heat_h <- function(n) max(9, n * 0.16)
heat_w <- .calc_heat_w(ncol(frac_mat))
heat_h <- .calc_heat_h(nrow(frac_mat))
show_row <- nrow(frac_mat) <= 80
show_col <- ncol(frac_mat) <= 60
pheatmap(frac_mat,
         main = "LM22 cell types (all) - samples × cell types",
         filename = file.path(out_dir, "02_heatmap_all_cell_types.png"),
         show_rownames = show_row, show_colnames = show_col,
         fontsize = ifelse(max(nrow(frac_mat), ncol(frac_mat)) > 60, 8, 10),
         cluster_rows = TRUE, cluster_cols = TRUE,
         border_color = NA,
         width = heat_w, height = heat_h)

if ("P-value" %in% colnames(results)) {
  p_hist <- ggplot(data.frame(P = results[, "P-value"]), aes(x = P)) +
    geom_histogram(bins = 20, color = "black") +
    theme_bw(base_size = 11) +
    labs(title = "CIBERSORT per-sample P-value distribution", x = "P-value", y = "Count")
  ggsave(file.path(out_dir, "03_pvalue_histogram.png"), p_hist, width = 6, height = 4.5, dpi = 150)
}

have_corr <- "Correlation" %in% colnames(results)
have_rmse <- "RMSE" %in% colnames(results)
if (have_corr && have_rmse) {
  df_cr <- results[, c("Correlation","RMSE"), drop = FALSE] %>%
    ensure_df() %>%
    tibble::rownames_to_column("sample")
  p_sc <- ggplot(df_cr, aes(x = Correlation, y = RMSE, label = sample)) +
    geom_point(size = 2, alpha = 0.8) +
    ggrepel::geom_text_repel(max.overlaps = 30, size = 3) +
    theme_bw(base_size = 11) +
    labs(title = "CIBERSORT fit metrics per sample", x = "Correlation", y = "RMSE") +
    scale_x_continuous(limits = c(min(df_cr$Correlation, na.rm = TRUE) - 0.05, 1))
  ggsave(file.path(out_dir, "04_scatter_correlation_vs_RMSE.png"), p_sc, width = 8, height = 5.5, dpi = 150)
}

summary_path <- file.path(out_dir, "RUN_SUMMARY.txt")
sink(summary_path)
cat("CIBERSORT RUN SUMMARY\n")
cat("=====================\n")
cat("Counts file:   ", counts_path, "\n")
cat("Metadata file: ", if (!is.na(meta_path) && file.exists(meta_path)) meta_path else "NOT PROVIDED", "\n")
cat("LM22 path:     ", lm22_path, "\n\n")
cat("Samples (input): ", ncol(count_mat), "\n")
cat("Genes (counts):  ", nrow(count_mat), "\n")
cat("Genes (kept):    ", nrow(expr_mat), "\n\n")
if ("P-value" %in% colnames(results)) {
  cat("Significant samples (P<=0.05): ", sum(results[, "P-value"] <= 0.05, na.rm = TRUE), "/", nrow(results), "\n")
}
if ("Correlation" %in% colnames(results)) {
  cat("Median Correlation: ", round(median(results[, "Correlation"], na.rm = TRUE), 3), "\n")
}
if ("RMSE" %in% colnames(results)) {
  cat("Median RMSE: ", round(median(results[, "RMSE"], na.rm = TRUE), 3), "\n")
}
cat("\nLM22 overlap report written to: ", file.path(out_dir, "LM22_overlap_report.txt"), "\n")
cat("\nMetadata used: ", if (meta_present) "YES" else "NO (skipped condition-based plots)", "\n")
cat("\nOutputs written to: ", out_dir, "\n")
cat("\nSession info:\n")
print(sessionInfo())
sink()

message("Done. All outputs in: ", out_dir)
'''

R_QA_STEP = r'''
library(dplyr)
library(readr)

out_dir <- Sys.getenv("PY_OUT_DIR")
res_path <- Sys.getenv("PY_RES_PATH")
if (!nzchar(res_path)) res_path <- file.path(out_dir, "CIBERSORT_results.csv")

res <- readr::read_csv(res_path, show_col_types = FALSE)
res <- as.data.frame(res)
rownames(res) <- res[[1]]
# Remove the first column (row names) - it may have auto-generated name like "...1"
res[[1]] <- NULL
res$sample <- rownames(res)
res <- res %>% dplyr::relocate(sample)

meta_cols <- intersect(colnames(res), c("P-value", "Correlation", "RMSE"))
cell_cols <- setdiff(colnames(res), c("sample", meta_cols))

to_numeric <- function(v) {
  if (is.numeric(v)) return(v)
  v <- as.character(v)
  v <- gsub(",", "", v)
  v <- gsub("%", "", v)
  v <- trimws(v)
  v <- gsub("\\s+", " ", v)
  v <- gsub("[^0-9eE+\\-\\.]", "", v)
  suppressWarnings(as.numeric(v))
}

if (length(cell_cols) > 0) {
  res[, cell_cols] <- lapply(res[, cell_cols, drop = FALSE], to_numeric)
}
if (length(meta_cols) > 0) {
  res[, meta_cols] <- lapply(res[, meta_cols, drop = FALSE], to_numeric)
}

non_numeric <- sapply(res[, cell_cols, drop = FALSE], function(x) !is.numeric(x))
if (any(non_numeric)) {
  stop(sprintf(
    "These columns are not numeric after coercion: %s",
    paste(names(which(non_numeric)), collapse = ", ")
  ))
}

res$sum_fractions <- rowSums(res[, cell_cols, drop = FALSE], na.rm = TRUE)
res$sum_ok <- res$sum_fractions >= 0.85 & res$sum_fractions <= 1.15

res$p_class <- cut(
  res$`P-value`,
  breaks = c(-Inf, 0.05, 0.1, Inf),
  labels = c("Good (<=0.05)", "Moderate (0.05-0.1)", "Poor (>0.1)")
)
res$corr_class <- cut(
  res$Correlation,
  breaks = c(-Inf, 0.2, 0.5, Inf),
  labels = c("Poor (<0.2)", "Moderate (0.2-0.5)", "Good (>=0.5)")
)
res$rmse_class <- cut(
  res$RMSE,
  breaks = c(-Inf, 0.7, 1.2, Inf),
  labels = c("Good (<=0.7)", "Moderate (0.7-1.2)", "Poor (>1.2)")
)

res$Quality_Category <- apply(res, 1, function(x) {
  p <- x["p_class"]; c <- x["corr_class"]; r <- x["rmse_class"]; sumok <- x["sum_ok"]
  if (is.na(p) | is.na(c) | is.na(r)) return("Incomplete data")
  if (p == "Good (<=0.05)" && c == "Good (>=0.5)" && r == "Good (<=0.7)" && sumok == "TRUE") {
    return("Excellent / Reliable")
  } else if (p %in% c("Good (<=0.05)", "Moderate (0.05-0.1)") &&
             c %in% c("Good (>=0.5)", "Moderate (0.2-0.5)") &&
             r %in% c("Good (<=0.7)", "Moderate (0.7-1.2)") &&
             sumok == "TRUE") {
    return("Moderate / Caution")
  } else {
    return("Poor / Unreliable")
  }
})

qc_out <- res %>%
  dplyr::select(sample, dplyr::all_of(cell_cols),
                dplyr::all_of(meta_cols),
                sum_fractions, sum_ok,
                p_class, corr_class, rmse_class,
                Quality_Category)

qc_file <- file.path(out_dir, "CIBERSORT_Quality_Assessment.csv")
write.csv(qc_out, qc_file, row.names = FALSE, fileEncoding = "UTF-8")

message("CIBERSORT Quality Assessment completed!")
message("Results saved to: ", qc_file)
print(table(res$Quality_Category, useNA = "ifany"))
'''

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _set_r_env(ro, env: dict[str, str]) -> None:
    """Set environment variables inside R (accessible via Sys.getenv in R)."""
    r_env = ro.r['Sys.setenv']
    r_env(**env)


def r_install_cran_packages(rpackages, StrVector, pkgs: list[str]) -> None:
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # cloud.r-project.org
    to_install = [p for p in pkgs if not rpackages.isinstalled(p)]
    if to_install:
        logging.info("Installing CRAN pkgs: %s", ", ".join(to_install))
        utils.install_packages(StrVector(to_install), repos="https://cloud.r-project.org")


def r_install_bioc_packages(rpackages, StrVector, pkgs: list[str]) -> None:
    if not rpackages.isinstalled('BiocManager'):
        r_install_cran_packages(rpackages, StrVector, ['BiocManager'])
    BiocManager = rpackages.importr('BiocManager')
    for p in pkgs:
        if not rpackages.isinstalled(p):
            logging.info("Installing Bioconductor pkg: %s", p)
            BiocManager.install(p, ask=False)


def maybe_install_base_dependencies(rpackages, StrVector, do_install: bool) -> None:
    """Minimal bootstrap so the main R code can run package checks."""
    if not do_install:
        return
    # lightweight pre-flight installs to avoid first-run pain
    r_install_cran_packages(rpackages, StrVector, ['devtools','readr','readxl','dplyr','tibble','stringr','tools',
                             'ggplot2','pheatmap','reshape2','tidyr','ggrepel','scales','data.table'])
    r_install_bioc_packages(rpackages, StrVector, ['edgeR', 'preprocessCore'])


def run_cibersort_pipeline(
    counts: str,
    out: str,
    lm22 : Path = LM22_PATH,
    meta: str | None = None,
    perm: int = 100,
    qn: bool = False,
    chunk_size: int = 60,
    install: bool = False,
    res_path: str | None = None,
) -> None:
    """
    Run CIBERSORT (LM22) pipeline + QA via rpy2.
    
    Args:
        counts: Path to counts/TPM table (tsv/csv/xlsx). First column = gene symbol.
        out: Output directory.
        lm22: Path to LM22 signature matrix (txt/tsv).
        meta: Optional metadata file with columns: sample_id, condition.
        out: Output directory.
        perm: CIBERSORT permutations (e.g., 100 or 1000). Default: 100.
        qn: Enable quantile normalization (QN). Default: False for RNA-seq/TPM.
        chunk_size: Samples per page for stacked bar plot. Default: 60.
        install: Pre-install base R packages (CRAN/Bioc) before running. Default: False.
        res_path: Optional explicit path to CIBERSORT_results.csv for QA step.
    
    Returns:
        None. Outputs are written to the specified output directory.
    """
    # CRITICAL: Lazy-initialize rpy2 HERE (in child process) to avoid Celery fork issues
    logging.info("Initializing rpy2/R for CIBERSORT pipeline...")
    ro, rpackages, StrVector = _lazy_init_rpy2()
    
    # Ensure absolute paths for stability
    counts = str(Path(counts).resolve())
    lm22 = str(Path(lm22).resolve())
    outdir = str(Path(out).resolve())
    meta = str(Path(meta).resolve()) if meta else ""

    os.makedirs(outdir, exist_ok=True)
    maybe_install_base_dependencies(rpackages, StrVector, install)

    # Set R env for the script
    _set_r_env(ro, {
        "PY_COUNTS_PATH": counts,
        "PY_META_PATH": meta,
        "PY_LM22_PATH": lm22,
        "PY_OUT_DIR": outdir,
        "PY_PERM": str(perm),
        "PY_QN": "true" if qn else "false",
        "PY_CHUNK_SIZE": str(chunk_size),
    })

    logging.info("Running main CIBERSORT R pipeline ...")
    
    try:
        # Capture R output/errors
        from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
        import sys
        
        # Run R pipeline with error handling
        result = ro.r(R_MAIN_PIPELINE)
        logging.info("✅ CIBERSORT R pipeline completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"❌ CIBERSORT R pipeline failed: {error_msg}")
        
        # Classify error types for better error messages
        if "missing values in 'row.names'" in error_msg or "row.names" in error_msg.lower():
            logging.error(
                "Data quality issue: Missing or empty gene IDs in counts file. "
                "This should have been caught by preprocessing. "
                "Please check counts file for empty/missing gene names."
            )
        elif "CIBERSORT" in error_msg or "library" in error_msg.lower():
            logging.error(
                "CIBERSORT package may not be installed. "
                "Please install manually: devtools::install_github('Moonerss/CIBERSORT')"
            )
        elif "install.packages" in error_msg:
            logging.error(
                "Package installation failed. Pre-install required R packages: "
                "edgeR, preprocessCore, data.table, readr, dplyr, ggplot2, etc."
            )
        
        raise RuntimeError(f"CIBERSORT pipeline execution failed: {error_msg}") from e

    # QA step
    _set_r_env(ro, {
        "PY_OUT_DIR": outdir,
        "PY_RES_PATH": str(Path(res_path).resolve()) if res_path else "",
    })
    logging.info("Running QA step ...")
    
    try:
        ro.r(R_QA_STEP)
        logging.info("✅ QA step completed successfully")
    except Exception as e:
        logging.error(f"❌ QA step failed: {e}")
        # QA is optional, don't fail the entire pipeline
        logging.warning("Continuing despite QA step failure...")

    logging.info("Done. Outputs in: %s", outdir)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CIBERSORT (LM22) pipeline + QA via rpy2."
    )
    p.add_argument("--counts", required=True, help="Counts/TPM table (tsv/csv/xlsx). First column = gene symbol.")
    p.add_argument("--lm22",    required=True, help="Path to LM22 signature matrix (txt/tsv).")
    p.add_argument("--out",     required=True, help="Output directory.")
    p.add_argument("--meta",    default=None, help="(Optional) metadata file with columns: sample_id, condition.")
    p.add_argument("--perm",    type=int, default=100, help="CIBERSORT permutations (e.g., 100 or 1000).")
    p.add_argument("--qn",      type=lambda s: s.lower() in {"1","true","t","yes","y"}, default=False,
                   help="Enable quantile normalization (QN). Defaults to False for RNA-seq/TPM.")
    p.add_argument("--chunk-size", type=int, default=60, help="Samples per page for stacked bar plot.")
    p.add_argument("--install", action="store_true",
                   help="Pre-install base R packages (CRAN/Bioc) before running.")
    p.add_argument("--res-path", default=None,
                   help="(Optional) explicit path to CIBERSORT_results.csv for QA step.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"], help="Python log level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point that parses arguments and calls run_pipeline."""
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # Fast sanity checks
    for pth, label in [(args.counts, "counts"), (args.lm22, "lm22")]:
        if not Path(pth).exists():
            logging.error("Missing %s file: %s", label, pth)
            return 2
    if args.meta and not Path(args.meta).exists():
        logging.warning("Metadata file not found, proceeding WITHOUT metadata: %s", args.meta)
        args.meta = None

    try:
        run_cibersort_pipeline(
            counts=args.counts,
            lm22=args.lm22,
            out=args.out,
            meta=args.meta,
            perm=args.perm,
            qn=args.qn,
            chunk_size=args.chunk_size,
            install=args.install,
            res_path=args.res_path,
        )
        return 0
    except Exception as e:
        logging.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    # CLI entry point - enables subprocess calls via python -m
    raise SystemExit(main())
