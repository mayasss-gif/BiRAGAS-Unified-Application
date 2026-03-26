#!/usr/bin/env python3
"""
Python wrapper around the provided R-based xCell pipeline using rpy2.

- Preserves the original logic and outputs.
- Optionally installs R package dependencies inside R (controlled by do_install parameter).

Requires:
  - Python
  - rpy2 (pip install rpy2)
  - A working R installation available on PATH
"""

import sys
from pathlib import Path

# Ensure rpy2 is available before proceeding
try:
    import rpy2.robjects as ro
    from rpy2.robjects import packages as rpackages
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects import conversion, default_converter
except Exception as e:
    sys.stderr.write(
        "\n[ERROR] rpy2 is required but not available. Install with: pip install rpy2\n"
    )
    raise

R = ro.r

# -------------------------
# R function definition
# -------------------------
R_PIPELINE = r'''
run_xcell_pipeline <- function(
  DATA_DIR  = "D:/AyassBio_Workspace_Downloads/xcell-deonv/Bulk-data-20251006T085217Z-1-001",
  EXPR_FILE = file.path(DATA_DIR, "GSE139061_Eadon_processed_QN_101419.csv"),
  META_FILE = file.path(DATA_DIR, "sample_metadata_patientAKI (1).tsv"),
  OUT_DIR   = file.path(DATA_DIR, "xcell_output_V29_BIO"),

  # install & tunables
  DO_INSTALL        = TRUE,
  THRESH_MED_HIGH   = 0.20,
  THRESH_MED_MOD    = 0.10,
  PRES_FDR_ALPHA    = 0.10,
  PRES_MIN_FRAC     = 0.35,
  UMAP_MIN_SAMPLES  = 20,
  MAX_CARDS         = Inf,
  TOP_N_CARD        = 25
) {

  # create out dirs
  if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, TRUE)
  DIR_MATRICES   <- file.path(OUT_DIR, "xCell_Scores");        dir.create(DIR_MATRICES,   TRUE)
 # DIR_SPILLOVER  <- file.path(OUT_DIR, "Spillover_Results");   dir.create(DIR_SPILLOVER,  TRUE)
  FIG_DIR        <- file.path(OUT_DIR, "cell_type_enrichment_profile_per_sample"); dir.create(FIG_DIR, TRUE)
  CARD_DIR       <- file.path(FIG_DIR, "Xcell_Analytical_Figures"); dir.create(CARD_DIR, TRUE)

  # ---- installs ----
  if (DO_INSTALL) {
    pkgs_cran <- c(
      "data.table","readr","stringr","edgeR","pheatmap","ggplot2","RColorBrewer","viridis",
      "dplyr","tidyr","tibble","knitr","kableExtra","rmarkdown","uwot","readxl",
      "cowplot","gridExtra"
    )
    to_get <- setdiff(pkgs_cran, rownames(installed.packages()))
    if (length(to_get)) install.packages(to_get, dependencies = TRUE)
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    BiocManager::install(c("xCell","AnnotationDbi","org.Hs.eg.db"), ask = FALSE, update = FALSE)
  }

  suppressPackageStartupMessages({
    library(data.table); library(readr); library(stringr)
    library(xCell); library(edgeR)
    library(AnnotationDbi); library(org.Hs.eg.db)
    library(pheatmap); library(ggplot2); library(RColorBrewer); library(viridis)
    library(dplyr); library(tidyr); library(tibble)
    library(knitr); library(kableExtra); library(rmarkdown)
    library(uwot); library(readxl)
    library(cowplot); library(gridExtra)
  })
  theme_set(ggplot2::theme_bw(base_size = 12))

  # Persist R session info for reproducibility
  writeLines(c(capture.output(sessionInfo())), file.path(OUT_DIR, "sessionInfo.txt"))

  # helper readers / utils --------------------------------------------------
  read_any_table <- function(path, guess_max = 1e6) {
    stopifnot(file.exists(path))
    ext <- tolower(tools::file_ext(path))
    if (ext %in% c("csv")) {
      return(as.data.table(readr::read_csv(path, guess_max = guess_max, show_col_types = FALSE)))
    } else if (ext %in% c("tsv")) {
      return(as.data.table(readr::read_tsv(path, guess_max = guess_max, show_col_types = FALSE)))
    } else if (ext %in% c("txt")) {
      dt <- tryCatch(as.data.table(readr::read_delim(path, delim = NULL, guess_max = guess_max, show_col_types = FALSE)),
                     error = function(e) data.table::fread(path))
      return(dt)
    } else if (ext %in% c("xlsx","xls")) {
      df <- readxl::read_excel(path)
      return(as.data.table(df))
    } else {
      stop(sprintf("Unsupported file type '%s'. Allowed: csv, tsv, txt, xlsx, xls.", ext))
    }
  }
  standardize_id <- function(x){
    x <- as.character(x); x <- trimws(x)
    x <- gsub("\\.(csv|tsv|txt|xlsx|xls)$", "", x, ignore.case = TRUE)
    x <- gsub("[^A-Za-z0-9]+","_", x); x <- gsub("_+","_", x); x <- gsub("^_+|_+$","", x)
    x <- tolower(x); x <- sub("^x(?=\\d)","", x, perl=TRUE)
    x
  }
  align_mat_to_samples <- function(mat, sample_ids, debug_dir = OUT_DIR) {
    stopifnot(!is.null(colnames(mat)))
    present <- intersect(sample_ids, colnames(mat))
    missing <- setdiff(sample_ids, colnames(mat))
    if (length(present) == 0) {
      utils::write.csv(data.frame(mat_cols = colnames(mat)),
                       file.path(debug_dir, "_debug_plotmat_cols.csv"), row.names = FALSE)
      utils::write.csv(data.frame(requested = sample_ids),
                       file.path(debug_dir, "_debug_requested_samples.csv"), row.names = FALSE)
      stop("No overlapping sample IDs between meta and plot_mat. See _debug_* CSVs in OUT_DIR.")
    }
    if (length(missing)) {
      message("Note: dropping ", length(missing), " meta samples absent in matrix: ",
              paste(head(missing, 10), collapse = ", "),
              if (length(missing) > 10) " ...")
    }
    mat[, present, drop = FALSE]
  }
  num_mat <- function(x) { x <- as.matrix(x); storage.mode(x) <- "numeric"; x }

  # 2) Metadata --------------------------------------------------------------
  meta_raw <- read_any_table(META_FILE)
  stopifnot(nrow(meta_raw)>0)
  names(meta_raw) <- tolower(gsub("[^A-Za-z0-9]+","_", names(meta_raw)))
  
  # Flexible sample column detection (trust upstream validation)
  sample_col <- if("sample_id" %in% names(meta_raw)) {
    "sample_id"
  } else if("sample" %in% names(meta_raw)) {
    "sample"
  } else if("sampleid" %in% names(meta_raw)) {
    "sampleid"
  } else if("sample_name" %in% names(meta_raw)) {
    "sample_name"
  } else {
    # Fallback: use first column
    message("No standard sample column found. Using first column: ", names(meta_raw)[1])
    names(meta_raw)[1]
  }
  
  # Flexible condition column detection
  condition_col <- if("condition" %in% names(meta_raw)) {
    "condition"
  } else if("group" %in% names(meta_raw)) {
    "group"
  } else if("treatment" %in% names(meta_raw)) {
    "treatment"
  } else if("class" %in% names(meta_raw)) {
    "class"
  } else {
    # Fallback: use second column
    if(length(names(meta_raw)) > 1) {
      message("No standard condition column found. Using second column: ", names(meta_raw)[2])
      names(meta_raw)[2]
    } else {
      stop("Metadata must have at least 2 columns (sample + condition)")
    }
  }
  
  meta_dt <- meta_raw[, .(sample_id=as.character(get(sample_col)), condition=as.character(get(condition_col)))]
  meta_dt[, sample_std := standardize_id(sample_id)]

  # 3) Expression (robust orientation) --------------------------------------
  expr_dt_raw <- read_any_table(EXPR_FILE); stopifnot(nrow(expr_dt_raw)>0)

  # A) genes in rows
  expr_A <- data.table::copy(expr_dt_raw); data.table::setnames(expr_A, names(expr_A)[1], "Gene")
  expr_A <- expr_A[!is.na(Gene) & Gene != ""]
  mat_A  <- as.data.frame(expr_A, stringsAsFactors=FALSE); rownames(mat_A) <- mat_A$Gene; mat_A$Gene <- NULL
  mat_A[] <- lapply(mat_A, function(x) suppressWarnings(as.numeric(x))); mat_A <- as.matrix(mat_A)
  cols_A_std <- standardize_id(colnames(mat_A))
  overlap_A  <- length(intersect(cols_A_std, meta_dt$sample_std))

  # B) samples in rows (transpose)
  expr_B <- data.table::copy(expr_dt_raw); data.table::setnames(expr_B, names(expr_B)[1], "sample_id")
  expr_B[, sample_std := standardize_id(sample_id)]; expr_B <- expr_B[!duplicated(sample_std)]
  rownames_B <- expr_B$sample_id; expr_B$sample_id <- expr_B$sample_std <- NULL
  mat_B <- as.data.frame(expr_B, stringsAsFactors=FALSE); rownames(mat_B) <- rownames_B
  mat_B[] <- lapply(mat_B, function(x) suppressWarnings(as.numeric(x))); mat_B <- base::t(as.matrix(mat_B))
  cols_B_std <- standardize_id(colnames(mat_B))
  overlap_B  <- length(intersect(cols_B_std, meta_dt$sample_std))

  expr <- if (overlap_B > overlap_A) { colnames(mat_B) <- cols_B_std; mat_B } else { colnames(mat_A) <- cols_A_std; mat_A }
  keep <- intersect(colnames(expr), meta_dt$sample_std); stopifnot(length(keep)>0)
  expr <- expr[, keep, drop=FALSE]; meta_dt <- meta_dt[match(keep, meta_dt$sample_std)]

  # 4) Ensembl→HGNC + collapse ---------------------------------------------
  ensembl_like <- if (length(rownames(expr))) mean(grepl("^ENSG[0-9]", rownames(expr))) > 0.5 else FALSE
  if (ensembl_like) {
    syms <- AnnotationDbi::mapIds(org.Hs.eg.db, keys=rownames(expr), keytype="ENSEMBL", column="SYMBOL", multiVals="first")
    keepg <- !is.na(syms); expr <- expr[keepg,,drop=FALSE]; rownames(expr) <- syms[keepg]
  }
  if (any(duplicated(rownames(expr)))) expr <- rowsum(expr, group=rownames(expr))

  # 5) Normalization heuristic ----------------------------------------------
  vals <- as.numeric(expr[is.finite(expr)]); med_val <- median(vals, na.rm=TRUE)
  int_frac <- mean(abs(vals - round(vals)) < 1e-8); max_val <- max(vals, na.rm=TRUE)
  if (int_frac > 0.95 && max_val > 100) {
    dge <- edgeR::DGEList(counts = round(expr))
    dge <- edgeR::calcNormFactors(dge, method="TMM")
    expr_norm <- edgeR::cpm(dge, log=TRUE, prior.count=1)
    normalization_method <- "edgeR_TMM_logCPM"
  } else if (med_val > 30 && max_val > 1000) {
    expr_norm <- log2(expr + 1); normalization_method <- "log2_safety"
  } else { expr_norm <- expr; normalization_method <- "as_is" }
  write.csv(expr_norm, file.path(OUT_DIR, sprintf("normalized_expression_%s.csv", normalization_method)))

  # 6) xCell + spillover ----------------------------------------------------
  data("xCell.data", package="xCell")
  spill_obj <- if ("spill.array" %in% names(xCell.data)) xCell.data$spill.array else xCell.data$spill

  # if (!is.null(spill_obj$K))  utils::write.csv(spill_obj$K,  file.path(DIR_SPILLOVER, "spillover_K.csv"), row.names = TRUE)
  # if (!is.null(spill_obj$fv)) utils::write.csv(spill_obj$fv, file.path(DIR_SPILLOVER, "spillover_fit_values.csv"), row.names = TRUE)

  raw_scores         <- xCell::rawEnrichmentAnalysis(expr = expr_norm, signatures = xCell.data$signatures, genes = xCell.data$genes)
  transformed_scores <- xCell::transformScores(scores = raw_scores, fit.vals = spill_obj$fv, scale = TRUE)
  xcell_scores       <- xCell::spillOver(transformedScores = transformed_scores, K = spill_obj$K, alpha = 0.5)

  write.csv(raw_scores,         file.path(DIR_MATRICES, "xcell_raw_scores.csv"))
  write.csv(transformed_scores, file.path(DIR_MATRICES, "xcell_transformed_scores.csv"))
  write.csv(xcell_scores,       file.path(DIR_MATRICES, "xcell_final_scores.csv"))

  PLOT_LAYER     <- "adjusted";   ANALYSIS_LAYER <- "adjusted"
  layers         <- list(raw = raw_scores, transformed = transformed_scores, adjusted = xcell_scores)
  plot_mat       <- layers[[PLOT_LAYER]]
  analysis_mat   <- layers[[ANALYSIS_LAYER]]
  layer_tag      <- paste0(" [", PLOT_LAYER, "]")

  # 6.1) Restrict to EXACTLY the 64 requested xCell cell types --------------
  ALLOWED_CT <- c(
    "Adipocytes","Astrocytes","B-cells","Basophils","CD4+ T-cells","CD4+ Tcm","CD4+ Tem",
    "CD4+ memory T-cells","CD4+ naive T-cells","CD8+ T-cells","CD8+ Tcm","CD8+ Tem",
    "CD8+ naive T-cells","CLP","CMP","Chondrocytes","Class-switched memory B-cells","DC",
    "Endothelial cells","Eosinophils","Epithelial cells","Erythrocytes","Fibroblasts","GMP",
    "HSC","Hepatocytes","Keratinocytes","MEP","MPP","MSC","Macrophages","Macrophages M1",
    "Macrophages M2","Mast cells","Megakaryocytes","Melanocytes","Memory B-cells",
    "Mesangial cells","Monocytes","Myocytes","NK cells","NKT","Neurons","Neutrophils",
    "Osteoblast","Pericytes","Plasma cells","Platelets","Preadipocytes","Sebocytes",
    "Skeletal muscle","Smooth muscle","Tgd cells","Th1 cells","Th2 cells","Tregs","aDC",
    "cDC","iDC","ly Endothelial cells","mv Endothelial cells","naive B-cells","pDC","pro B-cells"
  )

  layers <- lapply(layers, function(M) {
    if (is.null(M)) return(M)
    keep <- intersect(ALLOWED_CT, rownames(M))
    M[keep, , drop = FALSE]
  })
  plot_mat     <- layers[[PLOT_LAYER]]
  analysis_mat <- layers[[ANALYSIS_LAYER]]

  # 7) Significance (BetaDist; guarded) -------------------------------------
  which_sig <- "none"
  xcell_p <- tryCatch({ which_sig <- "xCellSignificanceBetaDist"; xCell::xCellSignificanceBetaDist(analysis_mat) }, error=function(e) NULL)
  if (is.null(xcell_p)) xcell_p <- tryCatch({ which_sig <- "xCellSignifcanceBetaDist"; xCell::xCellSignifcanceBetaDist(analysis_mat) }, error=function(e) NULL)
  message("Significance function used: ", which_sig)

  sig_counts_tbl <- NULL; fdr_mat <- NULL
  if (!is.null(xcell_p)) {
    p_mat  <- as.matrix(xcell_p)
    fdr_mat <- t(apply(p_mat, 1, function(p) p.adjust(p, "fdr")))
    rownames(fdr_mat) <- rownames(p_mat); colnames(fdr_mat) <- colnames(p_mat)
    #write.csv(xcell_p, file.path(DIR_MATRICES, "xcell_signif_pvals.csv"))
    #write.csv(fdr_mat,  file.path(DIR_MATRICES, "xcell_signif_fdr.csv"))
    sig_counts_tbl <- sort(rowSums(fdr_mat <= PRES_FDR_ALPHA, na.rm = TRUE), decreasing = TRUE)
    sig_counts_tbl <- data.frame(CellType = names(sig_counts_tbl), SigSamples = as.numeric(sig_counts_tbl))
    write.csv(sig_counts_tbl, file.path(DIR_MATRICES, "High_Variability_CellTypes.csv"), row.names = FALSE)
  }

  # 8) Heatmap helper --------------------------------------------------------
  hm_cols  <- colorRampPalette(brewer.pal(11, "RdYlBu"))(255)
  safe_heatmap <- function(mat, file, ann_col=NULL, ann_row=NULL, main=""){
    if (is.null(mat) || !all(is.finite(colMeans(mat, na.rm=TRUE)))) return(invisible(FALSE))
    pheatmap::pheatmap(
      mat,
      color = hm_cols, border_color = NA,
      annotation_col = ann_col, annotation_row = ann_row,
      clustering_method = "complete",
      show_rownames = nrow(mat) <= 80,
      show_colnames = TRUE,
      fontsize_col = ifelse(ncol(mat) > 80, 6, 9),
      main = main, filename = file, width = 12, height = 12
    )
    invisible(TRUE)
  }
  meta_simple <- data.frame(condition = factor(meta_dt$condition),
                            row.names = meta_dt$sample_std, check.names = FALSE)

  # 9) Core Figures ----------------------------------------------------------
  sample_order <- meta_dt$sample_std
  X_all <- align_mat_to_samples(plot_mat, sample_order) |> num_mat()
  safe_heatmap(
    X_all,
    file.path(FIG_DIR, "TME_Enrichment_Heatmap_AllCellTypes.png"),
    ann_col = meta_simple,
    main = paste0("Tumor Microenvironment Enrichment — All Cell Types", layer_tag)
  )

  var_sd <- apply(plot_mat, 1, sd, na.rm = TRUE)
  top_cts <- names(sort(var_sd, decreasing = TRUE))[seq_len(min(30, length(var_sd)))]
  top_var_tbl <- data.frame(CellType = top_cts, SD = round(var_sd[top_cts], 4), check.names = FALSE)
  #write.csv(top_var_tbl, file.path(DIR_MATRICES, "tbl_top_variable_ct.csv"), row.names = FALSE)
  X_top <- plot_mat[top_cts,, drop=FALSE]
  X_top <- align_mat_to_samples(X_top, sample_order) |> num_mat()
  safe_heatmap(
    X_top,
    file.path(FIG_DIR, "TopVariability_CellTypes_Heatmap.png"),
    ann_col = meta_simple,
    main = paste0("Top-Variability Cell Types (by SD)", layer_tag)
  )

  plot_mat_aligned <- align_mat_to_samples(plot_mat, sample_order) |> num_mat()
  pc <- prcomp(t(plot_mat_aligned), scale.=TRUE)
  pc_df <- data.frame(
    PC1=pc$x[,1], PC2=pc$x[,2],
    condition = meta_dt$condition[match(colnames(plot_mat_aligned), meta_dt$sample_std)]
  )
  ggplot(pc_df, aes(PC1, PC2, color=condition, shape=condition)) +
    geom_point(size=3, alpha=.9) +
    scale_color_viridis_d() +
    labs(title=paste0("xCell Score PCA by Condition", layer_tag), x="PC1", y="PC2") +
    theme(legend.position="right") -> p_pca
  ggsave(file.path(FIG_DIR, "xCell_Score_PCA_by_Condition.png"), p_pca, width=9, height=7, dpi=240)

  M <- suppressWarnings(cor(t(plot_mat_aligned), use="pairwise.complete.obs", method="spearman"))
  safe_heatmap(
    M,
    file.path(FIG_DIR, "CellType_Correlation_Spearman.png"),
    main = paste0("Cell-Type Correlation Matrix (Spearman)", layer_tag)
  )

  K <- min(12, length(top_cts))
  dir.create(file.path(FIG_DIR, "Boxplots_TopK"), showWarnings=FALSE)
  for (ct in top_cts[seq_len(K)]) {
    df <- data.frame(
      score=as.numeric(plot_mat_aligned[ct, ]),
      sample=colnames(plot_mat_aligned),
      condition = meta_dt$condition[match(colnames(plot_mat_aligned), meta_dt$sample_std)]
    )
    ggplot(df, aes(condition, score, fill=condition)) +
      geom_boxplot(outlier.shape=NA, alpha=.9) +
      geom_jitter(width=.2, alpha=.6) +
      scale_fill_viridis_d() +
      labs(title=paste0("xCell Enrichment — ", ct, layer_tag), y="Enrichment Score", x=NULL) +
      theme(legend.position="none") -> p_bx
    ggsave(file.path(FIG_DIR, "Boxplots_TopK", paste0(gsub("[/\\\\]", "_", ct), ".png")),
           p_bx, width=7, height=5, dpi=240)
  }

  # 11) UMAP (guarded)
  if (nrow(meta_dt) >= UMAP_MIN_SAMPLES) {
    set.seed(7)
    X_umap <- scale(t(plot_mat_aligned))
    X_umap[!is.finite(X_umap)] <- 0
    U <- uwot::umap(X_umap, n_neighbors = 15, min_dist = 0.2)
    umap_df <- data.frame(
      UMAP1=U[,1], UMAP2=U[,2],
      condition = meta_dt$condition[match(colnames(plot_mat_aligned), meta_dt$sample_std)]
    )
    ggplot(umap_df, aes(UMAP1, UMAP2, color=condition, shape=condition)) +
      geom_point(size=2.8, alpha=.9) + scale_color_viridis_d() +
      labs(title=paste0("UMAP of xCell Enrichment Space", layer_tag)) -> p_umap
    ggsave(file.path(FIG_DIR, "UMAP_xCell_EnrichmentSpace.png"), p_umap, width=8, height=6, dpi=240)
  }

  # 10) Composite indices + stacked composition -----------------------------
  IMMUNE_SET <- c(
    "B-cells","Class-switched memory B-cells","Memory B-cells","naive B-cells","pro B-cells","Plasma cells",
    "CD4+ T-cells","CD4+ memory T-cells","CD4+ naive T-cells","CD4+ Tcm","CD4+ Tem",
    "CD8+ T-cells","CD8+ naive T-cells","CD8+ Tcm","CD8+ Tem",
    "NK cells","NKT","Tregs","Th1 cells","Th2 cells","Tgd cells",
    "Monocytes","Macrophages","Macrophages M1","Macrophages M2","Mast cells","Neutrophils","Eosinophils",
    "DC","aDC","cDC","iDC","pDC","CLP","GMP","HSC","MPP"
  )
  STROMA_SET <- c(
    "Fibroblasts","MSC","Endothelial cells","ly Endothelial cells","mv Endothelial cells","Pericytes",
    "Epithelial cells","Keratinocytes","Chondrocytes","Myocytes","Smooth muscle","Skeletal muscle",
    "Osteoblast","Adipocytes","Preadipocytes","Megakaryocytes","Platelets","Hepatocytes",
    "Neurons","Astrocytes","Melanocytes","Mesangial cells","Erythrocytes","Sebocytes"
  )

  rowmean_safe <- function(M, rows) {
    rs <- intersect(rows, rownames(M))
    if (!length(rs)) return(rep(NA_real_, ncol(M)))
    colMeans(M[rs, , drop = FALSE], na.rm = TRUE)
  }

  plot_mat_aligned <- align_mat_to_samples(plot_mat, meta_dt$sample_std) |> num_mat()

  get_index_from_layers <- function(name, sample_ids) {
    for (lay in c("adjusted","transformed","raw")) {
      M <- layers[[lay]]
      if (!is.null(M) && name %in% rownames(M)) return(as.numeric(M[name, sample_ids]))
    }
    rep(NA_real_, length(sample_ids))
  }

  plot_df <- data.frame(
    sample_std = colnames(plot_mat_aligned),
    condition  = meta_dt$condition[match(colnames(plot_mat_aligned), meta_dt$sample_std)],
    t(plot_mat_aligned),
    check.names = FALSE
  )
  rownames(plot_df) <- plot_df$sample_std

  Imm_raw <- get_index_from_layers("ImmuneScore",           plot_df$sample_std)
  Str_raw <- get_index_from_layers("StromaScore",           plot_df$sample_std)
  Mic_raw <- get_index_from_layers("MicroenvironmentScore", plot_df$sample_std)

  if (all(is.na(Imm_raw))) Imm_raw <- rowmean_safe(plot_mat_aligned, IMMUNE_SET)
  if (all(is.na(Str_raw))) Str_raw <- rowmean_safe(plot_mat_aligned, STROMA_SET)
  if (all(is.na(Mic_raw))) Mic_raw <- (Imm_raw + Str_raw) / 2

  plot_df$ImmuneScore_xCell           <- Imm_raw
  plot_df$StromaScore_xCell           <- Str_raw
  plot_df$MicroenvironmentScore_xCell <- Mic_raw

  plot_df$TME_Inflammation_Index <-
    rowmean_safe(plot_mat_aligned, c("CD8+ T-cells","CD8+ Tem","CD8+ Tcm","NK cells","NKT")) -
    rowmean_safe(plot_mat_aligned, c("Tregs","Macrophages M2","Macrophages","Fibroblasts","Monocytes"))

  idx_long <- plot_df |>
    dplyr::select(condition,
                  ImmuneScore_xCell, StromaScore_xCell, MicroenvironmentScore_xCell,
                  TME_Inflammation_Index) |>
    tidyr::pivot_longer(-condition, names_to = "Index", values_to = "Value")

  index_labels <- c(
    ImmuneScore_xCell            = "ImmuneScore (xCell/proxy)",
    StromaScore_xCell            = "StromaScore (xCell/proxy)",
    MicroenvironmentScore_xCell  = "MicroenvironmentScore (xCell/proxy)",
    TME_Inflammation_Index       = "Hot Tumor — Inflamed / Immunologically Active TME\n(lower values → Cold Tumor — Immune-Excluded/Desert)"
  )

  p_idx <- ggplot(idx_long, aes(condition, Value, fill = condition)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = .2, alpha = .6) +
    scale_fill_viridis_d() +
    facet_wrap(~Index, scales = "free_y", labeller = as_labeller(index_labels)) +
    labs(title = paste0("Composite Indices by Condition", layer_tag),
         x = NULL, y = "Index Value")

  tme_vals <- dplyr::filter(idx_long, Index == "TME_Inflammation_Index")$Value
  yr <- range(tme_vals, na.rm = TRUE)
  ypad <- diff(yr) * 0.05
  x_guide <- 1.95

  ann_seg  <- data.frame(Index = "TME_Inflammation_Index",
                         x = x_guide, xend = x_guide,
                         y = yr[1] + ypad, yend = yr[2] - ypad)

  ann_hot  <- data.frame(Index = "TME_Inflammation_Index",
                         x = x_guide, y = yr[2] - ypad,
                         lab = "Higher = 'Hot'  (inflamed / immune-active)")

  ann_cold <- data.frame(Index = "TME_Inflammation_Index",
                         x = x_guide, y = yr[1] + ypad,
                         lab = "Lower = 'Cold'  (immune-silent / desert)")

  p_idx <- p_idx +
    scale_x_discrete(expand = ggplot2::expansion(mult = c(0.05, 0.30))) +
    coord_cartesian(clip = "off") +
    theme(plot.margin = grid::unit(c(5.5, 30, 5.5, 5.5), "pt")) +
    geom_segment(data = ann_seg,
                 aes(x = x, xend = xend, y = y, yend = yend),
                 inherit.aes = FALSE, colour = "grey30",
                 arrow = grid::arrow(length = grid::unit(0.18, "cm"), ends = "both")) +
    geom_text(data = ann_hot, aes(x = x, y = y, label = lab),
              inherit.aes = FALSE, hjust = 0, vjust = 0, size = 3.2, colour = "#B22222") +
    geom_text(data = ann_cold, aes(x = x, y = y, label = lab),
              inherit.aes = FALSE, hjust = 0, vjust = 1, size = 3.2, colour = "#1F4E79")

  ggsave(file.path(FIG_DIR, "Composite_Indices_by_Condition.png"), p_idx, width = 12, height = 8, dpi = 300)

  comp_long <- plot_df |>
    dplyr::select(sample_std, condition, tidyselect::all_of(rownames(plot_mat))) |>
    tidyr::pivot_longer(-c(sample_std, condition), names_to="CellType", values_to="Score") |>
    dplyr::group_by(sample_std) |>
    dplyr::mutate(Frac = Score / sum(Score, na.rm=TRUE)) |>
    dplyr::ungroup()

  p_stack <- ggplot(comp_long, aes(sample_std, Frac, fill=CellType)) +
    geom_bar(stat="identity") +
    facet_grid(~condition, scales="free_x", space="free_x") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    scale_fill_viridis_d(guide=guide_legend(title="Cell Type")) +
    labs(title=paste0("Stacked Composition of Enrichment (All Cell Types)", layer_tag),
         x="Samples", y="Fraction of Total Enrichment")
  ggsave(file.path(FIG_DIR, "Stacked_Composition_AllCellTypes.png"), p_stack, width=14, height=6, dpi=300)

  # 10.x) Ensure per-sample frame is complete + export summary ---------------
  SAMPLE_IDS <- colnames(plot_mat_aligned)
  missing_rows <- setdiff(SAMPLE_IDS, plot_df$sample_std)
  if (length(missing_rows)) {
    add_df <- data.frame(
      sample_std = missing_rows,
      condition  = meta_dt$condition[match(missing_rows, meta_dt$sample_std)],
      ImmuneScore_xCell = NA_real_,
      StromaScore_xCell = NA_real_,
      MicroenvironmentScore_xCell = NA_real_,
      TME_Inflammation_Index = NA_real_,
      check.names = FALSE
    )
    plot_df <- rbind(plot_df, add_df)
  }
  plot_df <- plot_df[match(SAMPLE_IDS, plot_df$sample_std), , drop = FALSE]

  per_sample_summary <- plot_df[, c("sample_std","condition",
                                    "ImmuneScore_xCell","StromaScore_xCell",
                                    "MicroenvironmentScore_xCell","TME_Inflammation_Index")]
  write.csv(per_sample_summary,
            file.path(DIR_MATRICES, "Cohort_TME_Matrix.csv"), row.names = FALSE)

  # 12) Per-sample cards -----------------------------------------------------
  label_map <- c(
    ImmuneScore_xCell           = "ImmuneScore (xCell/proxy)",
    StromaScore_xCell           = "StromaScore (xCell/proxy)",
    MicroenvironmentScore_xCell = "MicroenvironmentScore (xCell/proxy)",
    TME_Inflammation_Index      = "Hot Tumor — Inflamed / Immunologically Active TME",
    condition                   = "Condition"
  )

  card_samples <- SAMPLE_IDS
  if (is.finite(MAX_CARDS)) card_samples <- head(card_samples, MAX_CARDS)

  skipped <- character(0)
  for (s in card_samples) {
    vec <- plot_mat_aligned[, s]
    if (all(!is.finite(vec))) { skipped <- c(skipped, s); next }

    row_i <- plot_df[plot_df$sample_std == s, , drop = FALSE]
    if (nrow(row_i) == 0) { skipped <- c(skipped, s); next }

    ord   <- order(vec, names(vec), decreasing = TRUE, na.last = NA)
    topn  <- head(names(vec)[ord], TOP_N_CARD)
    dfbar <- data.frame(CellType = factor(topn, levels = rev(topn)),
                        Score    = as.numeric(vec[topn]))

    present_cols <- intersect(names(label_map), names(row_i))
    idx_tbl <- data.frame(
      Metric = unname(label_map[present_cols]),
      Value  = vapply(present_cols, function(k) {
        v <- row_i[[k]]
        if (is.numeric(v)) sprintf("%.3f", as.numeric(v)) else as.character(v)
      }, FUN.VALUE = character(1)),
      check.names = FALSE
    )

    p_bar <- ggplot(dfbar, aes(Score, CellType, fill = Score)) +
      geom_col() +
      scale_fill_viridis_c() +
      labs(title = paste0("xCell Enrichment — ", s, layer_tag),
           x = "Enrichment Score", y = NULL) +
      theme(legend.position = "none")

    p_tbl <- gridExtra::tableGrob(
      idx_tbl, rows = NULL,
      theme = gridExtra::ttheme_minimal(base_size = 10, padding = grid::unit(c(6,6), "mm"))
    )

    card <- cowplot::plot_grid(p_bar, p_tbl, ncol = 1, rel_heights = c(3, 1))
    ggsave(file.path(CARD_DIR, paste0(gsub("[^A-Za-z0-9]+","_", s), "_Card.png")),
           card, width = 8.5, height = 7.5, dpi = 300)
  }

  if (length(skipped)) writeLines(skipped, con = file.path(CARD_DIR, "_skipped_samples.txt"))

  # 13) BIO CONTEXT SUMMARY --------------------------------------------------
  xcell_bio_summary <- list(
    "aDC"="Activated dendritic cells; antigen-presenting cells that drive T-cell priming and cytokine release.",
    "Adipocytes"="Fat-storing cells that secrete adipokines and influence inflammation and metabolism.",
    "Astrocytes"="Glial cells maintaining neural homeostasis and contributing to neuroinflammatory signaling.",
    "B-cells"="Adaptive immune cells producing antibodies; key in humoral immune response.",
    "Basophils"="Granulocytes mediating allergic reactions and releasing histamine and IL-4.",
    "CD4+ memory T-cells"="Memory helper T-cells; provide rapid recall response upon antigen re-exposure.",
    "CD4+ naive T-cells"="Unactivated helper T-cells; precursors that differentiate into effector subsets.",
    "CD4+ T-cells"="Helper T-cells coordinating immune activation and cytokine secretion.",
    "CD4+ Tcm"="Central memory CD4+ T-cells; reside in lymphoid tissue and provide long-term immunity.",
    "CD4+ Tem"="Effector memory CD4+ T-cells; circulate in peripheral tissues and respond quickly to antigens.",
    "CD8+ naive T-cells"="Unactivated cytotoxic T-cells; precursors of effector CD8+ populations.",
    "CD8+ T-cells"="Cytotoxic T lymphocytes; kill infected or malignant cells via perforin and granzyme release.",
    "CD8+ Tcm"="Central memory CD8+ T-cells; maintain long-term cytotoxic memory.",
    "CD8+ Tem"="Effector memory CD8+ T-cells; provide rapid response at tissue sites.",
    "cDC"="Conventional dendritic cells; professional antigen-presenting cells activating T-cells.",
    "Chondrocytes"="Cartilage-producing cells involved in extracellular matrix and joint maintenance.",
    "Class-switched memory B-cells"="Mature B-cells expressing IgG/IgA/IgE after class switch recombination.",
    "CLP"="Common lymphoid progenitors; give rise to B, T, and NK lineages.",
    "CMP"="Common myeloid progenitors; precursors for granulocytes, erythrocytes, and macrophages.",
    "DC"="General dendritic cell population mediating antigen presentation and T-cell activation.",
    "Endothelial cells"="Vascular lining cells regulating permeability, angiogenesis, and leukocyte trafficking.",
    "Eosinophils"="Granulocytes defending against parasites and driving Th2-type inflammation.",
    "Epithelial cells"="Barrier cells covering organs; key in regeneration and mucosal immunity.",
    "Erythrocytes"="Red blood cells transporting oxygen; presence indicates blood contamination in samples.",
    "Fibroblasts"="Stromal cells synthesizing collagen and remodeling extracellular matrix.",
    "GMP"="Granulocyte-monocyte progenitors; intermediates for myeloid differentiation.",
    "Hepatocytes"="Liver parenchymal cells central to metabolism, detoxification, and protein synthesis.",
    "HSC"="Hematopoietic stem cells; multipotent progenitors of all blood lineages.",
    "iDC"="Immature dendritic cells; antigen-sampling cells before activation.",
    "Keratinocytes"="Epidermal cells producing keratin and cytokines; maintain skin barrier.",
    "ly Endothelial cells"="Lymphatic endothelial cells mediating drainage and immune cell trafficking.",
    "Macrophages"="Phagocytes orchestrating tissue homeostasis, repair, and immune activation.",
    "Macrophages M1"="Classically activated macrophages; pro-inflammatory, anti-tumor phenotype.",
    "Macrophages M2"="Alternatively activated macrophages; tissue-repairing, immunosuppressive phenotype.",
    "Mast cells"="Granule-rich cells mediating allergic responses and vascular permeability.",
    "Megakaryocytes"="Large bone-marrow cells producing platelets and cytokines.",
    "Melanocytes"="Pigment-producing cells; regulate oxidative stress and immune interactions.",
    "Memory B-cells"="Long-lived B-cells ensuring rapid antibody production upon re-challenge.",
    "MEP"="Megakaryocyte-erythroid progenitors; precursors of platelets and red cells.",
    "Mesangial cells"="Kidney glomerular support cells regulating filtration and vascular tone.",
    "Monocytes"="Circulating precursors of macrophages and dendritic cells.",
    "MPP"="Multipotent progenitors; intermediate stem cells generating multiple blood lineages.",
    "MSC"="Mesenchymal stem cells; multipotent stromal cells modulating immunity and repair.",
    "mv Endothelial cells"="Microvascular endothelial cells; form small-vessel networks and mediate exchange.",
    "Myocytes"="Muscle cells responsible for contractile function in skeletal or cardiac tissue.",
    "naive B-cells"="Unactivated B lymphocytes awaiting antigen encounter.",
    "Neurons"="Excitable cells transmitting signals within the nervous system.",
    "Neutrophils"="First-responder phagocytes mediating acute inflammation and pathogen clearance.",
    "NK cells"="Natural killer cells; innate cytotoxic effectors against tumors and virally infected cells.",
    "NKT"="Hybrid T/NK cells recognizing lipid antigens and bridging innate and adaptive immunity.",
    "Osteoblast"="Bone-forming cells secreting osteoid and regulating mineralization.",
    "pDC"="Plasmacytoid dendritic cells; major producers of type I interferons in viral defense.",
    "Pericytes"="Perivascular mural cells stabilizing capillaries and regulating blood flow.",
    "Plasma cells"="Terminally differentiated B-cells secreting antibodies.",
    "Platelets"="Anucleate blood fragments mediating hemostasis and releasing growth factors.",
    "Preadipocytes"="Precursors of adipocytes; can signal inflammation during adipose expansion.",
    "pro B-cells"="Early B-cell precursors undergoing immunoglobulin gene rearrangement.",
    "Sebocytes"="Sebum-producing skin cells contributing to lipid homeostasis and barrier function.",
    "Skeletal muscle"="Voluntary striated muscle; key for locomotion and energy metabolism.",
    "Smooth muscle"="Involuntary contractile cells in vessels and viscera.",
    "Tgd cells"="Gamma-delta T-cells; innate-like T-cells recognizing stress ligands.",
    "Th1 cells"="CD4+ helper subset producing IFN-γ; promotes cellular immunity.",
    "Th2 cells"="CD4+ helper subset producing IL-4/IL-5/IL-13; promotes humoral and allergic responses.",
    "Tregs"="Regulatory T-cells suppressing immune activation and maintaining tolerance.",
    "ImmuneScore"="Composite index summarizing overall immune-cell infiltration.",
    "StromaScore"="Composite index reflecting fibroblast and stromal content.",
    "MicroenvironmentScore"="Combined immune + stromal index representing overall TME composition."
  )
  BIO_DIR <- file.path(OUT_DIR, "ImmuneProfile_Context"); dir.create(BIO_DIR, showWarnings=FALSE)
  bio_df <- data.frame(CellType = names(xcell_bio_summary),
                       Context  = unname(unlist(xcell_bio_summary)),
                       check.names = FALSE)
  write.csv(bio_df, file.path(BIO_DIR, "xcell_bio_context.csv"), row.names = FALSE)

  # 13.1) Automated interpretations for ALL 64 CTs ---------------------------
  context_for <- function(ct) {
    if (!is.null(xcell_bio_summary[[ct]])) xcell_bio_summary[[ct]] else
      "Context varies by tissue and disease; integrate with histopathology and orthogonal assays."
  }
  median_by_ct <- apply(analysis_mat, 1, function(v) median(v, na.rm = TRUE))
  presence_frac <- rep(NA_real_, length(median_by_ct)); names(presence_frac) <- names(median_by_ct)
  if (exists("fdr_mat") && !is.null(fdr_mat)) presence_frac <- rowMeans(fdr_mat[rownames(analysis_mat), , drop = FALSE] <= PRES_FDR_ALPHA, na.rm = TRUE)

  level_from_median <- function(m){
    if (is.na(m)) return("Unknown")
    if (m >= THRESH_MED_HIGH) "High"
    else if (m >= THRESH_MED_MOD) "Moderate" else "Low"
  }
  status <- vapply(median_by_ct, level_from_median, FUN.VALUE = character(1))
  freq_presence <- ifelse(is.na(presence_frac), "NA",
                          ifelse(presence_frac >= PRES_MIN_FRAC, "Frequent", "Occasional/rare"))
  interpretation_for <- function(ct, med, pres_frac){
    lev <- level_from_median(med)
    pres_txt <- if (is.na(pres_frac)) "presence unknown"
    else if (pres_frac >= PRES_MIN_FRAC) "frequent presence" else "limited presence"
    paste0(ct, ": ", lev, " enrichment with ", pres_txt, ". ", context_for(ct))
  }
  interp_tbl <- data.frame(
    CellType         = names(median_by_ct),
    MedianEnrichment = as.numeric(median_by_ct),
    EnrichmentLevel  = status,
    PresenceFraction = as.numeric(presence_frac),
    PresenceClass    = freq_presence,
    Interpretation   = mapply(interpretation_for,
                              names(median_by_ct),
                              as.numeric(median_by_ct),
                              as.numeric(presence_frac),
                              SIMPLIFY = TRUE),
    check.names = FALSE
  )
  lev_order <- c("High","Moderate","Low","Unknown")
  interp_tbl$EnrichmentLevel <- factor(interp_tbl$EnrichmentLevel, levels = lev_order)
  interp_tbl <- interp_tbl[order(interp_tbl$EnrichmentLevel, -interp_tbl$PresenceFraction, interp_tbl$CellType), ]
  write.csv(interp_tbl, file.path(DIR_MATRICES, "xcell_cohort_celltype_enrichment_summary.csv"), row.names = FALSE)

  invisible(TRUE)
}
'''


from pathlib import Path
import rpy2.robjects as ro
R = ro.r

def run_xcell_pipeline(
    expr_file: str,
    meta_file: str,
    out_dir: str,
    do_install: bool = True,
    thresh_med_high: float = 0.20,
    thresh_med_mod: float = 0.10,
    pres_fdr_alpha: float = 0.10,
    pres_min_frac: float = 0.35,
    umap_min_samples: int = 20,
    max_cards: float = float('inf'),
    top_n_card: int = 25,
):
    """
    Run the xCell pipeline using rpy2.
    
    Args:
        expr_file: Complete path to the expression file (csv/tsv/txt/xlsx)
        meta_file: Complete path to the metadata file (csv/tsv/txt/xlsx)
        out_dir: Complete path to the output directory
        do_install: Whether to install/ensure R dependencies (default: True)
        thresh_med_high: Median threshold for 'High' enrichment (default: 0.20)
        thresh_med_mod: Median threshold for 'Moderate' enrichment (default: 0.10)
        pres_fdr_alpha: FDR alpha for presence testing (default: 0.10)
        pres_min_frac: Minimum fraction for 'Frequent' presence (default: 0.35)
        umap_min_samples: Minimum samples to run UMAP (default: 20)
        max_cards: Maximum number of sample cards (default: Inf)
        top_n_card: Top-N cell types per sample card (default: 25)
    
    Version-safe rpy2 call (works on Windows).
    Avoids ro.rinterface_lib and string-built R calls.
    """
    # Activate rpy2 conversion context (fix for multiprocessing/Celery)
    with conversion.localconverter(default_converter):
        # Load the R function definition into the R global environment
        R(R_PIPELINE)

        # Derive DATA_DIR from expr_file parent directory for R function compatibility
        data_dir = str(Path(expr_file).parent)

        # Handle Inf for R argument - use ro.r directly in context
        if max_cards == float('inf'):
            max_cards_r = ro.r('Inf')
        else:
            max_cards_r = max_cards

        # Fetch the R function and call it with named args
        r_fun = ro.globalenv['run_xcell_pipeline']
        r_fun(
            DATA_DIR=data_dir,
            EXPR_FILE=expr_file,
            META_FILE=meta_file,
            OUT_DIR=out_dir,
            DO_INSTALL=do_install,
            THRESH_MED_HIGH=thresh_med_high,
            THRESH_MED_MOD=thresh_med_mod,
            PRES_FDR_ALPHA=pres_fdr_alpha,
            PRES_MIN_FRAC=pres_min_frac,
            UMAP_MIN_SAMPLES=umap_min_samples,
            MAX_CARDS=max_cards_r,
            TOP_N_CARD=top_n_card,
        )
# if __name__ == "__main__":
#     run_pipeline(
#         expr_file=r"Lupus.csv",
#         meta_file=r"Lupus_meta.csv",
#         out_dir=r"xcell_output_V31_BIO",
#     )