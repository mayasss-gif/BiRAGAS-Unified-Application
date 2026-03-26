#!/usr/bin/env Rscript
# THIS IS A BACKUP CODE IF ANY THIGN GET WRONG USE THIS FILE AS R_CODE.R
options(stringsAsFactors = FALSE)

## ============================ OPTIONS =======================================
# Environment variables set by Python
get_bool  <- function(x, default = FALSE) {
  v <- toupper(Sys.getenv(x, ifelse(default, "TRUE", "FALSE"))); v %in% c("1","T","TRUE","YES","Y")
}
get_int   <- function(x, default = NA_integer_) {
  v <- suppressWarnings(as.integer(Sys.getenv(x, ""))); if (is.na(v)) default else v
}
get_num   <- function(x, default = NA_real_) {
  v <- suppressWarnings(as.numeric(Sys.getenv(x, ""))); if (is.na(v)) default else v
}
get_str   <- function(x, default = "") {
  v <- Sys.getenv(x, ""); if (!nzchar(v)) default else v
}

REF_DIR_PY            <- get_str("REF_DIR_PY", "")
OUT_DIR_PY            <- get_str("OUT_DIR_PY", "")
DROP_MITO_PY          <- get_bool("DROP_MITO_PY", FALSE)
ALLOW_DUP_PY          <- get_bool("ALLOW_DUP_PY", FALSE)
WARN_MIN_OVERLAP_PY   <- get_int ("WARN_MIN_OVERLAP_PY", 100L)
MAX_PROP_DEV_PY       <- get_num ("MAX_PROP_DEV_PY", 0.2)
TOP_MARKERS_PER_CT_PY <- get_int ("TOP_MARKERS_PER_CT_PY", 10L)
TOP_GENES_PER_SAMPLE_PY <- get_int("TOP_GENES_PER_SAMPLE_PY", 50L)
AUTO_INSTALL_PY       <- get_bool("AUTO_INSTALL_PY", TRUE)

if (!nzchar(REF_DIR_PY)) stop("Environment variable REF_DIR_PY not set; Python must pass it to R.")

## ============================ UTILITIES =====================================
msg <- function(...) cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] "), ..., "\n")

safe_dev_off <- function() {
  if (names(dev.cur()) != "null device") try(dev.off(), silent = TRUE)
}

ensure_double_matrix <- function(m) {
  if (isS4(m)) m <- as.matrix(m)
  if (!is.matrix(m)) m <- as.matrix(m)
  storage.mode(m) <- "double"; m
}

is_all_numeric_vec <- function(x) all(grepl("^[0-9]+$", as.character(x)))
norm_bc <- function(x){ x <- trimws(as.character(x)); x <- gsub("^X(?=\\d)","",x,perl=TRUE); x <- gsub("\\.", "-", x); x }
drop_mito <- function(genes) genes[!grepl("^MT-", toupper(genes))]
zero_var_rows <- function(m) { v <- apply(m, 1, function(x) var(x, na.rm=TRUE)); names(which(is.na(v) | v == 0)) }

collapse_dups <- function(m) {
  m <- ensure_double_matrix(m)
  if (is.null(rownames(m))) stop("Matrix lacks rownames.")
  if (!any(duplicated(rownames(m)))) return(m)
  msg("Duplicate gene IDs detected; collapsing by sum.")
  ux <- unique(rownames(m)); idx <- match(rownames(m), ux)
  sp <- Matrix::sparseMatrix(i = idx, j = seq_len(nrow(m)), x = 1,
                             dims = c(length(ux), nrow(m)))
  out <- ensure_double_matrix(as.matrix(sp %*% m))
  rownames(out) <- ux; colnames(out) <- colnames(m)
  out
}

drop_zero_expr_samples <- function(m, label="bulk") {
  m <- ensure_double_matrix(m)
  cs <- colSums(m, na.rm=TRUE); zc <- which(!is.finite(cs) | cs <= 0)
  if (length(zc)) {
    msg("Dropping ", length(zc), " ", label, " sample(s) with zero/NA expr: ",
        paste(colnames(m)[zc], collapse=", "))
    m <- m[, -zc, drop=FALSE]
  }
  if (ncol(m) == 0) stop("All ", label, " samples dropped; cannot proceed.")
  m
}

theme_pub <- function(){
  ggplot2::theme_bw(base_size=12) +
    ggplot2::theme(plot.title=ggplot2::element_text(face="bold"),
                   panel.grid.minor=ggplot2::element_blank())
}

cpm_mat <- function(m){
  m <- ensure_double_matrix(m)
  lib <- colSums(m,na.rm=TRUE); lib[!is.finite(lib)|lib==0] <- 1
  ensure_double_matrix(t(1e6*t(m)/lib))
}

harmonize_labels <- function(x) make.names(trimws(as.character(x)), unique = FALSE)

## =========================== PATHS & PARAMS =================================
REF_DIR <- REF_DIR_PY
OUT_DIR_DEFAULT <- file.path(REF_DIR, "bisque_deconv-results")
OUT_DIR <- if (!is.null(OUT_DIR_PY) && nzchar(OUT_DIR_PY)) OUT_DIR_PY else OUT_DIR_DEFAULT
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

DROP_MITO_GENES       <- isTRUE(DROP_MITO_PY)
ALLOW_DUPLICATE_GENES <- isTRUE(ALLOW_DUP_PY)
WARN_MIN_OVERLAP      <- as.integer(WARN_MIN_OVERLAP_PY)
MAX_PROP_DEVIATION    <- as.numeric(MAX_PROP_DEV_PY)
TOP_MARKERS_PER_CT    <- as.integer(TOP_MARKERS_PER_CT_PY)
TOP_GENES_PER_SAMPLE  <- as.integer(TOP_GENES_PER_SAMPLE_PY)
AUTO_INSTALL          <- isTRUE(AUTO_INSTALL_PY)

SC_COUNTS_FILE   <- file.path(REF_DIR, "sc_counts.tsv")
SC_META_FILE     <- file.path(REF_DIR, "sc_metadata.tsv")
BULK_COUNTS_FILE <- file.path(REF_DIR, "bulk_counts_overlap.tsv")

SAMPLE_META_FILE <- file.path(REF_DIR, "sample_metadata_patientAKI.tsv")  # optional
MARKER_FILE      <- file.path(REF_DIR, "marker_gene_table.tsv")           # optional

PLOT_DIR <- file.path(OUT_DIR, "deconv_visual_reports"); dir.create(PLOT_DIR, FALSE, TRUE)
TAB_DIR  <- file.path(OUT_DIR, "QC & Performance Metrics"); dir.create(TAB_DIR,  FALSE, TRUE)

## ============================ PACKAGES ======================================
cran_pkgs <- c("data.table","Matrix","tools","nnls","ggplot2","viridis",
               "pheatmap","corrplot","dplyr","tidyr","patchwork","ggrepel")
bio_pkgs  <- c("Biobase","BisqueRNA")

need_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
need_bio  <- bio_pkgs[!vapply(bio_pkgs,  requireNamespace, logical(1), quietly = TRUE)]

if (AUTO_INSTALL) {
  if (length(need_cran)) {
    msg("Installing CRAN packages: ", paste(need_cran, collapse=", "))
    install.packages(need_cran, repos = "https://cloud.r-project.org")
  }
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  if (length(need_bio)) {
    msg("Installing Bioconductor packages: ", paste(need_bio, collapse=", "))
    BiocManager::install(need_bio, update = FALSE, ask = FALSE)
  }
} else if (length(need_cran) || length(need_bio)) {
  stop("Missing R packages: ", paste(c(need_cran, need_bio), collapse=", "),
       "\nRun again with auto_install=TRUE or install manually.")
}

suppressPackageStartupMessages({
  lapply(c(cran_pkgs, bio_pkgs), library, character.only = TRUE)
})
capture.output(sessionInfo(), file = file.path(OUT_DIR, "R_sessionInfo.txt"))

## ======================= LOAD SINGLE-CELL DATA ==============================
stopifnot(file.exists(SC_COUNTS_FILE), file.exists(SC_META_FILE))
msg("Loading single-cell counts: ", SC_COUNTS_FILE)
cnt_full <- data.table::fread(SC_COUNTS_FILE, sep="\t", header=TRUE, data.table=FALSE)

# First column must be 'gene'; if not, treat the first as gene column anyway.
genes <- as.character(cnt_full[[1]]); cnt_full[[1]] <- NULL
m_sc <- ensure_double_matrix(as.matrix(cnt_full))
rownames(m_sc) <- genes
counts_bc_raw <- colnames(cnt_full)

msg("Loading metadata: ", SC_META_FILE)
md <- data.table::fread(SC_META_FILE, sep="\t", header=TRUE, data.table=FALSE)
if (!"cell_type" %in% names(md))     md$cell_type     <- NA_character_
if (!"individual_id" %in% names(md)) md$individual_id <- "ONE_SUBJECT"
if (!"cell_barcode" %in% names(md))  md$cell_barcode  <- NA_character_

counts_bc <- norm_bc(counts_bc_raw); meta_bc <- norm_bc(md$cell_barcode)
if (nrow(md) != length(counts_bc)) {
  stop(sprintf("Metadata rows (%d) != number of cells in counts (%d). Re-export so sizes match.",
               nrow(md), length(counts_bc)))
}
if (length(na.omit(unique(meta_bc))) <= 2 || is_all_numeric_vec(na.omit(meta_bc))) {
  md$cell_barcode <- counts_bc; meta_bc <- counts_bc
}
if (is_all_numeric_vec(counts_bc) && is_all_numeric_vec(meta_bc)) {
  msg("Counts & metadata both numeric; generating synthetic barcodes (in-memory).")
  syn <- sprintf("C%06d", seq_len(length(counts_bc)))
  colnames(m_sc) <- syn; md$cell_barcode <- syn
  counts_bc <- syn; meta_bc <- syn
} else {
  colnames(m_sc) <- counts_bc
}

# Align (transpose rescue + synthetic fallback)
ov <- intersect(colnames(m_sc), md$cell_barcode)
if (!length(ov) && any(rownames(m_sc) %in% md$cell_barcode)) {
  msg("Counts appear transposed; transposing ...")
  m_sc <- ensure_double_matrix(t(m_sc))
  ov <- intersect(colnames(m_sc), md$cell_barcode)
}
if (!length(ov)) {
  syn <- sprintf("C%06d", seq_len(ncol(m_sc)))
  colnames(m_sc) <- syn; md$cell_barcode <- syn; ov <- syn
}
m_sc <- m_sc[, ov, drop=FALSE]
md <- md[match(ov, md$cell_barcode), , drop=FALSE]
rownames(md) <- md$cell_barcode

# Build ExpressionSet for sc
pdat <- new("AnnotatedDataFrame", data = data.frame(
  cellType    = md$cell_type,
  SubjectName = md$individual_id,
  row.names   = rownames(md),
  check.names = FALSE
))
sc.eset <- Biobase::ExpressionSet(assayData = ensure_double_matrix(m_sc), phenoData = pdat)

## ============================== LOAD BULK ===================================
stopifnot(file.exists(BULK_COUNTS_FILE))
read_bulk <- function(path) {
  dt <- data.table::fread(path, sep="\t")
  cand <- c("gene","genes","gene_id","geneid","ensembl","ensembl_id","symbol",
            "gene_symbol","hgnc","Gene","GeneID","Gene_Symbol","Gene.Name","ID","Name")
  gcol <- intersect(cand, names(dt))[1]
  if (is.na(gcol)) {
    if (!is.numeric(dt[[1]]) && length(unique(dt[[1]])) == nrow(dt)) gcol <- names(dt)[1]
    else stop("Cannot detect gene column in bulk file.")
  }
  genes <- as.character(dt[[gcol]]); dt[[gcol]] <- NULL
  m <- ensure_double_matrix(as.matrix(dt))
  rownames(m) <- genes
  msg("Bulk matrix dims (genes x samples): ", nrow(m), " x ", ncol(m))
  m
}
bulk.mat <- read_bulk(BULK_COUNTS_FILE)

## ============================ SANITY + FILTERS ==============================
stopifnot(ncol(Biobase::exprs(sc.eset)) == nrow(Biobase::pData(sc.eset)))

if (any(Biobase::exprs(sc.eset) < 0, na.rm=TRUE)) stop("Negative values in single-cell counts.")
if (any(bulk.mat < 0, na.rm=TRUE))                stop("Negative values in bulk counts.")

if (ALLOW_DUPLICATE_GENES) {
  msg("Collapsing duplicate genes (safe rebuild)...")
  expr_new <- collapse_dups(Biobase::exprs(sc.eset))
  sc.eset <- Biobase::ExpressionSet(
    assayData = expr_new,
    phenoData = Biobase::phenoData(sc.eset)
  )
  bulk.mat <- collapse_dups(bulk.mat)
}

if (DROP_MITO_GENES) {
  keep_sc   <- drop_mito(rownames(sc.eset));    sc.eset  <- sc.eset[keep_sc, ]
  keep_bulk <- drop_mito(rownames(bulk.mat));   bulk.mat <- bulk.mat[keep_bulk, , drop=FALSE]
}

zv <- zero_var_rows(Biobase::exprs(sc.eset))
if (length(zv)) {
  sc.eset <- sc.eset[setdiff(rownames(Biobase::exprs(sc.eset)), zv), ]
  msg("Removed ", length(zv), " zero-variance genes from sc.")
}

bulk_zero_rows <- which(rowSums(bulk.mat == 0, na.rm=TRUE) == ncol(bulk.mat))
if (length(bulk_zero_rows)) {
  bulk.mat <- bulk.mat[-bulk_zero_rows, , drop=FALSE]
  msg("Removed ", length(bulk_zero_rows), " unexpressed genes from bulk (rows).")
}

# Ensure unique & ordered gene set
gene_set <- intersect(rownames(Biobase::exprs(sc.eset)), rownames(bulk.mat))
gene_set <- unique(gene_set)

if (length(gene_set) < WARN_MIN_OVERLAP) {
  msg("WARNING: low overlap gene count (", length(gene_set), "). Results may be less stable.")
}
if (length(gene_set) < 2L) {
  stop(sprintf("Too few overlapping genes after prefilters (|genes| = %d).", length(gene_set)))
}

# ExpressionSet-aware subsetting
sc.eset  <- sc.eset[gene_set, ]
bulk.mat <- ensure_double_matrix(bulk.mat[gene_set, , drop=FALSE])

# Dim sanity checks
msg(sprintf("Dims after alignment: sc.eset %d genes × %d cells; bulk %d genes × %d samples",
            nrow(Biobase::exprs(sc.eset)), ncol(Biobase::exprs(sc.eset)),
            nrow(bulk.mat), ncol(bulk.mat)))

# Drop bulk samples that are zero over the selected genes
bulk.mat <- drop_zero_expr_samples(bulk.mat, label="bulk")

## ============ DONOR-LEVEL PSEUDOBULK (CPM) & SIGNATURE MATRIX ===============
msg("Building donor-level pseudobulk (CPM) ...")
expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))     # genes × cells
ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
subj_vec<- as.character(Biobase::pData(sc.eset)$SubjectName)
key <- paste(subj_vec, ct_vec, sep="__")
idx <- split(seq_len(ncol(expr_sc)), key)
PB_counts <- sapply(idx, function(cols) Matrix::rowSums(expr_sc[, cols, drop=FALSE]))
PB_counts <- ensure_double_matrix(as.matrix(PB_counts))
PB_cpm <- cpm_mat(PB_counts)
pb_path <- file.path(OUT_DIR, "pseudobulk_donor_level.tsv")
data.table::fwrite(data.frame(gene = rownames(PB_cpm), PB_cpm, check.names = FALSE),
                   pb_path, sep = "\t")
msg("Saved donor-level pseudobulk → ", pb_path)

keep_cells <- !is.na(ct_vec) & nzchar(ct_vec)
if (any(!keep_cells)) {
  msg("Dropping ", sum(!keep_cells), " cells with missing/empty cell_type.")
  sc.eset <- sc.eset[, keep_cells]
  expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))
  ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
}

expr <- ensure_double_matrix(Biobase::exprs(sc.eset))        # genes × cells
lib  <- colSums(expr); lib[lib == 0] <- 1
expr_cpm <- ensure_double_matrix(t(1e6 * t(expr) / lib))
idxs <- split(seq_along(ct_vec), ct_vec)
Z <- sapply(idxs, function(cols) rowMeans(expr_cpm[, cols, drop=FALSE], na.rm=TRUE))
Z <- ensure_double_matrix(as.matrix(Z))
colnames(Z) <- make.names(colnames(Z), unique=TRUE)
Z[!is.finite(Z)] <- 0
keep_gene_sig <- rowSums(Z) > 0
if (any(!keep_gene_sig)) {
  msg("Signature: dropping ", sum(!keep_gene_sig), " genes with all-zero across cell types.")
  Z <- Z[keep_gene_sig, , drop=FALSE]
}
sig_df <- data.frame(gene = rownames(Z), Z, check.names = FALSE)
sig_path <- file.path(OUT_DIR, "signature_matrix.tsv")
data.table::fwrite(sig_df, sig_path, sep="\t")
msg("Signature matrix saved: ", sig_path)

## ===================== BISQUE (>=2 subjects) or NNLS (1) ====================
n_subjects <- length(unique(Biobase::pData(sc.eset)$SubjectName))
msg("Single-cell unique subjects detected: ", n_subjects)

if (n_subjects >= 2) {
  genes_use <- intersect(rownames(sc.eset), rownames(bulk.mat))
  bulk_use  <- ensure_double_matrix(bulk.mat[genes_use, , drop=FALSE])
  bulk_use  <- drop_zero_expr_samples(bulk_use, label="bulk (pre-Bisque)")
  bulk.eset <- Biobase::ExpressionSet(assayData = bulk_use)
  msg("Running BisqueRNA::ReferenceBasedDecomposition ...")
  res <- BisqueRNA::ReferenceBasedDecomposition(
    bulk.eset     = bulk.eset,
    sc.eset       = sc.eset,
    markers       = NULL,
    cell.types    = "cellType",
    subject.names = "SubjectName",
    use.overlap   = FALSE,
    verbose       = TRUE,
    old.cpm       = TRUE
  )
  bulk_props <- ensure_double_matrix(res$bulk.props)
} else {
  msg("Fallback: only ONE subject in scRNA-seq; running NNLS on signature matrix.")
  common_genes <- intersect(rownames(Z), rownames(bulk.mat))
  if (length(common_genes) < WARN_MIN_OVERLAP)
    msg("WARNING: low overlap gene count for NNLS (", length(common_genes), ").")
  A <- ensure_double_matrix(Z[common_genes, , drop=FALSE])        # genes × CT
  B <- ensure_double_matrix(bulk.mat[common_genes, , drop=FALSE]) # genes × samples
  A[!is.finite(A)] <- 0; B[!is.finite(B)] <- 0
  keep_gene <- rowSums(A) > 0
  if (any(!keep_gene)) { msg("NNLS: dropping ", sum(!keep_gene), " genes with zero signature.");
    A <- A[keep_gene, , drop=FALSE]; B <- B[keep_gene, , drop=FALSE] }
  keep_ct <- colSums(A) > 0
  if (any(!keep_ct)) { msg("NNLS: dropping ", sum(!keep_ct), " CT with zero signature.");
    A <- A[, keep_ct, drop=FALSE] }
  if (ncol(A) == 0) stop("NNLS: no non-zero cell types remain after filtering.")
  P_nnls <- matrix(NA_real_, nrow=ncol(A), ncol=ncol(B),
                   dimnames=list(colnames(A), colnames(B)))
  for (j in seq_len(ncol(B))) {
    fit <- nnls::nnls(A, B[, j]); p <- as.numeric(fit$x); p[!is.finite(p)] <- 0
    s <- sum(p); if (is.finite(s) && s > 0) p <- p / s
    P_nnls[, j] <- p
  }
  bulk_props <- ensure_double_matrix(P_nnls)
  res <- list(bulk.props = bulk_props, genes.used = rownames(A))
}

## =============================== OUTPUTS ====================================
prop_sums  <- colSums(bulk_props)
bad_sum    <- which(abs(prop_sums - 1) > MAX_PROP_DEVIATION)
if (length(bad_sum))
  msg("WARNING: |sum(props)-1| >", MAX_PROP_DEVIATION, " for samples: ",
      paste(colnames(bulk_props)[bad_sum], collapse=", "))

# TSV (canonical)
data.table::fwrite(
  data.frame(cell_type = rownames(bulk_props), bulk_props, check.names = FALSE),
  file.path(OUT_DIR, "bisque_bulk_proportions.tsv"), sep = "\t"
)
# CSV (for older Python code that expects it)
utils::write.csv(
  data.frame(bulk_props, check.names = FALSE),
  file = file.path(OUT_DIR, "Bisque_proportions.csv"),
  row.names = TRUE
)

if (!is.null(res$sc.props)) {
  data.table::fwrite(
    data.frame(cell_type = rownames(res$sc.props), res$sc.props, check.names = FALSE),
    file.path(OUT_DIR, "bisque_sc_proportions.tsv"), sep = "\t"
  )
}

if (!is.null(res$genes.used)) {
  writeLines(res$genes.used, file.path(OUT_DIR, "bisque_genes_used.txt"))
} else {
  writeLines(rownames(bulk.mat), file.path(OUT_DIR, "bisque_genes_used.txt"))
}

# Bar of sums (open an explicit device)
png(filename = file.path(OUT_DIR, "prop_sums.png"), width = 1200, height = 400)
par(mar=c(6,4,2,1))
barplot(prop_sums, las = 2, main = "Sum of estimated proportions per sample")
abline(h = 1, col = "red", lty = 2)
safe_dev_off()

## ======================= POST-HOC QC + VISUALS ==============================
PROP_FILE <- file.path(OUT_DIR, "bisque_bulk_proportions.tsv")
SIG_FILE  <- file.path(OUT_DIR, "signature_matrix.tsv")
stopifnot(file.exists(PROP_FILE), file.exists(SIG_FILE))

props_dt <- data.table::fread(PROP_FILE)
stopifnot("cell_type" %in% names(props_dt))
P_raw <- as.matrix(props_dt[, -1, with=FALSE]); rownames(P_raw) <- props_dt$cell_type
P_raw <- ensure_double_matrix(P_raw)
samples <- colnames(P_raw)

sig_dt <- data.table::fread(SIG_FILE)
gcol <- if ("gene" %in% names(sig_dt)) "gene" else names(sig_dt)[1]
genes <- sig_dt[[gcol]]; sig_dt[[gcol]] <- NULL
Z_raw <- ensure_double_matrix(as.matrix(sig_dt)); rownames(Z_raw) <- genes

# Harmonize CT labels
colnames(Z_raw) <- harmonize_labels(colnames(Z_raw))
rownames(P_raw) <- harmonize_labels(rownames(P_raw))
if (any(duplicated(colnames(Z_raw))))  stop("Duplicate CTs in signature (Z).")
if (any(duplicated(rownames(P_raw))))  stop("Duplicate CTs in proportions (P).")
CT <- intersect(colnames(Z_raw), rownames(P_raw))
if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d) after harmonization.", length(CT)))
Z_use <- ensure_double_matrix(Z_raw[, CT, drop=FALSE])   # genes x CT
P_use <- ensure_double_matrix(P_raw[CT,  , drop=FALSE])  # CT x samples

# Tidy for plots
props_dt_use <- data.frame(cell_type = rownames(P_use), P_use, check.names = FALSE)
props_long <- data.table::melt(data.table::as.data.table(props_dt_use), id.vars="cell_type",
                               variable.name="sample", value.name="prop")
wide <- data.table::dcast(props_long, sample ~ cell_type, value.var="prop")
prop_mat <- as.matrix(wide[, -1, with=FALSE]); rownames(prop_mat) <- wide$sample
prop_mat <- ensure_double_matrix(prop_mat)

has_sample_meta <- !is.null(SAMPLE_META_FILE) && file.exists(SAMPLE_META_FILE)
if (has_sample_meta) meta <- data.table::fread(SAMPLE_META_FILE)

have_bulk <- file.exists(BULK_COUNTS_FILE)
if (have_bulk) {
  bulk_dt <- data.table::fread(BULK_COUNTS_FILE)
  gcol2 <- intersect(c("gene","symbol","Gene","Gene_Symbol","ID","Name","GeneID","genes"), names(bulk_dt))[1]
  if (is.na(gcol2)) gcol2 <- names(bulk_dt)[1]
  bgenes <- bulk_dt[[gcol2]]; bulk_dt[[gcol2]] <- NULL
  X_bulk <- ensure_double_matrix(as.matrix(bulk_dt)); rownames(X_bulk) <- as.character(bgenes)
  X_cpm  <- cpm_mat(X_bulk)
}

have_sc_props <- !is.null(res$sc.props)
if (have_sc_props) {
  Ct_raw <- ensure_double_matrix(as.matrix(res$sc.props))
  rownames(Ct_raw) <- harmonize_labels(rownames(res$sc.props))
}

# --- QC: sums table + hist
prop_sums <- colSums(P_use, na.rm=TRUE)
qc_df <- data.frame(sample=names(prop_sums), sum_props=as.numeric(prop_sums),
                    abs_dev=abs(as.numeric(prop_sums)-1))
data.table::fwrite(qc_df, file.path(TAB_DIR,"qc_prop_sums.tsv"), sep="\t")

g_hist <- ggplot2::ggplot(qc_df, ggplot2::aes(sum_props)) +
  ggplot2::geom_histogram(bins=30, alpha=.9) +
  ggplot2::geom_vline(xintercept=1, linetype=2, color="red") +
  theme_pub() + ggplot2::labs(title="Distribution of proportion sums", x="Sum of proportions", y="Count")
ggplot2::ggsave(file.path(PLOT_DIR,"02_qc_sum_hist.png"), g_hist, width=6.2, height=4.5, dpi=200)

# --- Stacked composition
hc <- hclust(dist(prop_mat), method = "complete")
sample_order <- rownames(prop_mat)[hc$order]
props_long$sample <- factor(props_long$sample, levels = sample_order)
gg1 <- ggplot2::ggplot(props_long, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
  ggplot2::geom_bar(stat = "identity", width = 0.95) +
  ggplot2::labs(title = "Cell-type composition per sample (cluster-ordered)", x = "Sample", y = "Proportion") +
  ggplot2::theme_bw() + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=90, vjust=0.5, hjust=1))
ggplot2::ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), gg1, width = 16, height = 7, dpi = 200)

# --- Heatmap of proportions
pheatmap::pheatmap(prop_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
                   filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
                   main="Proportions heatmap", width=10, height=13)

# --- Violin/box of CT distributions
g_boxv <- ggplot2::ggplot(props_long, ggplot2::aes(cell_type, prop, fill=cell_type)) +
  ggplot2::geom_violin(trim=FALSE, alpha=.85) + ggplot2::geom_boxplot(width=.12, outlier.size=.4) +
  viridis::scale_fill_viridis(discrete=TRUE, guide="none") + theme_pub() +
  ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45, hjust=1)) +
  ggplot2::labs(title="Distribution of Bisque-estimated proportions", x="Cell type", y="Proportion")
ggplot2::ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), g_boxv, width=11, height=6.5, dpi=200)

## =================== OPTIONAL MARKER CHECKS & EXPORTS =======================
eps <- 1e-8
other_mean <- function(mat, ct) { if (ncol(mat)==1) return(rep(0, nrow(mat))); rowMeans(mat[, setdiff(colnames(mat), ct), drop=FALSE], na.rm=TRUE) }
spec_list <- lapply(colnames(Z_use), function(ct){
  spec <- log2((Z_use[, ct] + eps) / (other_mean(Z_use, ct) + eps))
  data.frame(gene = rownames(Z_use), cell_type = ct, specificity = spec, expr_ct = Z_use[, ct], stringsAsFactors = FALSE)
})
spec_df <- data.table::rbindlist(spec_list)
markers_sel <- spec_df[order(cell_type, -specificity), .SD[1:min(.N, TOP_MARKERS_PER_CT)], by = cell_type]
data.table::fwrite(markers_sel, file.path(TAB_DIR, "top_markers_per_celltype.tsv"), sep="\t")

if (!is.null(MARKER_FILE) && file.exists(MARKER_FILE) && have_bulk) {
  mk <- data.table::fread(MARKER_FILE)
  if (nrow(mk)) {
    data.table::setnames(mk, old = names(mk), new = tolower(names(mk)))
    gene_col <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","feature","id","geneid","name"), names(mk))[1]
    ct_col   <- intersect(c("cell_type","celltype","cell_type_label","celltype_label","cluster","label","cell","ct"), names(mk))[1]
    if (!is.na(gene_col) && !is.na(ct_col)) {
      markers <- mk[, .(gene = get(gene_col), cell_type = harmonize_labels(get(ct_col)))]
      markers <- unique(markers[!is.na(gene) & nzchar(gene) & !is.na(cell_type) & nzchar(cell_type)])
      bulk_dt0 <- data.table::fread(BULK_COUNTS_FILE)
      gcol3 <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","id","name","geneid","genes"), names(bulk_dt0))[1]
      if (is.na(gcol3)) gcol3 <- names(bulk_dt0)[1]
      bgenes <- bulk_dt0[[gcol3]]; bulk_dt0[[gcol3]] <- NULL
      bulk_mat <- ensure_double_matrix(as.matrix(bulk_dt0)); rownames(bulk_mat) <- as.character(bgenes)
      bulk_cpm <- cpm_mat(bulk_mat)
      ct_ok <- intersect(unique(markers$cell_type), rownames(P_use))
      ctab <- lapply(ct_ok, function(ct) {
        gs <- intersect(markers[cell_type == ct, gene], rownames(bulk_cpm))
        if (!length(gs)) return(NULL)
        score <- colMeans(bulk_cpm[gs, , drop = FALSE])
        common <- intersect(names(score), colnames(P_use))
        if (length(common) < 3) return(NULL)
        rho <- suppressWarnings(cor(score[common], P_use[ct, common], method = "spearman"))
        data.frame(cell_type = ct, n_genes = length(gs), n_samples = length(common), spearman_rho = rho)
      })
      ctab <- data.table::rbindlist(ctab, fill = TRUE)
      if (!is.null(ctab) && nrow(ctab)) data.table::fwrite(ctab, file.path(TAB_DIR, "marker_prop_spearman.tsv"), sep = "\t")
    }
  }
}

msg("DONE. Outputs written to: ", OUT_DIR)
msg("Tables → ", TAB_DIR)
msg("Plots  → ", PLOT_DIR)
