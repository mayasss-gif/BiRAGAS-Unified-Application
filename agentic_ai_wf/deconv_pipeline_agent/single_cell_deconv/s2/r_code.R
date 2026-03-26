# # # #!/usr/bin/env Rscript

# # # options(stringsAsFactors = FALSE)
# # # ### NEW: deterministic randomness for any sampling/plots
# # # set.seed(42L)

# # # ## ============================ OPTIONS =======================================
# # # # Environment variables set by Python
# # # get_bool  <- function(x, default = FALSE) {
# # #   v <- toupper(Sys.getenv(x, ifelse(default, "TRUE", "FALSE"))); v %in% c("1","T","TRUE","YES","Y")
# # # }
# # # get_int   <- function(x, default = NA_integer_) {
# # #   v <- suppressWarnings(as.integer(Sys.getenv(x, ""))); if (is.na(v)) default else v
# # # }
# # # get_num   <- function(x, default = NA_real_) {
# # #   v <- suppressWarnings(as.numeric(Sys.getenv(x, ""))); if (is.na(v)) default else v
# # # }
# # # get_str   <- function(x, default = "") {
# # #   v <- Sys.getenv(x, ""); if (!nzchar(v)) default else v
# # # }

# # # REF_DIR_PY            <- get_str("REF_DIR_PY", "")
# # # OUT_DIR_PY            <- get_str("OUT_DIR_PY", "")
# # # DROP_MITO_PY          <- get_bool("DROP_MITO_PY", FALSE)
# # # ALLOW_DUP_PY          <- get_bool("ALLOW_DUP_PY", FALSE)
# # # WARN_MIN_OVERLAP_PY   <- get_int ("WARN_MIN_OVERLAP_PY", 100L)
# # # MAX_PROP_DEV_PY       <- get_num ("MAX_PROP_DEV_PY", 0.2)
# # # TOP_MARKERS_PER_CT_PY <- get_int ("TOP_MARKERS_PER_CT_PY", 10L)
# # # TOP_GENES_PER_SAMPLE_PY <- get_int("TOP_GENES_PER_SAMPLE_PY", 50L)
# # # AUTO_INSTALL_PY       <- get_bool("AUTO_INSTALL_PY", TRUE)

# # # if (!nzchar(REF_DIR_PY)) stop("Environment variable REF_DIR_PY not set; Python must pass it to R.")

# # # ## ============================ UTILITIES =====================================
# # # msg <- function(...) cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] "), ..., "\n")

# # # safe_dev_off <- function() {
# # #   if (names(dev.cur()) != "null device") try(dev.off(), silent = TRUE)
# # # }

# # # ensure_double_matrix <- function(m) {
# # #   if (isS4(m)) m <- as.matrix(m)
# # #   if (!is.matrix(m)) m <- as.matrix(m)
# # #   storage.mode(m) <- "double"; m
# # # }

# # # is_all_numeric_vec <- function(x) all(grepl("^[0-9]+$", as.character(x)))
# # # norm_bc <- function(x){ x <- trimws(as.character(x)); x <- gsub("^X(?=\\d)","",x,perl=TRUE); x <- gsub("\\.", "-", x); x }
# # # drop_mito <- function(genes) genes[!grepl("^MT-", toupper(genes))]
# # # zero_var_rows <- function(m) { v <- apply(m, 1, function(x) var(x, na.rm=TRUE)); names(which(is.na(v) | v == 0)) }

# # # collapse_dups <- function(m) {
# # #   m <- ensure_double_matrix(m)
# # #   if (is.null(rownames(m))) stop("Matrix lacks rownames.")
# # #   if (!any(duplicated(rownames(m)))) return(m)
# # #   msg("Duplicate gene IDs detected; collapsing by sum.")
# # #   ux <- unique(rownames(m)); idx <- match(rownames(m), ux)
# # #   sp <- Matrix::sparseMatrix(i = idx, j = seq_len(nrow(m)), x = 1,
# # #                              dims = c(length(ux), nrow(m)))
# # #   out <- ensure_double_matrix(as.matrix(sp %*% m))
# # #   rownames(out) <- ux; colnames(out) <- colnames(m)
# # #   out
# # # }

# # # drop_zero_expr_samples <- function(m, label="bulk") {
# # #   m <- ensure_double_matrix(m)
# # #   cs <- colSums(m, na.rm=TRUE); zc <- which(!is.finite(cs) | cs <= 0)
# # #   if (length(zc)) {
# # #     msg("Dropping ", length(zc), " ", label, " sample(s) with zero/NA expr: ",
# # #         paste(colnames(m)[zc], collapse=", "))
# # #     m <- m[, -zc, drop=FALSE]
# # #   }
# # #   if (ncol(m) == 0) stop("All ", label, " samples dropped; cannot proceed.")
# # #   m
# # # }

# # # theme_pub <- function(){
# # #   ggplot2::theme_bw(base_size=12) +
# # #     ggplot2::theme(plot.title=ggplot2::element_text(face="bold"),
# # #                    panel.grid.minor=ggplot2::element_blank())
# # # }

# # # cpm_mat <- function(m){
# # #   m <- ensure_double_matrix(m)
# # #   lib <- colSums(m,na.rm=TRUE); lib[!is.finite(lib)|lib==0] <- 1
# # #   ensure_double_matrix(t(1e6*t(m)/lib))
# # # }

# # # harmonize_labels <- function(x) make.names(trimws(as.character(x)), unique = FALSE)

# # # ## =========================== PATHS & PARAMS =================================
# # # REF_DIR <- REF_DIR_PY
# # # OUT_DIR_DEFAULT <- file.path(REF_DIR, "bisque_deconv-results")
# # # OUT_DIR <- if (!is.null(OUT_DIR_PY) && nzchar(OUT_DIR_PY)) OUT_DIR_PY else OUT_DIR_DEFAULT
# # # dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# # # DROP_MITO_GENES       <- isTRUE(DROP_MITO_PY)
# # # ALLOW_DUPLICATE_GENES <- isTRUE(ALLOW_DUP_PY)
# # # WARN_MIN_OVERLAP      <- as.integer(WARN_MIN_OVERLAP_PY)
# # # MAX_PROP_DEVIATION    <- as.numeric(MAX_PROP_DEV_PY)
# # # TOP_MARKERS_PER_CT    <- as.integer(TOP_MARKERS_PER_CT_PY)
# # # TOP_GENES_PER_SAMPLE  <- as.integer(TOP_GENES_PER_SAMPLE_PY)
# # # AUTO_INSTALL          <- isTRUE(AUTO_INSTALL_PY)

# # # SC_COUNTS_FILE   <- file.path(REF_DIR, "sc_counts.tsv")
# # # SC_META_FILE     <- file.path(REF_DIR, "sc_metadata.tsv")
# # # BULK_COUNTS_FILE <- file.path(REF_DIR, "bulk_counts_overlap.tsv")

# # # SAMPLE_META_FILE <- file.path(REF_DIR, "sample_metadata_patientAKI.tsv")  # optional
# # # MARKER_FILE      <- file.path(REF_DIR, "marker_gene_table.tsv")           # optional

# # # PLOT_DIR <- file.path(OUT_DIR, "deconv_visual_reports"); dir.create(PLOT_DIR, FALSE, TRUE)
# # # TAB_DIR  <- file.path(OUT_DIR, "QC & Performance Metrics"); dir.create(TAB_DIR,  FALSE, TRUE)

# # # ## ============================ PACKAGES ======================================
# # # cran_pkgs <- c("data.table","Matrix","tools","nnls","ggplot2","viridis",
# # #                "pheatmap","corrplot","dplyr","tidyr","patchwork","ggrepel")
# # # bio_pkgs  <- c("Biobase","BisqueRNA")

# # # need_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
# # # need_bio  <- bio_pkgs[!vapply(bio_pkgs,  requireNamespace, logical(1), quietly = TRUE)]

# # # if (AUTO_INSTALL) {
# # #   if (length(need_cran)) {
# # #     msg("Installing CRAN packages: ", paste(need_cran, collapse=", "))
# # #     install.packages(need_cran, repos = "https://cloud.r-project.org")
# # #   }
# # #   if (!requireNamespace("BiocManager", quietly = TRUE)) {
# # #     install.packages("BiocManager", repos = "https://cloud.r-project.org")
# # #   }
# # #   if (length(need_bio)) {
# # #     msg("Installing Bioconductor packages: ", paste(need_bio, collapse=", "))
# # #     BiocManager::install(need_bio, update = FALSE, ask = FALSE)
# # #   }
# # # } else if (length(need_cran) || length(need_bio)) {
# # #   stop("Missing R packages: ", paste(c(need_cran, need_bio), collapse=", "),
# # #        "\nRun again with auto_install=TRUE or install manually.")
# # # }

# # # suppressPackageStartupMessages({
# # #   lapply(c(cran_pkgs, bio_pkgs), library, character.only = TRUE)
# # # })
# # # capture.output(sessionInfo(), file = file.path(OUT_DIR, "R_sessionInfo.txt"))

# # # ## ======================= LOAD SINGLE-CELL DATA ==============================
# # # stopifnot(file.exists(SC_COUNTS_FILE), file.exists(SC_META_FILE))
# # # msg("Loading single-cell counts: ", SC_COUNTS_FILE)
# # # cnt_full <- data.table::fread(SC_COUNTS_FILE, sep="\t", header=TRUE, data.table=FALSE)

# # # # First column must be 'gene'; if not, treat the first as gene column anyway.
# # # genes <- as.character(cnt_full[[1]]); cnt_full[[1]] <- NULL
# # # m_sc <- ensure_double_matrix(as.matrix(cnt_full))
# # # rownames(m_sc) <- genes
# # # counts_bc_raw <- colnames(cnt_full)

# # # msg("Loading metadata: ", SC_META_FILE)
# # # md <- data.table::fread(SC_META_FILE, sep="\t", header=TRUE, data.table=FALSE)
# # # if (!"cell_type" %in% names(md))     md$cell_type     <- NA_character_
# # # if (!"individual_id" %in% names(md)) md$individual_id <- "ONE_SUBJECT"
# # # if (!"cell_barcode" %in% names(md))  md$cell_barcode  <- NA_character_

# # # counts_bc <- norm_bc(counts_bc_raw); meta_bc <- norm_bc(md$cell_barcode)
# # # if (nrow(md) != length(counts_bc)) {
# # #   stop(sprintf("Metadata rows (%d) != number of cells in counts (%d). Re-export so sizes match.",
# # #                nrow(md), length(counts_bc)))
# # # }
# # # if (length(na.omit(unique(meta_bc))) <= 2 || is_all_numeric_vec(na.omit(meta_bc))) {
# # #   md$cell_barcode <- counts_bc; meta_bc <- counts_bc
# # # }
# # # if (is_all_numeric_vec(counts_bc) && is_all_numeric_vec(meta_bc)) {
# # #   msg("Counts & metadata both numeric; generating synthetic barcodes (in-memory).")
# # #   syn <- sprintf("C%06d", seq_len(length(counts_bc)))
# # #   colnames(m_sc) <- syn; md$cell_barcode <- syn
# # #   counts_bc <- syn; meta_bc <- syn
# # # } else {
# # #   colnames(m_sc) <- counts_bc
# # # }

# # # # Align (transpose rescue + synthetic fallback)
# # # ov <- intersect(colnames(m_sc), md$cell_barcode)
# # # if (!length(ov) && any(rownames(m_sc) %in% md$cell_barcode)) {
# # #   msg("Counts appear transposed; transposing ...")
# # #   m_sc <- ensure_double_matrix(t(m_sc))
# # #   ov <- intersect(colnames(m_sc), md$cell_barcode)
# # # }
# # # if (!length(ov)) {
# # #   syn <- sprintf("C%06d", seq_len(ncol(m_sc)))
# # #   colnames(m_sc) <- syn; md$cell_barcode <- syn; ov <- syn
# # # }
# # # m_sc <- m_sc[, ov, drop=FALSE]
# # # md <- md[match(ov, md$cell_barcode), , drop=FALSE]
# # # rownames(md) <- md$cell_barcode

# # # # Build ExpressionSet for sc
# # # pdat <- new("AnnotatedDataFrame", data = data.frame(
# # #   cellType    = md$cell_type,
# # #   SubjectName = md$individual_id,
# # #   row.names   = rownames(md),
# # #   check.names = FALSE
# # # ))
# # # sc.eset <- Biobase::ExpressionSet(assayData = ensure_double_matrix(m_sc), phenoData = pdat)

# # # ## ============================== LOAD BULK ===================================
# # # stopifnot(file.exists(BULK_COUNTS_FILE))
# # # read_bulk <- function(path) {
# # #   dt <- data.table::fread(path, sep="\t")
# # #   cand <- c("gene","genes","gene_id","geneid","ensembl","ensembl_id","symbol",
# # #             "gene_symbol","hgnc","Gene","GeneID","Gene_Symbol","Gene.Name","ID","Name")
# # #   gcol <- intersect(cand, names(dt))[1]
# # #   if (is.na(gcol)) {
# # #     if (!is.numeric(dt[[1]]) && length(unique(dt[[1]])) == nrow(dt)) gcol <- names(dt)[1]
# # #     else stop("Cannot detect gene column in bulk file.")
# # #   }
# # #   genes <- as.character(dt[[gcol]]); dt[[gcol]] <- NULL
# # #   m <- ensure_double_matrix(as.matrix(dt))
# # #   rownames(m) <- genes
# # #   msg("Bulk matrix dims (genes x samples): ", nrow(m), " x ", ncol(m))
# # #   m
# # # }
# # # bulk.mat <- read_bulk(BULK_COUNTS_FILE)

# # # ## ============================ SANITY + FILTERS ==============================
# # # stopifnot(ncol(Biobase::exprs(sc.eset)) == nrow(Biobase::pData(sc.eset)))

# # # if (any(Biobase::exprs(sc.eset) < 0, na.rm=TRUE)) stop("Negative values in single-cell counts.")
# # # if (any(bulk.mat < 0, na.rm=TRUE))                stop("Negative values in bulk counts.")

# # # if (ALLOW_DUPLICATE_GENES) {
# # #   msg("Collapsing duplicate genes (safe rebuild)...")
# # #   expr_new <- collapse_dups(Biobase::exprs(sc.eset))
# # #   sc.eset <- Biobase::ExpressionSet(
# # #     assayData = expr_new,
# # #     phenoData = Biobase::phenoData(sc.eset)
# # #   )
# # #   bulk.mat <- collapse_dups(bulk.mat)
# # # }

# # # if (DROP_MITO_GENES) {
# # #   keep_sc   <- drop_mito(rownames(sc.eset));    sc.eset  <- sc.eset[keep_sc, ]
# # #   keep_bulk <- drop_mito(rownames(bulk.mat));   bulk.mat <- bulk.mat[keep_bulk, , drop=FALSE]
# # # }

# # # zv <- zero_var_rows(Biobase::exprs(sc.eset))
# # # if (length(zv)) {
# # #   sc.eset <- sc.eset[setdiff(rownames(Biobase::exprs(sc.eset)), zv), ]
# # #   msg("Removed ", length(zv), " zero-variance genes from sc.")
# # # }

# # # bulk_zero_rows <- which(rowSums(bulk.mat == 0, na.rm=TRUE) == ncol(bulk.mat))
# # # if (length(bulk_zero_rows)) {
# # #   bulk.mat <- bulk.mat[-bulk_zero_rows, , drop=FALSE]
# # #   msg("Removed ", length(bulk_zero_rows), " unexpressed genes from bulk (rows).")
# # # }

# # # # Ensure unique & ordered gene set
# # # gene_set <- intersect(rownames(Biobase::exprs(sc.eset)), rownames(bulk.mat))
# # # gene_set <- unique(gene_set)

# # # if (length(gene_set) < WARN_MIN_OVERLAP) {
# # #   msg("WARNING: low overlap gene count (", length(gene_set), "). Results may be less stable.")
# # # }
# # # if (length(gene_set) < 2L) {
# # #   stop(sprintf("Too few overlapping genes after prefilters (|genes| = %d).", length(gene_set)))
# # # }

# # # # ExpressionSet-aware subsetting
# # # sc.eset  <- sc.eset[gene_set, ]
# # # bulk.mat <- ensure_double_matrix(bulk.mat[gene_set, , drop=FALSE])

# # # # Dim sanity checks
# # # msg(sprintf("Dims after alignment: sc.eset %d genes × %d cells; bulk %d genes × %d samples",
# # #             nrow(Biobase::exprs(sc.eset)), ncol(Biobase::exprs(sc.eset)),
# # #             nrow(bulk.mat), ncol(bulk.mat)))

# # # # Drop bulk samples that are zero over the selected genes
# # # bulk.mat <- drop_zero_expr_samples(bulk.mat, label="bulk")

# # # ## ============ DONOR-LEVEL PSEUDOBULK (CPM) & SIGNATURE MATRIX ===============
# # # msg("Building donor-level pseudobulk (CPM) ...")
# # # expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))     # genes × cells
# # # ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# # # subj_vec<- as.character(Biobase::pData(sc.eset)$SubjectName)
# # # key <- paste(subj_vec, ct_vec, sep="__")
# # # idx <- split(seq_len(ncol(expr_sc)), key)
# # # PB_counts <- sapply(idx, function(cols) Matrix::rowSums(expr_sc[, cols, drop=FALSE]))
# # # PB_counts <- ensure_double_matrix(as.matrix(PB_counts))
# # # PB_cpm <- cpm_mat(PB_counts)
# # # pb_path <- file.path(OUT_DIR, "pseudobulk_donor_level.tsv")
# # # data.table::fwrite(data.frame(gene = rownames(PB_cpm), PB_cpm, check.names = FALSE),
# # #                    pb_path, sep = "\t")
# # # msg("Saved donor-level pseudobulk → ", pb_path)

# # # keep_cells <- !is.na(ct_vec) & nzchar(ct_vec)
# # # if (any(!keep_cells)) {
# # #   msg("Dropping ", sum(!keep_cells), " cells with missing/empty cell_type.")
# # #   sc.eset <- sc.eset[, keep_cells]
# # #   expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))
# # #   ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# # # }

# # # expr <- ensure_double_matrix(Biobase::exprs(sc.eset))        # genes × cells
# # # lib  <- colSums(expr); lib[lib == 0] <- 1
# # # expr_cpm <- ensure_double_matrix(t(1e6 * t(expr) / lib))
# # # idxs <- split(seq_along(ct_vec), ct_vec)
# # # Z <- sapply(idxs, function(cols) rowMeans(expr_cpm[, cols, drop=FALSE], na.rm=TRUE))
# # # Z <- ensure_double_matrix(as.matrix(Z))
# # # colnames(Z) <- make.names(colnames(Z), unique=TRUE)
# # # Z[!is.finite(Z)] <- 0
# # # keep_gene_sig <- rowSums(Z) > 0
# # # if (any(!keep_gene_sig)) {
# # #   msg("Signature: dropping ", sum(!keep_gene_sig), " genes with all-zero across cell types.")
# # #   Z <- Z[keep_gene_sig, , drop=FALSE]
# # # }
# # # sig_df <- data.frame(gene = rownames(Z), Z, check.names = FALSE)
# # # sig_path <- file.path(OUT_DIR, "signature_matrix.tsv")
# # # data.table::fwrite(sig_df, sig_path, sep="\t")
# # # msg("Signature matrix saved: ", sig_path)

# # # ## ===================== BISQUE (>=2 subjects) or NNLS (1) ====================
# # # n_subjects <- length(unique(Biobase::pData(sc.eset)$SubjectName))
# # # msg("Single-cell unique subjects detected: ", n_subjects)

# # # if (n_subjects >= 2) {
# # #   genes_use <- intersect(rownames(sc.eset), rownames(bulk.mat))
# # #   bulk_use  <- ensure_double_matrix(bulk.mat[genes_use, , drop=FALSE])
# # #   bulk_use  <- drop_zero_expr_samples(bulk_use, label="bulk (pre-Bisque)")
# # #   bulk.eset <- Biobase::ExpressionSet(assayData = bulk_use)
# # #   msg("Running BisqueRNA::ReferenceBasedDecomposition ...")
# # #   res <- BisqueRNA::ReferenceBasedDecomposition(
# # #     bulk.eset     = bulk.eset,
# # #     sc.eset       = sc.eset,
# # #     markers       = NULL,
# # #     cell.types    = "cellType",
# # #     subject.names = "SubjectName",
# # #     use.overlap   = FALSE,
# # #     verbose       = TRUE,
# # #     old.cpm       = TRUE
# # #   )
# # #   bulk_props <- ensure_double_matrix(res$bulk.props)
# # # } else {
# # #   msg("Fallback: only ONE subject in scRNA-seq; running NNLS on signature matrix.")
# # #   common_genes <- intersect(rownames(Z), rownames(bulk.mat))
# # #   if (length(common_genes) < WARN_MIN_OVERLAP)
# # #     msg("WARNING: low overlap gene count for NNLS (", length(common_genes), ").")
# # #   A <- ensure_double_matrix(Z[common_genes, , drop=FALSE])        # genes × CT
# # #   B <- ensure_double_matrix(bulk.mat[common_genes, , drop=FALSE]) # genes × samples
# # #   A[!is.finite(A)] <- 0; B[!is.finite(B)] <- 0
# # #   keep_gene <- rowSums(A) > 0
# # #   if (any(!keep_gene)) { msg("NNLS: dropping ", sum(!keep_gene), " genes with zero signature.");
# # #     A <- A[keep_gene, , drop=FALSE]; B <- B[keep_gene, , drop=FALSE] }
# # #   keep_ct <- colSums(A) > 0
# # #   if (any(!keep_ct)) { msg("NNLS: dropping ", sum(!keep_ct), " CT with zero signature.");
# # #     A <- A[, keep_ct, drop=FALSE] }
# # #   if (ncol(A) == 0) stop("NNLS: no non-zero cell types remain after filtering.")
# # #   P_nnls <- matrix(NA_real_, nrow=ncol(A), ncol=ncol(B),
# # #                    dimnames=list(colnames(A), colnames(B)))
# # #   for (j in seq_len(ncol(B))) {
# # #     fit <- nnls::nnls(A, B[, j]); p <- as.numeric(fit$x); p[!is.finite(p)] <- 0
# # #     s <- sum(p); if (is.finite(s) && s > 0) p <- p / s
# # #     P_nnls[, j] <- p
# # #   }
# # #   bulk_props <- ensure_double_matrix(P_nnls)
# # #   res <- list(bulk.props = bulk_props, genes.used = rownames(A))
# # # }

# # # ## =============================== OUTPUTS ====================================
# # # prop_sums  <- colSums(bulk_props)
# # # bad_sum    <- which(abs(prop_sums - 1) > MAX_PROP_DEVIATION)
# # # if (length(bad_sum))
# # #   msg("WARNING: |sum(props)-1| >", MAX_PROP_DEVIATION, " for samples: ",
# # #       paste(colnames(bulk_props)[bad_sum], collapse=", "))

# # # # TSV (canonical)
# # # data.table::fwrite(
# # #   data.frame(cell_type = rownames(bulk_props), bulk_props, check.names = FALSE),
# # #   file.path(OUT_DIR, "bisque_bulk_proportions.tsv"), sep = "\t"
# # # )
# # # # CSV (for older Python code that expects it)
# # # utils::write.csv(
# # #   data.frame(bulk_props, check.names = FALSE),
# # #   file = file.path(OUT_DIR, "Bisque_proportions.csv"),
# # #   row.names = TRUE
# # # )

# # # if (!is.null(res$sc.props)) {
# # #   data.table::fwrite(
# # #     data.frame(cell_type = rownames(res$sc.props), res$sc.props, check.names = FALSE),
# # #     file.path(OUT_DIR, "bisque_sc_proportions.tsv"), sep = "\t"
# # #   )
# # # }

# # # if (!is.null(res$genes.used)) {
# # #   writeLines(res$genes.used, file.path(OUT_DIR, "bisque_genes_used.txt"))
# # # } else {
# # #   writeLines(rownames(bulk.mat), file.path(OUT_DIR, "bisque_genes_used.txt"))
# # # }

# # # # Bar of sums (open an explicit device)
# # # png(filename = file.path(OUT_DIR, "prop_sums.png"), width = 1200, height = 400)
# # # par(mar=c(6,4,2,1))
# # # barplot(prop_sums, las = 2, main = "Sum of estimated proportions per sample")
# # # abline(h = 1, col = "red", lty = 2)
# # # safe_dev_off()

# # # ## ======================= POST-HOC QC + VISUALS ==============================
# # # PROP_FILE <- file.path(OUT_DIR, "bisque_bulk_proportions.tsv")
# # # SIG_FILE  <- file.path(OUT_DIR, "signature_matrix.tsv")
# # # stopifnot(file.exists(PROP_FILE), file.exists(SIG_FILE))

# # # props_dt <- data.table::fread(PROP_FILE)
# # # stopifnot("cell_type" %in% names(props_dt))
# # # P_raw <- as.matrix(props_dt[, -1, with=FALSE]); rownames(P_raw) <- props_dt$cell_type
# # # P_raw <- ensure_double_matrix(P_raw)
# # # samples <- colnames(P_raw)

# # # sig_dt <- data.table::fread(SIG_FILE)
# # # gcol <- if ("gene" %in% names(sig_dt)) "gene" else names(sig_dt)[1]
# # # genes <- sig_dt[[gcol]]; sig_dt[[gcol]] <- NULL
# # # Z_raw <- ensure_double_matrix(as.matrix(sig_dt)); rownames(Z_raw) <- genes

# # # # Harmonize CT labels
# # # colnames(Z_raw) <- harmonize_labels(colnames(Z_raw))
# # # rownames(P_raw) <- harmonize_labels(rownames(P_raw))
# # # if (any(duplicated(colnames(Z_raw))))  stop("Duplicate CTs in signature (Z).")
# # # if (any(duplicated(rownames(P_raw))))  stop("Duplicate CTs in proportions (P).")
# # # CT <- intersect(colnames(Z_raw), rownames(P_raw))
# # # if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d) after harmonization.", length(CT)))
# # # Z_use <- ensure_double_matrix(Z_raw[, CT, drop=FALSE])   # genes x CT
# # # P_use <- ensure_double_matrix(P_raw[CT,  , drop=FALSE])  # CT x samples

# # # # Tidy for plots
# # # props_dt_use <- data.frame(cell_type = rownames(P_use), P_use, check.names = FALSE)
# # # props_long <- data.table::melt(data.table::as.data.table(props_dt_use), id.vars="cell_type",
# # #                                variable.name="sample", value.name="prop")
# # # wide <- data.table::dcast(props_long, sample ~ cell_type, value.var="prop")
# # # prop_mat <- as.matrix(wide[, -1, with=FALSE]); rownames(prop_mat) <- wide$sample
# # # prop_mat <- ensure_double_matrix(prop_mat)

# # # has_sample_meta <- !is.null(SAMPLE_META_FILE) && file.exists(SAMPLE_META_FILE)
# # # if (has_sample_meta) meta <- data.table::fread(SAMPLE_META_FILE)

# # # have_bulk <- file.exists(BULK_COUNTS_FILE)
# # # if (have_bulk) {
# # #   bulk_dt <- data.table::fread(BULK_COUNTS_FILE)
# # #   gcol2 <- intersect(c("gene","symbol","Gene","Gene_Symbol","ID","Name","GeneID","genes"), names(bulk_dt))[1]
# # #   if (is.na(gcol2)) gcol2 <- names(bulk_dt)[1]
# # #   bgenes <- bulk_dt[[gcol2]]; bulk_dt[[gcol2]] <- NULL
# # #   X_bulk <- ensure_double_matrix(as.matrix(bulk_dt)); rownames(X_bulk) <- as.character(bgenes)
# # #   X_cpm  <- cpm_mat(X_bulk)
# # # }

# # # have_sc_props <- !is.null(res$sc.props)
# # # if (have_sc_props) {
# # #   Ct_raw <- ensure_double_matrix(as.matrix(res$sc.props))
# # #   rownames(Ct_raw) <- harmonize_labels(rownames(res$sc.props))
# # # }

# # # # --- QC: sums table + hist
# # # prop_sums <- colSums(P_use, na.rm=TRUE)
# # # qc_df <- data.frame(sample=names(prop_sums), sum_props=as.numeric(prop_sums),
# # #                     abs_dev=abs(as.numeric(prop_sums)-1))
# # # data.table::fwrite(qc_df, file.path(TAB_DIR,"qc_prop_sums.tsv"), sep="\t")

# # # g_hist <- ggplot2::ggplot(qc_df, ggplot2::aes(sum_props)) +
# # #   ggplot2::geom_histogram(bins=30, alpha=.9) +
# # #   ggplot2::geom_vline(xintercept=1, linetype=2, color="red") +
# # #   theme_pub() + ggplot2::labs(title="Distribution of proportion sums", x="Sum of proportions", y="Count")
# # # ggplot2::ggsave(file.path(PLOT_DIR,"02_qc_sum_hist.png"), g_hist, width=6.2, height=4.5, dpi=200)

# # # # --- Stacked composition
# # # hc <- hclust(dist(prop_mat), method = "complete")
# # # sample_order <- rownames(prop_mat)[hc$order]
# # # props_long$sample <- factor(props_long$sample, levels = sample_order)
# # # gg1 <- ggplot2::ggplot(props_long, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
# # #   ggplot2::geom_bar(stat = "identity", width = 0.95) +
# # #   ggplot2::labs(title = "Cell-type composition per sample (cluster-ordered)", x = "Sample", y = "Proportion") +
# # #   ggplot2::theme_bw() + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=90, vjust=0.5, hjust=1))
# # # ggplot2::ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), gg1, width = 16, height = 7, dpi = 200)

# # # # --- Heatmap of proportions
# # # pheatmap::pheatmap(prop_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
# # #                    filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
# # #                    main="Proportions heatmap", width=10, height=13)

# # # # --- Violin/box of CT distributions
# # # g_boxv <- ggplot2::ggplot(props_long, ggplot2::aes(cell_type, prop, fill=cell_type)) +
# # #   ggplot2::geom_violin(trim=FALSE, alpha=.85) + ggplot2::geom_boxplot(width=.12, outlier.size=.4) +
# # #   viridis::scale_fill_viridis(discrete=TRUE, guide="none") + theme_pub() +
# # #   ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45, hjust=1)) +
# # #   ggplot2::labs(title="Distribution of Bisque-estimated proportions", x="Cell type", y="Proportion")
# # # ggplot2::ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), g_boxv, width=11, height=6.5, dpi=200)

# # # ## =================== OPTIONAL MARKER CHECKS & EXPORTS =======================
# # # eps <- 1e-8
# # # other_mean <- function(mat, ct) { if (ncol(mat)==1) return(rep(0, nrow(mat))); rowMeans(mat[, setdiff(colnames(mat), ct), drop=FALSE], na.rm=TRUE) }
# # # spec_list <- lapply(colnames(Z_use), function(ct){
# # #   spec <- log2((Z_use[, ct] + eps) / (other_mean(Z_use, ct) + eps))
# # #   data.frame(gene = rownames(Z_use), cell_type = ct, specificity = spec, expr_ct = Z_use[, ct], stringsAsFactors = FALSE)
# # # })
# # # spec_df <- data.table::rbindlist(spec_list)
# # # markers_sel <- spec_df[order(cell_type, -specificity), .SD[1:min(.N, TOP_MARKERS_PER_CT)], by = cell_type]
# # # data.table::fwrite(markers_sel, file.path(TAB_DIR, "top_markers_per_celltype.tsv"), sep="\t")

# # # if (!is.null(MARKER_FILE) && file.exists(MARKER_FILE) && have_bulk) {
# # #   mk <- data.table::fread(MARKER_FILE)
# # #   if (nrow(mk)) {
# # #     data.table::setnames(mk, old = names(mk), new = tolower(names(mk)))
# # #     gene_col <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","feature","id","geneid","name"), names(mk))[1]
# # #     ct_col   <- intersect(c("cell_type","celltype","cell_type_label","celltype_label","cluster","label","cell","ct"), names(mk))[1]
# # #     if (!is.na(gene_col) && !is.na(ct_col)) {
# # #       markers <- mk[, .(gene = get(gene_col), cell_type = harmonize_labels(get(ct_col)))]
# # #       markers <- unique(markers[!is.na(gene) & nzchar(gene) & !is.na(cell_type) & nzchar(cell_type)])
# # #       bulk_dt0 <- data.table::fread(BULK_COUNTS_FILE)
# # #       gcol3 <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","id","name","geneid","genes"), names(bulk_dt0))[1]
# # #       if (is.na(gcol3)) gcol3 <- names(bulk_dt0)[1]
# # #       bgenes <- bulk_dt0[[gcol3]]; bulk_dt0[[gcol3]] <- NULL
# # #       bulk_mat <- ensure_double_matrix(as.matrix(bulk_dt0)); rownames(bulk_mat) <- as.character(bgenes)
# # #       bulk_cpm <- cpm_mat(bulk_mat)
# # #       ct_ok <- intersect(unique(markers$cell_type), rownames(P_use))
# # #       ctab <- lapply(ct_ok, function(ct) {
# # #         gs <- intersect(markers[cell_type == ct, gene], rownames(bulk_cpm))
# # #         if (!length(gs)) return(NULL)
# # #         score <- colMeans(bulk_cpm[gs, , drop = FALSE])
# # #         common <- intersect(names(score), colnames(P_use))
# # #         if (length(common) < 3) return(NULL)
# # #         rho <- suppressWarnings(cor(score[common], P_use[ct, common], method = "spearman"))
# # #         data.frame(cell_type = ct, n_genes = length(gs), n_samples = length(common), spearman_rho = rho)
# # #       })
# # #       ctab <- data.table::rbindlist(ctab, fill = TRUE)
# # #       if (!is.null(ctab) && nrow(ctab)) data.table::fwrite(ctab, file.path(TAB_DIR, "marker_prop_spearman.tsv"), sep = "\t")
# # #     }
# # #   }
# # # }
# # # # ====================== S3 ADD-ONS: OVERVIEW + SAMPLE CARDS ===================
# # # # (Safe: only runs if objects exist; otherwise writes placeholders.)
# # # safe_ggsave <- function(path, plot=NULL, width=8, height=5, dpi=200, title_when_empty=NULL){
# # #   if (!is.null(plot)) {
# # #     ggplot2::ggsave(path, plot, width=width, height=height, dpi=dpi)
# # #   } else {
# # #     png(path, width=width*100, height=height*100)
# # #     par(mar=c(2,2,2,2)); plot.new()
# # #     ttl <- if (is.null(title_when_empty)) basename(path) else title_when_empty
# # #     title(ttl); mtext("No data to plot", side=3, line=-1, col="gray40")
# # #     dev.off()
# # #   }
# # # }

# # # dir.create(file.path(PLOT_DIR, "S3_sample_cards"), showWarnings = FALSE, recursive = TRUE)

# # # # ---------- S3 overview: overall composition pie ----------
# # # overall_pie <- NULL
# # # if (exists("props_long") && nrow(props_long)) {
# # #   dfpie <- props_long[, .(prop = mean(prop, na.rm=TRUE)), by=cell_type]
# # #   dfpie <- dfpie[prop > 0]
# # #   if (nrow(dfpie)) {
# # #     overall_pie <- ggplot2::ggplot(dfpie, ggplot2::aes(x="", y=prop, fill=cell_type)) +
# # #       ggplot2::geom_bar(stat="identity", width=0.9) +
# # #       ggplot2::coord_polar(theta="y") +
# # #       viridis::scale_fill_viridis(discrete=TRUE, guide=NULL) +
# # #       ggplot2::labs(title="Overall mean composition (across samples)", y=NULL, x=NULL) +
# # #       ggplot2::theme_void() +
# # #       ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold"))
# # #   }
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S3_overview_composition_pie.png"), overall_pie, 7, 7,
# # #             title_when_empty="Overall mean composition (pie)")

# # # # ---------- S3 overview: CT prevalence across samples ----------
# # # ct_prev_plot <- NULL
# # # if (exists("P_use") && ncol(P_use) > 0) {
# # #   prev <- data.table::as.data.table(P_use)
# # #   prev[, cell_type := rownames(P_use)]
# # #   prev_long <- data.table::melt(prev, id.vars="cell_type", variable.name="sample", value.name="prop")
# # #   prev_ct <- prev_long[, .(prevalence = sum(prop > 0, na.rm=TRUE)), by=cell_type]
# # #   if (nrow(prev_ct)) {
# # #     ct_prev_plot <- ggplot2::ggplot(prev_ct, ggplot2::aes(reorder(cell_type, prevalence), prevalence)) +
# # #       ggplot2::geom_col() + ggplot2::coord_flip() +
# # #       ggplot2::labs(title="CT prevalence (#samples with prop>0)", x="Cell type", y="#Samples") +
# # #       ggplot2::theme_bw()
# # #   }
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S3_celltype_prevalence.png"), ct_prev_plot, 8, 7,
# # #             title_when_empty="CT prevalence")

# # # # ---------- S3 overview: sample × sample correlation of compositions ----------
# # # corr_plot_path <- file.path(PLOT_DIR, "S3_props_correlation_heatmap.png")
# # # if (exists("prop_mat") && nrow(prop_mat) > 1 && ncol(prop_mat) > 1) {
# # #   # rows = samples, cols = CTs
# # #   cm <- suppressWarnings(cor(t(prop_mat), method="spearman", use="pairwise.complete.obs"))
# # #   cm[!is.finite(cm)] <- 0
# # #   tryCatch({
# # #     pheatmap::pheatmap(cm, filename=corr_plot_path, main="Sample×Sample Spearman correlation (compositions)",
# # #                        cluster_rows=TRUE, cluster_cols=TRUE, width=10, height=10)
# # #   }, error=function(e){
# # #     safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty=paste("Correlation heatmap:", e$message))
# # #   })
# # # } else {
# # #   safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty="Correlation heatmap (insufficient data)")
# # # }

# # # # ---------- S3 overview: PCA of compositions ----------
# # # pca_plot <- NULL
# # # if (exists("prop_mat") && nrow(prop_mat) >= 2 && ncol(prop_mat) >= 2) {
# # #   X <- prop_mat
# # #   X <- X[, colSums(is.finite(X)) == nrow(X), drop=FALSE]
# # #   if (ncol(X) >= 2) {
# # #     pc <- tryCatch(prcomp(X, center=TRUE, scale.=TRUE), error=function(e) NULL)
# # #     if (!is.null(pc)) {
# # #       sc <- data.frame(PC1=pc$x[,1], PC2=pc$x[,2], sample=rownames(pc$x))
# # #       pca_plot <- ggplot2::ggplot(sc, ggplot2::aes(PC1, PC2, label=sample)) +
# # #         ggplot2::geom_point() + ggrepel::geom_text_repel(size=3, max.overlaps=40) +
# # #         ggplot2::theme_bw() + ggplot2::labs(title="PCA of sample compositions")
# # #     }
# # #   }
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S3_props_PCA.png"), pca_plot, 8, 6,
# # #             title_when_empty="PCA of compositions (insufficient data)")

# # # # ---------- S1 sanity: mean signature per CT ----------
# # # sig_mean_plot <- NULL
# # # if (exists("Z_use") && ncol(Z_use) >= 1) {
# # #   means <- colMeans(Z_use, na.rm=TRUE)
# # #   dfm <- data.frame(cell_type=names(means), mean_expr=as.numeric(means))
# # #   sig_mean_plot <- ggplot2::ggplot(dfm, ggplot2::aes(reorder(cell_type, mean_expr), mean_expr)) +
# # #     ggplot2::geom_col() + ggplot2::coord_flip() +
# # #     ggplot2::theme_bw() + ggplot2::labs(title="Mean signature per CT (CPM)", x="Cell type", y="Mean CPM")
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S1_signature_ct_means.png"), sig_mean_plot, 7, 6,
# # #             title_when_empty="Mean signature per CT")

# # # # ---------- S3: per-sample QC TSV + SAMPLE CARDS (PNG) ----------
# # # qc_out <- file.path(TAB_DIR, "S3_sample_qc.tsv")
# # # qc_df2 <- data.frame()
# # # if (exists("P_use") && ncol(P_use) > 0) {
# # #   prop_sums_local <- colSums(P_use, na.rm=TRUE)
# # #   abs_dev <- abs(prop_sums_local - 1)
# # #   qc_df2 <- data.frame(sample = names(prop_sums_local),
# # #                        sum_props = as.numeric(prop_sums_local),
# # #                        abs_dev = as.numeric(abs_dev))
# # #   data.table::fwrite(qc_df2, qc_out, sep="\t")
# # # } else {
# # #   data.table::fwrite(data.frame(), qc_out, sep="\t")
# # # }

# # # # Helper: top genes plot for a sample (from X_cpm, if present)
# # # top_genes_plot <- function(sample_id, topN=10){
# # #   if (!exists("X_cpm") || !is.matrix(X_cpm) || !(sample_id %in% colnames(X_cpm))) return(NULL)
# # #   v <- X_cpm[, sample_id]; if (all(!is.finite(v))) return(NULL)
# # #   ord <- order(v, decreasing=TRUE); k <- min(topN, length(ord))
# # #   if (k < 1) return(NULL)
# # #   sel <- ord[seq_len(k)]
# # #   df <- data.frame(gene=names(v)[sel], CPM=as.numeric(v[sel]))
# # #   ggplot2::ggplot(df, ggplot2::aes(reorder(gene, CPM), CPM)) +
# # #     ggplot2::geom_col() + ggplot2::coord_flip() +
# # #     ggplot2::theme_bw() + ggplot2::labs(title=paste0("Top ", k, " genes in ", sample_id), x=NULL, y="CPM")
# # # }

# # # # Helper: per-sample composition stacked bar (single sample)
# # # sample_comp_plot <- function(sample_id){
# # #   if (!exists("props_long") || !nrow(props_long)) return(NULL)
# # #   dl <- props_long[sample == sample_id]
# # #   if (!nrow(dl)) return(NULL)
# # #   ggplot2::ggplot(dl, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
# # #     ggplot2::geom_bar(stat="identity", width=0.8) +
# # #     viridis::scale_fill_viridis(discrete=TRUE, guide="none") +
# # #     ggplot2::theme_bw() + ggplot2::labs(title=paste0("Composition – ", sample_id), x=NULL, y="Proportion")
# # # }

# # # # Helper: sample QC tile
# # # sample_qc_tile <- function(sample_id){
# # #   sp <- tryCatch(qc_df2[qc_df2$sample == sample_id, , drop=FALSE], error=function(e) NULL)
# # #   if (is.null(sp) || !nrow(sp)) return(NULL)
# # #   lbl <- sprintf("sum(props)=%.3f\n|sum-1|=%.3f", sp$sum_props[1], sp$abs_dev[1])
# # #   ggplot2::ggplot(data.frame(x=1,y=1,label=lbl), ggplot2::aes(x,y,label=label)) +
# # #     ggplot2::geom_text(size=5) + ggplot2::theme_void() +
# # #     ggplot2::labs(title="QC")
# # # }

# # # # Render sample cards
# # # if (exists("prop_mat") && nrow(prop_mat) >= 1) {
# # #   smpls <- rownames(prop_mat)
# # #   for (s in smpls) {
# # #     p1 <- sample_comp_plot(s)
# # #     p2 <- top_genes_plot(s, topN = max(5, min(10, if (exists("TOP_GENES_PER_SAMPLE")) TOP_GENES_PER_SAMPLE else 10)))
# # #     p3 <- sample_qc_tile(s)
# # #     # Arrange with patchwork if at least one plot exists
# # #     if (!is.null(p1) || !is.null(p2) || !is.null(p3)) {
# # #       card <- NULL
# # #       if (!is.null(p1) && !is.null(p2) && !is.null(p3)) card <- p1 + p2 + p3 + patchwork::plot_layout(ncol=3, widths=c(1.2,1,0.6))
# # #       else if (!is.null(p1) && !is.null(p2)) card <- p1 + p2 + patchwork::plot_layout(ncol=2, widths=c(1.2,1))
# # #       else if (!is.null(p1)) card <- p1
# # #       else if (!is.null(p2)) card <- p2
# # #       else card <- p3
# # #       outp <- file.path(PLOT_DIR, "S3_sample_cards", paste0(s, ".png"))
# # #       safe_ggsave(outp, card, width=15, height=5, dpi=200, title_when_empty=paste("Sample card:", s))
# # #     } else {
# # #       safe_ggsave(file.path(PLOT_DIR, "S3_sample_cards", paste0(s, ".png")), NULL, 12, 4,
# # #                   title_when_empty=paste("Sample card:", s))
# # #     }
# # #   }
# # # } else {
# # #   # No compositions: still produce a note
# # #   safe_ggsave(file.path(PLOT_DIR, "S3_sample_cards", "no_samples.png"), NULL, 10, 3,
# # #               title_when_empty="No sample compositions available for cards")
# # # }
# # # # ==================== END S3 ADD-ONS =========================================

# # # msg("DONE. Outputs written to: ", OUT_DIR)
# # # msg("Tables → ", TAB_DIR)
# # # msg("Plots  → ", PLOT_DIR)


# # # #----------------------------------------------------------------------------------------------------------###

# # # #!/usr/bin/env Rscript

# # # options(stringsAsFactors = FALSE)
# # # ### NEW: deterministic randomness for any sampling/plots
# # # set.seed(42L)

# # # ## ============================ OPTIONS =======================================
# # # # Environment variables set by Python
# # # get_bool  <- function(x, default = FALSE) {
# # #   v <- toupper(Sys.getenv(x, ifelse(default, "TRUE", "FALSE"))); v %in% c("1","T","TRUE","YES","Y")
# # # }
# # # get_int   <- function(x, default = NA_integer_) {
# # #   v <- suppressWarnings(as.integer(Sys.getenv(x, ""))); if (is.na(v)) default else v
# # # }
# # # get_num   <- function(x, default = NA_real_) {
# # #   v <- suppressWarnings(as.numeric(Sys.getenv(x, ""))); if (is.na(v)) default else v
# # # }
# # # get_str   <- function(x, default = "") {
# # #   v <- Sys.getenv(x, ""); if (!nzchar(v)) default else v
# # # }

# # # REF_DIR_PY            <- get_str("REF_DIR_PY", "")
# # # OUT_DIR_PY            <- get_str("OUT_DIR_PY", "")
# # # DROP_MITO_PY          <- get_bool("DROP_MITO_PY", FALSE)
# # # ALLOW_DUP_PY          <- get_bool("ALLOW_DUP_PY", FALSE)
# # # WARN_MIN_OVERLAP_PY   <- get_int ("WARN_MIN_OVERLAP_PY", 100L)
# # # MAX_PROP_DEV_PY       <- get_num ("MAX_PROP_DEV_PY", 0.2)
# # # TOP_MARKERS_PER_CT_PY <- get_int ("TOP_MARKERS_PER_CT_PY", 10L)
# # # TOP_GENES_PER_SAMPLE_PY <- get_int("TOP_GENES_PER_SAMPLE_PY", 50L)
# # # AUTO_INSTALL_PY       <- get_bool("AUTO_INSTALL_PY", TRUE)

# # # if (!nzchar(REF_DIR_PY)) stop("Environment variable REF_DIR_PY not set; Python must pass it to R.")

# # # ## ============================ UTILITIES =====================================
# # # msg <- function(...) cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] "), ..., "\n")

# # # safe_dev_off <- function() {
# # #   if (names(dev.cur()) != "null device") try(dev.off(), silent = TRUE)
# # # }

# # # ensure_double_matrix <- function(m) {
# # #   if (isS4(m)) m <- as.matrix(m)
# # #   if (!is.matrix(m)) m <- as.matrix(m)
# # #   storage.mode(m) <- "double"; m
# # # }

# # # is_all_numeric_vec <- function(x) all(grepl("^[0-9]+$", as.character(x)))
# # # norm_bc <- function(x){ x <- trimws(as.character(x)); x <- gsub("^X(?=\\d)","",x,perl=TRUE); x <- gsub("\\.", "-", x); x }
# # # drop_mito <- function(genes) genes[!grepl("^MT-", toupper(genes))]
# # # zero_var_rows <- function(m) { v <- apply(m, 1, function(x) var(x, na.rm=TRUE)); names(which(is.na(v) | v == 0)) }

# # # collapse_dups <- function(m) {
# # #   m <- ensure_double_matrix(m)
# # #   if (is.null(rownames(m))) stop("Matrix lacks rownames.")
# # #   if (!any(duplicated(rownames(m)))) return(m)
# # #   msg("Duplicate gene IDs detected; collapsing by sum.")
# # #   ux <- unique(rownames(m)); idx <- match(rownames(m), ux)
# # #   sp <- Matrix::sparseMatrix(i = idx, j = seq_len(nrow(m)), x = 1,
# # #                              dims = c(length(ux), nrow(m)))
# # #   out <- ensure_double_matrix(as.matrix(sp %*% m))
# # #   rownames(out) <- ux; colnames(out) <- colnames(m)
# # #   out
# # # }

# # # drop_zero_expr_samples <- function(m, label="bulk") {
# # #   m <- ensure_double_matrix(m)
# # #   cs <- colSums(m, na.rm=TRUE); zc <- which(!is.finite(cs) | cs <= 0)
# # #   if (length(zc)) {
# # #     msg("Dropping ", length(zc), " ", label, " sample(s) with zero/NA expr: ",
# # #         paste(colnames(m)[zc], collapse=", "))
# # #     m <- m[, -zc, drop=FALSE]
# # #   }
# # #   if (ncol(m) == 0) stop("All ", label, " samples dropped; cannot proceed.")
# # #   m
# # # }

# # # theme_pub <- function(){
# # #   ggplot2::theme_bw(base_size=12) +
# # #     ggplot2::theme(plot.title=ggplot2::element_text(face="bold"),
# # #                    panel.grid.minor=ggplot2::element_blank())
# # # }

# # # cpm_mat <- function(m){
# # #   m <- ensure_double_matrix(m)
# # #   lib <- colSums(m,na.rm=TRUE); lib[!is.finite(lib)|lib==0] <- 1
# # #   ensure_double_matrix(t(1e6*t(m)/lib))
# # # }

# # # harmonize_labels <- function(x) make.names(trimws(as.character(x)), unique = FALSE)

# # # ## =========================== PATHS & PARAMS =================================
# # # REF_DIR <- REF_DIR_PY
# # # OUT_DIR_DEFAULT <- file.path(REF_DIR, "bisque_deconv-results")
# # # OUT_DIR <- if (!is.null(OUT_DIR_PY) && nzchar(OUT_DIR_PY)) OUT_DIR_PY else OUT_DIR_DEFAULT
# # # dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# # # DROP_MITO_GENES       <- isTRUE(DROP_MITO_PY)
# # # ALLOW_DUPLICATE_GENES <- isTRUE(ALLOW_DUP_PY)
# # # WARN_MIN_OVERLAP      <- as.integer(WARN_MIN_OVERLAP_PY)
# # # MAX_PROP_DEVIATION    <- as.numeric(MAX_PROP_DEV_PY)
# # # TOP_MARKERS_PER_CT    <- as.integer(TOP_MARKERS_PER_CT_PY)
# # # TOP_GENES_PER_SAMPLE  <- as.integer(TOP_GENES_PER_SAMPLE_PY)
# # # AUTO_INSTALL          <- isTRUE(AUTO_INSTALL_PY)

# # # SC_COUNTS_FILE   <- file.path(REF_DIR, "sc_counts.tsv")
# # # SC_META_FILE     <- file.path(REF_DIR, "sc_metadata.tsv")
# # # BULK_COUNTS_FILE <- file.path(REF_DIR, "bulk_counts_overlap.tsv")

# # # SAMPLE_META_FILE <- file.path(REF_DIR, "sample_metadata_patientAKI.tsv")  # optional
# # # MARKER_FILE      <- file.path(REF_DIR, "marker_gene_table.tsv")           # optional

# # # PLOT_DIR <- file.path(OUT_DIR, "deconv_visual_reports"); dir.create(PLOT_DIR, FALSE, TRUE)
# # # TAB_DIR  <- file.path(OUT_DIR, "QC & Performance Metrics"); dir.create(TAB_DIR,  FALSE, TRUE)

# # # ## ============================ PACKAGES ======================================
# # # cran_pkgs <- c("data.table","Matrix","tools","nnls","ggplot2","viridis",
# # #                "pheatmap","corrplot","dplyr","tidyr","patchwork","ggrepel")
# # # bio_pkgs  <- c("Biobase","BisqueRNA")

# # # need_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
# # # need_bio  <- bio_pkgs[!vapply(bio_pkgs,  requireNamespace, logical(1), quietly = TRUE)]

# # # if (AUTO_INSTALL) {
# # #   if (length(need_cran)) {
# # #     msg("Installing CRAN packages: ", paste(need_cran, collapse=", "))
# # #     install.packages(need_cran, repos = "https://cloud.r-project.org")
# # #   }
# # #   if (!requireNamespace("BiocManager", quietly = TRUE)) {
# # #     install.packages("BiocManager", repos = "https://cloud.r-project.org")
# # #   }
# # #   if (length(need_bio)) {
# # #     msg("Installing Bioconductor packages: ", paste(need_bio, collapse=", "))
# # #     BiocManager::install(need_bio, update = FALSE, ask = FALSE)
# # #   }
# # # } else if (length(need_cran) || length(need_bio)) {
# # #   stop("Missing R packages: ", paste(c(need_cran, need_bio), collapse=", "),
# # #        "\nRun again with auto_install=TRUE or install manually.")
# # # }

# # # suppressPackageStartupMessages({
# # #   lapply(c(cran_pkgs, bio_pkgs), library, character.only = TRUE)
# # # })
# # # capture.output(sessionInfo(), file = file.path(OUT_DIR, "R_sessionInfo.txt"))

# # # ## ======================= LOAD SINGLE-CELL DATA ==============================
# # # stopifnot(file.exists(SC_COUNTS_FILE), file.exists(SC_META_FILE))
# # # msg("Loading single-cell counts: ", SC_COUNTS_FILE)
# # # cnt_full <- data.table::fread(SC_COUNTS_FILE, sep="\t", header=TRUE, data.table=FALSE)

# # # # First column must be 'gene'; if not, treat the first as gene column anyway.
# # # genes <- as.character(cnt_full[[1]]); cnt_full[[1]] <- NULL
# # # m_sc <- ensure_double_matrix(as.matrix(cnt_full))
# # # rownames(m_sc) <- genes
# # # counts_bc_raw <- colnames(cnt_full)

# # # msg("Loading metadata: ", SC_META_FILE)
# # # md <- data.table::fread(SC_META_FILE, sep="\t", header=TRUE, data.table=FALSE)
# # # if (!"cell_type" %in% names(md))     md$cell_type     <- NA_character_
# # # if (!"individual_id" %in% names(md)) md$individual_id <- "ONE_SUBJECT"
# # # if (!"cell_barcode" %in% names(md))  md$cell_barcode  <- NA_character_

# # # counts_bc <- norm_bc(counts_bc_raw); meta_bc <- norm_bc(md$cell_barcode)
# # # if (nrow(md) != length(counts_bc)) {
# # #   stop(sprintf("Metadata rows (%d) != number of cells in counts (%d). Re-export so sizes match.",
# # #                nrow(md), length(counts_bc)))
# # # }
# # # if (length(na.omit(unique(meta_bc))) <= 2 || is_all_numeric_vec(na.omit(meta_bc))) {
# # #   md$cell_barcode <- counts_bc; meta_bc <- counts_bc
# # # }
# # # if (is_all_numeric_vec(counts_bc) && is_all_numeric_vec(meta_bc)) {
# # #   msg("Counts & metadata both numeric; generating synthetic barcodes (in-memory).")
# # #   syn <- sprintf("C%06d", seq_len(length(counts_bc)))
# # #   colnames(m_sc) <- syn; md$cell_barcode <- syn
# # #   counts_bc <- syn; meta_bc <- syn
# # # } else {
# # #   colnames(m_sc) <- counts_bc
# # # }

# # # # Align (transpose rescue + synthetic fallback)
# # # ov <- intersect(colnames(m_sc), md$cell_barcode)
# # # if (!length(ov) && any(rownames(m_sc) %in% md$cell_barcode)) {
# # #   msg("Counts appear transposed; transposing ...")
# # #   m_sc <- ensure_double_matrix(t(m_sc))
# # #   ov <- intersect(colnames(m_sc), md$cell_barcode)
# # # }
# # # if (!length(ov)) {
# # #   syn <- sprintf("C%06d", seq_len(ncol(m_sc)))
# # #   colnames(m_sc) <- syn; md$cell_barcode <- syn; ov <- syn
# # # }
# # # m_sc <- m_sc[, ov, drop=FALSE]
# # # md <- md[match(ov, md$cell_barcode), , drop=FALSE]
# # # rownames(md) <- md$cell_barcode

# # # # Build ExpressionSet for sc
# # # pdat <- new("AnnotatedDataFrame", data = data.frame(
# # #   cellType    = md$cell_type,
# # #   SubjectName = md$individual_id,
# # #   row.names   = rownames(md),
# # #   check.names = FALSE
# # # ))
# # # sc.eset <- Biobase::ExpressionSet(assayData = ensure_double_matrix(m_sc), phenoData = pdat)

# # # ## ============================== LOAD BULK ===================================
# # # stopifnot(file.exists(BULK_COUNTS_FILE))
# # # read_bulk <- function(path) {
# # #   dt <- data.table::fread(path, sep="\t")
# # #   cand <- c("gene","genes","gene_id","geneid","ensembl","ensembl_id","symbol",
# # #             "gene_symbol","hgnc","Gene","GeneID","Gene_Symbol","Gene.Name","ID","Name")
# # #   gcol <- intersect(cand, names(dt))[1]
# # #   if (is.na(gcol)) {
# # #     if (!is.numeric(dt[[1]]) && length(unique(dt[[1]])) == nrow(dt)) gcol <- names(dt)[1]
# # #     else stop("Cannot detect gene column in bulk file.")
# # #   }
# # #   genes <- as.character(dt[[gcol]]); dt[[gcol]] <- NULL
# # #   m <- ensure_double_matrix(as.matrix(dt))
# # #   rownames(m) <- genes
# # #   msg("Bulk matrix dims (genes x samples): ", nrow(m), " x ", ncol(m))

# # #   ### NEW: Auto-transpose bulk if samples look like rows
# # #   if (ncol(m) < nrow(m) && all(grepl("^[[:alnum:]_.-]+$", rownames(m)))) {
# # #     gene_like <- sum(nchar(rownames(m)) <= 20) / nrow(m)
# # #     if (gene_like < 0.3) { # rownames look like sample IDs instead of genes
# # #       msg("Bulk appears transposed; transposing ...")
# # #       m <- ensure_double_matrix(t(m))
# # #       # best-effort rowname restore (fallback to original column names in file)
# # #       rownames(m) <- colnames(dt)
# # #       msg("New bulk dims (genes x samples): ", nrow(m), " x ", ncol(m))
# # #     }
# # #   }

# # #   m
# # # }
# # # bulk.mat <- read_bulk(BULK_COUNTS_FILE)

# # # ## ============================ SANITY + FILTERS ==============================
# # # stopifnot(ncol(Biobase::exprs(sc.eset)) == nrow(Biobase::pData(sc.eset)))

# # # if (any(Biobase::exprs(sc.eset) < 0, na.rm=TRUE)) stop("Negative values in single-cell counts.")
# # # if (any(bulk.mat < 0, na.rm=TRUE))                stop("Negative values in bulk counts.")

# # # if (ALLOW_DUPLICATE_GENES) {
# # #   msg("Collapsing duplicate genes (safe rebuild)...")
# # #   expr_new <- collapse_dups(Biobase::exprs(sc.eset))
# # #   sc.eset <- Biobase::ExpressionSet(
# # #     assayData = expr_new,
# # #     phenoData = Biobase::phenoData(sc.eset)
# # #   )
# # #   bulk.mat <- collapse_dups(bulk.mat)
# # # }

# # # if (DROP_MITO_GENES) {
# # #   keep_sc   <- drop_mito(rownames(sc.eset));    sc.eset  <- sc.eset[keep_sc, ]
# # #   keep_bulk <- drop_mito(rownames(bulk.mat));   bulk.mat <- bulk.mat[keep_bulk, , drop=FALSE]
# # # }

# # # zv <- zero_var_rows(Biobase::exprs(sc.eset))
# # # if (length(zv)) {
# # #   sc.eset <- sc.eset[setdiff(rownames(Biobase::exprs(sc.eset)), zv), ]
# # #   msg("Removed ", length(zv), " zero-variance genes from sc.")
# # # }

# # # bulk_zero_rows <- which(rowSums(bulk.mat == 0, na.rm=TRUE) == ncol(bulk.mat))
# # # if (length(bulk_zero_rows)) {
# # #   bulk.mat <- bulk.mat[-bulk_zero_rows, , drop=FALSE]
# # #   msg("Removed ", length(bulk_zero_rows), " unexpressed genes from bulk (rows).")
# # # }

# # # # Ensure unique & ordered gene set
# # # gene_set <- intersect(rownames(Biobase::exprs(sc.eset)), rownames(bulk.mat))
# # # gene_set <- unique(gene_set)

# # # if (length(gene_set) < WARN_MIN_OVERLAP) {
# # #   msg("WARNING: low overlap gene count (", length(gene_set), "). Results may be less stable.")
# # # }
# # # if (length(gene_set) < 2L) {
# # #   stop(sprintf("Too few overlapping genes after prefilters (|genes| = %d).", length(gene_set)))
# # # }

# # # # ExpressionSet-aware subsetting
# # # sc.eset  <- sc.eset[gene_set, ]
# # # bulk.mat <- ensure_double_matrix(bulk.mat[gene_set, , drop=FALSE])

# # # # Dim sanity checks
# # # msg(sprintf("Dims after alignment: sc.eset %d genes × %d cells; bulk %d genes × %d samples",
# # #             nrow(Biobase::exprs(sc.eset)), ncol(Biobase::exprs(sc.eset)),
# # #             nrow(bulk.mat), ncol(bulk.mat)))

# # # # Drop bulk samples that are zero over the selected genes
# # # bulk.mat <- drop_zero_expr_samples(bulk.mat, label="bulk")

# # # ## ============ DONOR-LEVEL PSEUDOBULK (CPM) & SIGNATURE MATRIX ===============
# # # msg("Building donor-level pseudobulk (CPM) ...")
# # # expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))     # genes × cells
# # # ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# # # subj_vec<- as.character(Biobase::pData(sc.eset)$SubjectName)
# # # key <- paste(subj_vec, ct_vec, sep="__")
# # # idx <- split(seq_len(ncol(expr_sc)), key)
# # # PB_counts <- sapply(idx, function(cols) Matrix::rowSums(expr_sc[, cols, drop=FALSE]))
# # # PB_counts <- ensure_double_matrix(as.matrix(PB_counts))
# # # PB_cpm <- cpm_mat(PB_counts)
# # # pb_path <- file.path(OUT_DIR, "pseudobulk_donor_level.tsv")
# # # data.table::fwrite(data.frame(gene = rownames(PB_cpm), PB_cpm, check.names = FALSE),
# # #                    pb_path, sep = "\t")
# # # msg("Saved donor-level pseudobulk → ", pb_path)

# # # keep_cells <- !is.na(ct_vec) & nzchar(ct_vec)
# # # if (any(!keep_cells)) {
# # #   msg("Dropping ", sum(!keep_cells), " cells with missing/empty cell_type.")
# # #   sc.eset <- sc.eset[, keep_cells]
# # #   expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))
# # #   ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# # # }

# # # expr <- ensure_double_matrix(Biobase::exprs(sc.eset))        # genes × cells
# # # lib  <- colSums(expr); lib[lib == 0] <- 1
# # # expr_cpm <- ensure_double_matrix(t(1e6 * t(expr) / lib))
# # # idxs <- split(seq_along(ct_vec), ct_vec)
# # # Z <- sapply(idxs, function(cols) rowMeans(expr_cpm[, cols, drop=FALSE], na.rm=TRUE))
# # # Z <- ensure_double_matrix(as.matrix(Z))
# # # colnames(Z) <- make.names(colnames(Z), unique=TRUE)
# # # Z[!is.finite(Z)] <- 0
# # # keep_gene_sig <- rowSums(Z) > 0
# # # if (any(!keep_gene_sig)) {
# # #   msg("Signature: dropping ", sum(!keep_gene_sig), " genes with all-zero across cell types.")
# # #   Z <- Z[keep_gene_sig, , drop=FALSE]
# # # }
# # # sig_df <- data.frame(gene = rownames(Z), Z, check.names = FALSE)
# # # sig_path <- file.path(OUT_DIR, "signature_matrix.tsv")
# # # data.table::fwrite(sig_df, sig_path, sep="\t")
# # # msg("Signature matrix saved: ", sig_path)

# # # ## ===================== BISQUE (>=2 subjects) or NNLS (1) ====================
# # # n_subjects <- length(unique(Biobase::pData(sc.eset)$SubjectName))
# # # msg("Single-cell unique subjects detected: ", n_subjects)

# # # if (n_subjects >= 2) {
# # #   genes_use <- intersect(rownames(sc.eset), rownames(bulk.mat))
# # #   bulk_use  <- ensure_double_matrix(bulk.mat[genes_use, , drop=FALSE])
# # #   bulk_use  <- drop_zero_expr_samples(bulk_use, label="bulk (pre-Bisque)")
# # #   bulk.eset <- Biobase::ExpressionSet(assayData = bulk_use)
# # #   msg("Running BisqueRNA::ReferenceBasedDecomposition ...")
# # #   res <- BisqueRNA::ReferenceBasedDecomposition(
# # #     bulk.eset     = bulk.eset,
# # #     sc.eset       = sc.eset,
# # #     markers       = NULL,
# # #     cell.types    = "cellType",
# # #     subject.names = "SubjectName",
# # #     use.overlap   = FALSE,
# # #     verbose       = TRUE,
# # #     old.cpm       = TRUE
# # #   )
# # #   bulk_props <- ensure_double_matrix(res$bulk.props)
# # # } else {
# # #   msg("Fallback: only ONE subject in scRNA-seq; running NNLS on signature matrix.")
# # #   common_genes <- intersect(rownames(Z), rownames(bulk.mat))
# # #   if (length(common_genes) < WARN_MIN_OVERLAP)
# # #     msg("WARNING: low overlap gene count for NNLS (", length(common_genes), ").")
# # #   A <- ensure_double_matrix(Z[common_genes, , drop=FALSE])        # genes × CT
# # #   B <- ensure_double_matrix(bulk.mat[common_genes, , drop=FALSE]) # genes × samples
# # #   A[!is.finite(A)] <- 0; B[!is.finite(B)] <- 0
# # #   keep_gene <- rowSums(A) > 0
# # #   if (any(!keep_gene)) { msg("NNLS: dropping ", sum(!keep_gene), " genes with zero signature.");
# # #     A <- A[keep_gene, , drop=FALSE]; B <- B[keep_gene, , drop=FALSE] }
# # #   keep_ct <- colSums(A) > 0
# # #   if (any(!keep_ct)) { msg("NNLS: dropping ", sum(!keep_ct), " CT with zero signature.");
# # #     A <- A[, keep_ct, drop=FALSE] }
# # #   if (ncol(A) == 0) stop("NNLS: no non-zero cell types remain after filtering.")
# # #   P_nnls <- matrix(NA_real_, nrow=ncol(A), ncol=ncol(B),
# # #                    dimnames=list(colnames(A), colnames(B)))
# # #   for (j in seq_len(ncol(B))) {
# # #     fit <- nnls::nnls(A, B[, j]); p <- as.numeric(fit$x); p[!is.finite(p)] <- 0
# # #     s <- sum(p); if (is.finite(s) && s > 0) p <- p / s
# # #     P_nnls[, j] <- p
# # #   }
# # #   bulk_props <- ensure_double_matrix(P_nnls)
# # #   res <- list(bulk.props = bulk_props, genes.used = rownames(A))
# # # }

# # # ## =============================== OUTPUTS ====================================
# # # prop_sums  <- colSums(bulk_props)
# # # bad_sum    <- which(abs(prop_sums - 1) > MAX_PROP_DEVIATION)
# # # if (length(bad_sum))
# # #   msg("WARNING: |sum(props)-1| >", MAX_PROP_DEVIATION, " for samples: ",
# # #       paste(colnames(bulk_props)[bad_sum], collapse=", "))

# # # # TSV (canonical)
# # # data.table::fwrite(
# # #   data.frame(cell_type = rownames(bulk_props), bulk_props, check.names = FALSE),
# # #   file.path(OUT_DIR, "bisque_bulk_proportions.tsv"), sep = "\t"
# # # )
# # # # CSV (for older Python code that expects it)
# # # utils::write.csv(
# # #   data.frame(bulk_props, check.names = FALSE),
# # #   file = file.path(OUT_DIR, "Bisque_proportions.csv"),
# # #   row.names = TRUE
# # # )

# # # if (!is.null(res$sc.props)) {
# # #   data.table::fwrite(
# # #     data.frame(cell_type = rownames(res$sc.props), res$sc.props, check.names = FALSE),
# # #     file.path(OUT_DIR, "bisque_sc_proportions.tsv"), sep = "\t"
# # #   )
# # # }

# # # if (!is.null(res$genes.used)) {
# # #   writeLines(res$genes.used, file.path(OUT_DIR, "bisque_genes_used.txt"))
# # # } else {
# # #   writeLines(rownames(bulk.mat), file.path(OUT_DIR, "bisque_genes_used.txt"))
# # # }

# # # # Bar of sums (open an explicit device)
# # # png(filename = file.path(OUT_DIR, "prop_sums.png"), width = 1200, height = 400)
# # # par(mar=c(6,4,2,1))
# # # barplot(prop_sums, las = 2, main = "Sum of estimated proportions per sample")
# # # abline(h = 1, col = "red", lty = 2)
# # # safe_dev_off()

# # # ## ======================= POST-HOC QC + VISUALS ==============================
# # # PROP_FILE <- file.path(OUT_DIR, "bisque_bulk_proportions.tsv")
# # # SIG_FILE  <- file.path(OUT_DIR, "signature_matrix.tsv")
# # # stopifnot(file.exists(PROP_FILE), file.exists(SIG_FILE))

# # # props_dt <- data.table::fread(PROP_FILE)
# # # stopifnot("cell_type" %in% names(props_dt))
# # # P_raw <- as.matrix(props_dt[, -1, with=FALSE]); rownames(P_raw) <- props_dt$cell_type
# # # P_raw <- ensure_double_matrix(P_raw)
# # # samples <- colnames(P_raw)

# # # sig_dt <- data.table::fread(SIG_FILE)
# # # gcol <- if ("gene" %in% names(sig_dt)) "gene" else names(sig_dt)[1]
# # # genes <- sig_dt[[gcol]]; sig_dt[[gcol]] <- NULL
# # # Z_raw <- ensure_double_matrix(as.matrix(sig_dt)); rownames(Z_raw) <- genes

# # # # Harmonize CT labels
# # # colnames(Z_raw) <- harmonize_labels(colnames(Z_raw))
# # # rownames(P_raw) <- harmonize_labels(rownames(P_raw))
# # # if (any(duplicated(colnames(Z_raw))))  stop("Duplicate CTs in signature (Z).")
# # # if (any(duplicated(rownames(P_raw))))  stop("Duplicate CTs in proportions (P).")
# # # CT <- intersect(colnames(Z_raw), rownames(P_raw))
# # # if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d) after harmonization.", length(CT)))
# # # Z_use <- ensure_double_matrix(Z_raw[, CT, drop=FALSE])   # genes x CT
# # # P_use <- ensure_double_matrix(P_raw[CT,  , drop=FALSE])  # CT x samples

# # # ### NEW: Top genes per sample based on reconstructed expression Xhat = Z_use %*% P_use
# # # try({
# # #   Xhat <- ensure_double_matrix(Z_use %*% P_use)  # genes × samples
# # #   topN <- max(1L, TOP_GENES_PER_SAMPLE)
# # #   tops <- lapply(seq_len(ncol(Xhat)), function(j){
# # #     sj <- colnames(Xhat)[j]
# # #     v  <- Xhat[, j]
# # #     ord <- order(v, decreasing = TRUE)
# # #     k   <- min(topN, length(ord))
# # #     data.frame(sample = sj,
# # #                rank   = seq_len(k),
# # #                gene   = rownames(Xhat)[ord][seq_len(k)],
# # #                score  = as.numeric(v[ord][seq_len(k)]),
# # #                stringsAsFactors = FALSE)
# # #   })
# # #   tops_df <- data.table::rbindlist(tops)
# # #   data.table::fwrite(tops_df, file.path(TAB_DIR, sprintf("top_%d_genes_per_sample.tsv", topN)), sep="\t")
# # # }, silent = TRUE)

# # # # Tidy for plots
# # # props_dt_use <- data.frame(cell_type = rownames(P_use), P_use, check.names = FALSE)
# # # props_long <- data.table::melt(data.table::as.data.table(props_dt_use), id.vars="cell_type",
# # #                                variable.name="sample", value.name="prop")
# # # wide <- data.table::dcast(props_long, sample ~ cell_type, value.var="prop")
# # # prop_mat <- as.matrix(wide[, -1, with=FALSE]); rownames(prop_mat) <- wide$sample
# # # prop_mat <- ensure_double_matrix(prop_mat)

# # # has_sample_meta <- !is.null(SAMPLE_META_FILE) && file.exists(SAMPLE_META_FILE)
# # # if (has_sample_meta) meta <- data.table::fread(SAMPLE_META_FILE)

# # # ### NEW: Group means by cohort/condition if metadata present
# # # if (has_sample_meta) try({
# # #   meta_cols <- tolower(names(meta))
# # #   s_col <- intersect(c("sample","sample_id","id","name"), meta_cols)
# # #   g_col <- intersect(c("group","condition","cohort","status","arm"), meta_cols)
# # #   if (length(s_col) && length(g_col)) {
# # #     s_col <- s_col[1]; g_col <- g_col[1]
# # #     meta2 <- data.frame(sample=as.character(meta[[s_col]]),
# # #                         group =as.character(meta[[g_col]]),
# # #                         stringsAsFactors = FALSE)
# # #     K <- intersect(rownames(prop_mat), meta2$sample)
# # #     if (length(K) >= 3) {
# # #       pm <- prop_mat[K, , drop=FALSE]
# # #       meta3 <- meta2[match(K, meta2$sample), ]
# # #       means <- by(pm, meta3$group, function(M) colMeans(M, na.rm=TRUE))
# # #       means_mat <- do.call(rbind, means)
# # #       means_mat <- ensure_double_matrix(means_mat)
# # #       data.table::fwrite(
# # #         data.frame(group=rownames(means_mat), means_mat, check.names=FALSE),
# # #         file.path(TAB_DIR,"group_mean_proportions.tsv"), sep="\t"
# # #       )
# # #       pheatmap::pheatmap(
# # #         means_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
# # #         filename=file.path(PLOT_DIR,"F4_group_mean_proportions.png"),
# # #         main="Group mean cell-type proportions", width=9, height=6.5
# # #       )
# # #     }
# # #   }
# # # }, silent = TRUE)

# # # have_bulk <- file.exists(BULK_COUNTS_FILE)
# # # if (have_bulk) {
# # #   bulk_dt <- data.table::fread(BULK_COUNTS_FILE)
# # #   gcol2 <- intersect(c("gene","symbol","Gene","Gene_Symbol","ID","Name","GeneID","genes"), names(bulk_dt))[1]
# # #   if (is.na(gcol2)) gcol2 <- names(bulk_dt)[1]
# # #   bgenes <- bulk_dt[[gcol2]]; bulk_dt[[gcol2]] <- NULL
# # #   X_bulk <- ensure_double_matrix(as.matrix(bulk_dt)); rownames(X_bulk) <- as.character(bgenes)
# # #   X_cpm  <- cpm_mat(X_bulk)
# # # }

# # # ### NEW: Residual diagnostics comparing observed bulk CPM vs reconstructed Xhat
# # # if (have_bulk) try({
# # #   g_common <- intersect(rownames(Z_use), rownames(X_cpm))
# # #   if (length(g_common) >= 50) {
# # #     Zg   <- ensure_double_matrix(Z_use[g_common, , drop=FALSE])
# # #     Psg  <- ensure_double_matrix(P_use)
# # #     Xhat <- ensure_double_matrix(Zg %*% Psg)                 # genes × samples (reconstructed)
# # #     s_common <- intersect(colnames(X_cpm), colnames(Xhat))
# # #     if (length(s_common) >= 3) {
# # #       Y <- ensure_double_matrix(X_cpm[g_common, s_common, drop=FALSE])
# # #       H <- ensure_double_matrix(Xhat[     , s_common, drop=FALSE])
# # #       diag_stats <- lapply(seq_along(s_common), function(i){
# # #         sj <- s_common[i]
# # #         y  <- as.numeric(Y[, i])
# # #         h  <- as.numeric(H[, i])
# # #         r2 <- tryCatch({ summary(stats::lm(y ~ h))$r.squared }, error=function(e) NA_real_)
# # #         rho <- suppressWarnings(stats::cor(y, h, method="spearman"))
# # #         data.frame(sample=sj, R2=r2, spearman_rho=rho, stringsAsFactors = FALSE)
# # #       })
# # #       diag_df <- data.table::rbindlist(diag_stats)
# # #       data.table::fwrite(diag_df, file.path(TAB_DIR, "residual_diagnostics.tsv"), sep="\t")

# # #       g_r2 <- ggplot2::ggplot(diag_df, ggplot2::aes(R2)) +
# # #         ggplot2::geom_histogram(bins=30, alpha=.9) +
# # #         theme_pub() + ggplot2::labs(title="Reconstruction R² across samples", x="R²", y="Count")
# # #       ggplot2::ggsave(file.path(PLOT_DIR,"03_reconstruction_r2_hist.png"), g_r2, width=6.2, height=4.5, dpi=200)
# # #     }
# # #   }
# # # }, silent = TRUE)

# # # have_sc_props <- !is.null(res$sc.props)
# # # if (have_sc_props) {
# # #   Ct_raw <- ensure_double_matrix(as.matrix(res$sc.props))
# # #   rownames(Ct_raw) <- harmonize_labels(rownames(res$sc.props))
# # # }

# # # # --- QC: sums table + hist
# # # prop_sums <- colSums(P_use, na.rm=TRUE)
# # # qc_df <- data.frame(sample=names(prop_sums), sum_props=as.numeric(prop_sums),
# # #                     abs_dev=abs(as.numeric(prop_sums)-1))
# # # data.table::fwrite(qc_df, file.path(TAB_DIR,"qc_prop_sums.tsv"), sep="\t")

# # # g_hist <- ggplot2::ggplot(qc_df, ggplot2::aes(sum_props)) +
# # #   ggplot2::geom_histogram(bins=30, alpha=.9) +
# # #   ggplot2::geom_vline(xintercept=1, linetype=2, color="red") +
# # #   theme_pub() + ggplot2::labs(title="Distribution of proportion sums", x="Sum of proportions", y="Count")
# # # ggplot2::ggsave(file.path(PLOT_DIR,"02_qc_sum_hist.png"), g_hist, width=6.2, height=4.5, dpi=200)

# # # # --- Stacked composition
# # # hc <- hclust(dist(prop_mat), method = "complete")
# # # sample_order <- rownames(prop_mat)[hc$order]
# # # props_long$sample <- factor(props_long$sample, levels = sample_order)
# # # gg1 <- ggplot2::ggplot(props_long, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
# # #   ggplot2::geom_bar(stat = "identity", width = 0.95) +
# # #   ggplot2::labs(title = "Cell-type composition per sample (cluster-ordered)", x = "Sample", y = "Proportion") +
# # #   ggplot2::theme_bw() + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=90, vjust=0.5, hjust=1))
# # # ggplot2::ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), gg1, width = 16, height = 7, dpi = 200)

# # # # --- Heatmap of proportions
# # # pheatmap::pheatmap(prop_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
# # #                    filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
# # #                    main="Proportions heatmap", width=10, height=13)

# # # # --- Violin/box of CT distributions
# # # g_boxv <- ggplot2::ggplot(props_long, ggplot2::aes(cell_type, prop, fill=cell_type)) +
# # #   ggplot2::geom_violin(trim=FALSE, alpha=.85) + ggplot2::geom_boxplot(width=.12, outlier.size=.4) +
# # #   viridis::scale_fill_viridis(discrete=TRUE, guide="none") + theme_pub() +
# # #   ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45, hjust=1)) +
# # #   ggplot2::labs(title="Distribution of Bisque-estimated proportions", x="Cell type", y="Proportion")
# # # ggplot2::ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), g_boxv, width=11, height=6.5, dpi=200)

# # # ## =================== OPTIONAL MARKER CHECKS & EXPORTS =======================
# # # eps <- 1e-8
# # # other_mean <- function(mat, ct) { if (ncol(mat)==1) return(rep(0, nrow(mat))); rowMeans(mat[, setdiff(colnames(mat), ct), drop=FALSE], na.rm=TRUE) }
# # # spec_list <- lapply(colnames(Z_use), function(ct){
# # #   spec <- log2((Z_use[, ct] + eps) / (other_mean(Z_use, ct) + eps))
# # #   data.frame(gene = rownames(Z_use), cell_type = ct, specificity = spec, expr_ct = Z_use[, ct], stringsAsFactors = FALSE)
# # # })
# # # spec_df <- data.table::rbindlist(spec_list)
# # # markers_sel <- spec_df[order(cell_type, -specificity), .SD[1:min(.N, TOP_MARKERS_PER_CT)], by = cell_type]
# # # data.table::fwrite(markers_sel, file.path(TAB_DIR, "top_markers_per_celltype.tsv"), sep="\t")

# # # if (!is.null(MARKER_FILE) && file.exists(MARKER_FILE) && have_bulk) {
# # #   mk <- data.table::fread(MARKER_FILE)
# # #   if (nrow(mk)) {
# # #     data.table::setnames(mk, old = names(mk), new = tolower(names(mk)))
# # #     gene_col <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","feature","id","geneid","name"), names(mk))[1]
# # #     ct_col   <- intersect(c("cell_type","celltype","cell_type_label","celltype_label","cluster","label","cell","ct"), names(mk))[1]
# # #     if (!is.na(gene_col) && !is.na(ct_col)) {
# # #       markers <- mk[, .(gene = get(gene_col), cell_type = harmonize_labels(get(ct_col)))]
# # #       markers <- unique(markers[!is.na(gene) & nzchar(gene) & !is.na(cell_type) & nzchar(cell_type)])
# # #       bulk_dt0 <- data.table::fread(BULK_COUNTS_FILE)
# # #       gcol3 <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","id","name","geneid","genes"), names(bulk_dt0))[1]
# # #       if (is.na(gcol3)) gcol3 <- names(bulk_dt0)[1]
# # #       bgenes <- bulk_dt0[[gcol3]]; bulk_dt0[[gcol3]] <- NULL
# # #       bulk_mat <- ensure_double_matrix(as.matrix(bulk_dt0)); rownames(bulk_mat) <- as.character(bgenes)
# # #       bulk_cpm <- cpm_mat(bulk_mat)
# # #       ct_ok <- intersect(unique(markers$cell_type), rownames(P_use))
# # #       ctab <- lapply(ct_ok, function(ct) {
# # #         gs <- intersect(markers[cell_type == ct, gene], rownames(bulk_cpm))
# # #         if (!length(gs)) return(NULL)
# # #         score <- colMeans(bulk_cpm[gs, , drop = FALSE])
# # #         common <- intersect(names(score), colnames(P_use))
# # #         if (length(common) < 3) return(NULL)
# # #         rho <- suppressWarnings(cor(score[common], P_use[ct, common], method = "spearman"))
# # #         data.frame(cell_type = ct, n_genes = length(gs), n_samples = length(common), spearman_rho = rho)
# # #       })
# # #       ctab <- data.table::rbindlist(ctab, fill = TRUE)
# # #       if (!is.null(ctab) && nrow(ctab)) data.table::fwrite(ctab, file.path(TAB_DIR, "marker_prop_spearman.tsv"), sep = "\t")
# # #     }
# # #   }
# # # }

# # # # ====================== S3 ADD-ONS: OVERVIEW + SAMPLE CARDS ===================
# # # # (Safe: only runs if objects exist; otherwise writes placeholders.)
# # # safe_ggsave <- function(path, plot=NULL, width=8, height=5, dpi=200, title_when_empty=NULL){
# # #   if (!is.null(plot)) {
# # #     ggplot2::ggsave(path, plot, width=width, height=height, dpi=dpi)
# # #   } else {
# # #     png(path, width=width*100, height=height*100)
# # #     par(mar=c(2,2,2,2)); plot.new()
# # #     ttl <- if (is.null(title_when_empty)) basename(path) else title_when_empty
# # #     title(ttl); mtext("No data to plot", side=3, line=-1, col="gray40")
# # #     dev.off()
# # #   }
# # # }

# # # dir.create(file.path(PLOT_DIR, "S3_sample_cards"), showWarnings = FALSE, recursive = TRUE)

# # # # ---------- S3 overview: overall composition pie ----------
# # # overall_pie <- NULL
# # # if (exists("props_long") && nrow(props_long)) {
# # #   dfpie <- props_long[, .(prop = mean(prop, na.rm=TRUE)), by=cell_type]
# # #   dfpie <- dfpie[prop > 0]
# # #   if (nrow(dfpie)) {
# # #     overall_pie <- ggplot2::ggplot(dfpie, ggplot2::aes(x="", y=prop, fill=cell_type)) +
# # #       ggplot2::geom_bar(stat="identity", width=0.9) +
# # #       ggplot2::coord_polar(theta="y") +
# # #       viridis::scale_fill_viridis(discrete=TRUE, guide=NULL) +
# # #       ggplot2::labs(title="Overall mean composition (across samples)", y=NULL, x=NULL) +
# # #       ggplot2::theme_void() +
# # #       ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold"))
# # #   }
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S3_overview_composition_pie.png"), overall_pie, 7, 7,
# # #             title_when_empty="Overall mean composition (pie)")

# # # # ---------- S3 overview: CT prevalence across samples ----------
# # # ct_prev_plot <- NULL
# # # if (exists("P_use") && ncol(P_use) > 0) {
# # #   prev <- data.table::as.data.table(P_use)
# # #   prev[, cell_type := rownames(P_use)]
# # #   prev_long <- data.table::melt(prev, id.vars="cell_type", variable.name="sample", value.name="prop")
# # #   prev_ct <- prev_long[, .(prevalence = sum(prop > 0, na.rm=TRUE)), by=cell_type]
# # #   if (nrow(prev_ct)) {
# # #     ct_prev_plot <- ggplot2::ggplot(prev_ct, ggplot2::aes(reorder(cell_type, prevalence), prevalence)) +
# # #       ggplot2::geom_col() + ggplot2::coord_flip() +
# # #       ggplot2::labs(title="CT prevalence (#samples with prop>0)", x="Cell type", y="#Samples") +
# # #       ggplot2::theme_bw()
# # #   }
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S3_celltype_prevalence.png"), ct_prev_plot, 8, 7,
# # #             title_when_empty="CT prevalence")

# # # # ---------- S3 overview: sample × sample correlation of compositions ----------
# # # corr_plot_path <- file.path(PLOT_DIR, "S3_props_correlation_heatmap.png")
# # # if (exists("prop_mat") && nrow(prop_mat) > 1 && ncol(prop_mat) > 1) {
# # #   # rows = samples, cols = CTs
# # #   cm <- suppressWarnings(cor(t(prop_mat), method="spearman", use="pairwise.complete.obs"))
# # #   cm[!is.finite(cm)] <- 0
# # #   tryCatch({
# # #     pheatmap::pheatmap(cm, filename=corr_plot_path, main="Sample×Sample Spearman correlation (compositions)",
# # #                        cluster_rows=TRUE, cluster_cols=TRUE, width=10, height=10)
# # #   }, error=function(e){
# # #     safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty=paste("Correlation heatmap:", e$message))
# # #   })
# # # } else {
# # #   safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty="Correlation heatmap (insufficient data)")
# # # }

# # # # ---------- S3 overview: PCA of compositions ----------
# # # pca_plot <- NULL
# # # if (exists("prop_mat") && nrow(prop_mat) >= 2 && ncol(prop_mat) >= 2) {
# # #   X <- prop_mat
# # #   X <- X[, colSums(is.finite(X)) == nrow(X), drop=FALSE]
# # #   if (ncol(X) >= 2) {
# # #     pc <- tryCatch(prcomp(X, center=TRUE, scale.=TRUE), error=function(e) NULL)
# # #     if (!is.null(pc)) {
# # #       sc <- data.frame(PC1=pc$x[,1], PC2=pc$x[,2], sample=rownames(pc$x))
# # #       pca_plot <- ggplot2::ggplot(sc, ggplot2::aes(PC1, PC2, label=sample)) +
# # #         ggplot2::geom_point() + ggrepel::geom_text_repel(size=3, max.overlaps=40) +
# # #         ggplot2::theme_bw() + ggplot2::labs(title="PCA of sample compositions")
# # #     }
# # #   }
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S3_props_PCA.png"), pca_plot, 8, 6,
# # #             title_when_empty="PCA of compositions (insufficient data)")

# # # # ---------- S1 sanity: mean signature per CT ----------
# # # sig_mean_plot <- NULL
# # # if (exists("Z_use") && ncol(Z_use) >= 1) {
# # #   means <- colMeans(Z_use, na.rm=TRUE)
# # #   dfm <- data.frame(cell_type=names(means), mean_expr=as.numeric(means))
# # #   sig_mean_plot <- ggplot2::ggplot(dfm, ggplot2::aes(reorder(cell_type, mean_expr), mean_expr)) +
# # #     ggplot2::geom_col() + ggplot2::coord_flip() +
# # #     ggplot2::theme_bw() + ggplot2::labs(title="Mean signature per CT (CPM)", x="Cell type", y="Mean CPM")
# # # }
# # # safe_ggsave(file.path(PLOT_DIR,"S1_signature_ct_means.png"), sig_mean_plot, 7, 6,
# # #             title_when_empty="Mean signature per CT")

# # # # ---------- S3: per-sample QC TSV + SAMPLE CARDS (PNG) ----------
# # # qc_out <- file.path(TAB_DIR, "S3_sample_qc.tsv")
# # # qc_df2 <- data.frame()
# # # if (exists("P_use") && ncol(P_use) > 0) {
# # #   prop_sums_local <- colSums(P_use, na.rm=TRUE)
# # #   abs_dev <- abs(prop_sums_local - 1)
# # #   qc_df2 <- data.frame(sample = names(prop_sums_local),
# # #                        sum_props = as.numeric(prop_sums_local),
# # #                        abs_dev = as.numeric(abs_dev))
# # #   data.table::fwrite(qc_df2, qc_out, sep="\t")
# # # } else {
# # #   data.table::fwrite(data.frame(), qc_out, sep="\t")
# # # }

# # # # Helper: top genes plot for a sample (from X_cpm, if present)
# # # top_genes_plot <- function(sample_id, topN=10){
# # #   if (!exists("X_cpm") || !is.matrix(X_cpm) || !(sample_id %in% colnames(X_cpm))) return(NULL)
# # #   v <- X_cpm[, sample_id]; if (all(!is.finite(v))) return(NULL)
# # #   ord <- order(v, decreasing=TRUE); k <- min(topN, length(ord))
# # #   if (k < 1) return(NULL)
# # #   sel <- ord[seq_len(k)]
# # #   df <- data.frame(gene=names(v)[sel], CPM=as.numeric(v[sel]))
# # #   ggplot2::ggplot(df, ggplot2::aes(reorder(gene, CPM), CPM)) +
# # #     ggplot2::geom_col() + ggplot2::coord_flip() +
# # #     ggplot2::theme_bw() + ggplot2::labs(title=paste0("Top ", k, " genes in ", sample_id), x=NULL, y="CPM")
# # # }

# # # # Helper: per-sample composition stacked bar (single sample)
# # # sample_comp_plot <- function(sample_id){
# # #   if (!exists("props_long") || !nrow(props_long)) return(NULL)
# # #   dl <- props_long[sample == sample_id]
# # #   if (!nrow(dl)) return(NULL)
# # #   ggplot2::ggplot(dl, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
# # #     ggplot2::geom_bar(stat="identity", width=0.8) +
# # #     viridis::scale_fill_viridis(discrete=TRUE, guide="none") +
# # #     ggplot2::theme_bw() + ggplot2::labs(title=paste0("Composition – ", sample_id), x=NULL, y="Proportion")
# # # }

# # # # Helper: sample QC tile
# # # sample_qc_tile <- function(sample_id){
# # #   sp <- tryCatch(qc_df2[qc_df2$sample == sample_id, , drop=FALSE], error=function(e) NULL)
# # #   if (is.null(sp) || !nrow(sp)) return(NULL)
# # #   lbl <- sprintf("sum(props)=%.3f\n|sum-1|=%.3f", sp$sum_props[1], sp$abs_dev[1])
# # #   ggplot2::ggplot(data.frame(x=1,y=1,label=lbl), ggplot2::aes(x,y,label=label)) +
# # #     ggplot2::geom_text(size=5) + ggplot2::theme_void() +
# # #     ggplot2::labs(title="QC")
# # # }

# # # # Render sample cards
# # # if (exists("prop_mat") && nrow(prop_mat) >= 1) {
# # #   smpls <- rownames(prop_mat)
# # #   for (s in smpls) {
# # #     p1 <- sample_comp_plot(s)
# # #     p2 <- top_genes_plot(s, topN = max(5, min(10, if (exists("TOP_GENES_PER_SAMPLE")) TOP_GENES_PER_SAMPLE else 10)))
# # #     p3 <- sample_qc_tile(s)
# # #     # Arrange with patchwork if at least one plot exists
# # #     if (!is.null(p1) || !is.null(p2) || !is.null(p3)) {
# # #       card <- NULL
# # #       if (!is.null(p1) && !is.null(p2) && !is.null(p3)) card <- p1 + p2 + p3 + patchwork::plot_layout(ncol=3, widths=c(1.2,1,0.6))
# # #       else if (!is.null(p1) && !is.null(p2)) card <- p1 + p2 + patchwork::plot_layout(ncol=2, widths=c(1.2,1))
# # #       else if (!is.null(p1)) card <- p1
# # #       else if (!is.null(p2)) card <- p2
# # #       else card <- p3
# # #       outp <- file.path(PLOT_DIR, "S3_sample_cards", paste0(s, ".png"))
# # #       safe_ggsave(outp, card, width=15, height=5, dpi=200, title_when_empty=paste("Sample card:", s))
# # #     } else {
# # #       safe_ggsave(file.path(PLOT_DIR, "S3_sample_cards", paste0(s, ".png")), NULL, 12, 4,
# # #                   title_when_empty=paste("Sample card:", s))
# # #     }
# # #   }
# # # } else {
# # #   # No compositions: still produce a note
# # #   safe_ggsave(file.path(PLOT_DIR, "S3_sample_cards", "no_samples.png"), NULL, 10, 3,
# # #               title_when_empty="No sample compositions available for cards")
# # # }

# # # ### NEW: Minimal self-contained HTML summary
# # # try({
# # #   html <- file.path(OUT_DIR, "bisque_summary.html")
# # #   lines <- c(
# # #     "<!doctype html>",
# # #     "<html><head><meta charset='utf-8'><title>Bisque Summary</title>",
# # #     "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;margin:24px;}h1,h2{margin-top:1.2em} code{background:#f6f8fa;padding:2px 4px;border-radius:4px}table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px}</style>",
# # #     "</head><body>",
# # #     sprintf("<h1>Bisque Deconvolution Summary</h1><p><b>Output dir:</b> <code>%s</code></p>", OUT_DIR),
# # #     "<h2>Key Tables</h2><ul>",
# # #     "<li><a href='bisque_bulk_proportions.tsv'>bisque_bulk_proportions.tsv</a></li>",
# # #     "<li><a href='signature_matrix.tsv'>signature_matrix.tsv</a></li>",
# # #     "<li><a href='bisque_genes_used.txt'>bisque_genes_used.txt</a></li>",
# # #     sprintf("<li><a href='%s'>qc_prop_sums.tsv</a></li>", file.path("QC & Performance Metrics","qc_prop_sums.tsv")),
# # #     sprintf("<li><a href='%s'>top_markers_per_celltype.tsv</a></li>", file.path("QC & Performance Metrics","top_markers_per_celltype.tsv")),
# # #     sprintf("<li><a href='%s'>top_%d_genes_per_sample.tsv</a></li>", file.path("QC & Performance Metrics", sprintf("top_%d_genes_per_sample.tsv", TOP_GENES_PER_SAMPLE))),
# # #     "</ul>",
# # #     "<h2>Diagnostic Plots</h2><ul>",
# # #     "<li><a href='prop_sums.png'>prop_sums.png</a></li>",
# # #     sprintf("<li><a href='%s'>02_qc_sum_hist.png</a></li>", file.path("deconv_visual_reports","02_qc_sum_hist.png")),
# # #     sprintf("<li><a href='%s'>plot_stacked_bar_by_sample_clustered.png</a></li>", file.path("deconv_visual_reports","plot_stacked_bar_by_sample_clustered.png")),
# # #     sprintf("<li><a href='%s'>F3_heatmap_proportions.png</a></li>", file.path("deconv_visual_reports","F3_heatmap_proportions.png")),
# # #     sprintf("<li><a href='%s'>F3_box_violin_cohort.png</a></li>", file.path("deconv_visual_reports","F3_box_violin_cohort.png")),
# # #     sprintf("<li><a href='%s'>03_reconstruction_r2_hist.png</a></li>", file.path("deconv_visual_reports","03_reconstruction_r2_hist.png")),
# # #     sprintf("<li><a href='%s'>F4_group_mean_proportions.png</a></li>", file.path("deconv_visual_reports","F4_group_mean_proportions.png")),
# # #     "</ul>",
# # #     "<p style='color:#666'>Generated automatically by the Bisque runner.</p>",
# # #     "</body></html>"
# # #   )
# # #   lines <- unlist(lines, use.names = FALSE)
# # #   con <- file(html, open="wt", encoding="UTF-8"); on.exit(close(con), add=TRUE)
# # #   writeLines(lines, con)
# # #   msg("Wrote summary HTML → ", html)
# # # }, silent = TRUE)

# # # # ==================== END S3 ADD-ONS =========================================

# # # msg("DONE. Outputs written to: ", OUT_DIR)
# # # msg("Tables → ", TAB_DIR)
# # # msg("Plots  → ", PLOT_DIR)

# # # #--------------------------------------#

# # #!/usr/bin/env Rscript

# # options(stringsAsFactors = FALSE)
# # ### NEW: deterministic randomness for any sampling/plots
# # set.seed(42L)

# # ## ============================ OPTIONS =======================================
# # # Environment variables set by Python
# # get_bool  <- function(x, default = FALSE) {
# #   v <- toupper(Sys.getenv(x, ifelse(default, "TRUE", "FALSE"))); v %in% c("1","T","TRUE","YES","Y")
# # }
# # get_int   <- function(x, default = NA_integer_) {
# #   v <- suppressWarnings(as.integer(Sys.getenv(x, ""))); if (is.na(v)) default else v
# # }
# # get_num   <- function(x, default = NA_real_) {
# #   v <- suppressWarnings(as.numeric(Sys.getenv(x, ""))); if (is.na(v)) default else v
# # }
# # get_str   <- function(x, default = "") {
# #   v <- Sys.getenv(x, ""); if (!nzchar(v)) default else v
# # }

# # REF_DIR_PY            <- get_str("REF_DIR_PY", "")
# # OUT_DIR_PY            <- get_str("OUT_DIR_PY", "")
# # DROP_MITO_PY          <- get_bool("DROP_MITO_PY", FALSE)
# # ALLOW_DUP_PY          <- get_bool("ALLOW_DUP_PY", FALSE)
# # WARN_MIN_OVERLAP_PY   <- get_int ("WARN_MIN_OVERLAP_PY", 100L)
# # MAX_PROP_DEV_PY       <- get_num ("MAX_PROP_DEV_PY", 0.2)
# # TOP_MARKERS_PER_CT_PY <- get_int ("TOP_MARKERS_PER_CT_PY", 10L)
# # TOP_GENES_PER_SAMPLE_PY <- get_int("TOP_GENES_PER_SAMPLE_PY", 50L)
# # AUTO_INSTALL_PY       <- get_bool("AUTO_INSTALL_PY", TRUE)

# # if (!nzchar(REF_DIR_PY)) stop("Environment variable REF_DIR_PY not set; Python must pass it to R.")

# # ## ============================ UTILITIES =====================================
# # msg <- function(...) cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] "), ..., "\n")

# # safe_dev_off <- function() {
# #   if (names(dev.cur()) != "null device") try(dev.off(), silent = TRUE)
# # }

# # ensure_double_matrix <- function(m) {
# #   if (isS4(m)) m <- as.matrix(m)
# #   if (!is.matrix(m)) m <- as.matrix(m)
# #   storage.mode(m) <- "double"; m
# # }

# # is_all_numeric_vec <- function(x) all(grepl("^[0-9]+$", as.character(x)))
# # norm_bc <- function(x){ x <- trimws(as.character(x)); x <- gsub("^X(?=\\d)","",x,perl=TRUE); x <- gsub("\\.", "-", x); x }
# # drop_mito <- function(genes) genes[!grepl("^MT-", toupper(genes))]
# # zero_var_rows <- function(m) { v <- apply(m, 1, function(x) var(x, na.rm=TRUE)); names(which(is.na(v) | v == 0)) }

# # collapse_dups <- function(m) {
# #   m <- ensure_double_matrix(m)
# #   if (is.null(rownames(m))) stop("Matrix lacks rownames.")
# #   if (!any(duplicated(rownames(m)))) return(m)
# #   msg("Duplicate gene IDs detected; collapsing by sum.")
# #   ux <- unique(rownames(m)); idx <- match(rownames(m), ux)
# #   sp <- Matrix::sparseMatrix(i = idx, j = seq_len(nrow(m)), x = 1,
# #                              dims = c(length(ux), nrow(m)))
# #   out <- ensure_double_matrix(as.matrix(sp %*% m))
# #   rownames(out) <- ux; colnames(out) <- colnames(m)
# #   out
# # }

# # drop_zero_expr_samples <- function(m, label="bulk") {
# #   m <- ensure_double_matrix(m)
# #   cs <- colSums(m, na.rm=TRUE); zc <- which(!is.finite(cs) | cs <= 0)
# #   if (length(zc)) {
# #     msg("Dropping ", length(zc), " ", label, " sample(s) with zero/NA expr: ",
# #         paste(colnames(m)[zc], collapse=", "))
# #     m <- m[, -zc, drop=FALSE]
# #   }
# #   if (ncol(m) == 0) stop("All ", label, " samples dropped; cannot proceed.")
# #   m
# # }

# # theme_pub <- function(){
# #   ggplot2::theme_bw(base_size=12) +
# #     ggplot2::theme(plot.title=ggplot2::element_text(face="bold"),
# #                    panel.grid.minor=ggplot2::element_blank())
# # }

# # cpm_mat <- function(m){
# #   m <- ensure_double_matrix(m)
# #   lib <- colSums(m,na.rm=TRUE); lib[!is.finite(lib)|lib==0] <- 1
# #   ensure_double_matrix(t(1e6*t(m)/lib))
# # }

# # harmonize_labels <- function(x) make.names(trimws(as.character(x)), unique = FALSE)

# # ## =========================== PATHS & PARAMS =================================
# # REF_DIR <- REF_DIR_PY
# # OUT_DIR_DEFAULT <- file.path(REF_DIR, "bisque_deconv-results")
# # OUT_DIR <- if (!is.null(OUT_DIR_PY) && nzchar(OUT_DIR_PY)) OUT_DIR_PY else OUT_DIR_DEFAULT
# # DirOk1 <- dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# # DROP_MITO_GENES       <- isTRUE(DROP_MITO_PY)
# # ALLOW_DUPLICATE_GENES <- isTRUE(ALLOW_DUP_PY)
# # WARN_MIN_OVERLAP      <- as.integer(WARN_MIN_OVERLAP_PY)
# # MAX_PROP_DEVIATION    <- as.numeric(MAX_PROP_DEV_PY)
# # TOP_MARKERS_PER_CT    <- as.integer(TOP_MARKERS_PER_CT_PY)
# # TOP_GENES_PER_SAMPLE  <- as.integer(TOP_GENES_PER_SAMPLE_PY)
# # AUTO_INSTALL          <- isTRUE(AUTO_INSTALL_PY)

# # SC_COUNTS_FILE   <- file.path(REF_DIR, "sc_counts.tsv")
# # SC_META_FILE     <- file.path(REF_DIR, "sc_metadata.tsv")
# # BULK_COUNTS_FILE <- file.path(REF_DIR, "bulk_counts_overlap.tsv")

# # SAMPLE_META_FILE <- file.path(REF_DIR, "sample_metadata_patientAKI.tsv")  # optional
# # MARKER_FILE      <- file.path(REF_DIR, "marker_gene_table.tsv")           # optional

# # PLOT_DIR <- file.path(OUT_DIR, "deconv_visual_reports"); dir.create(PLOT_DIR, FALSE, TRUE)
# # TAB_DIR  <- file.path(OUT_DIR, "QC & Performance Metrics"); dir.create(TAB_DIR,  FALSE, TRUE)

# # ## ============================ PACKAGES ======================================
# # cran_pkgs <- c("data.table","Matrix","tools","nnls","ggplot2","viridis",
# #                "pheatmap","corrplot","dplyr","tidyr","patchwork","ggrepel")
# # bio_pkgs  <- c("Biobase","BisqueRNA")

# # need_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
# # need_bio  <- bio_pkgs[!vapply(bio_pkgs,  requireNamespace, logical(1), quietly = TRUE)]

# # if (AUTO_INSTALL) {
# #   if (length(need_cran)) {
# #     msg("Installing CRAN packages: ", paste(need_cran, collapse=", "))
# #     install.packages(need_cran, repos = "https://cloud.r-project.org")
# #   }
# #   if (!requireNamespace("BiocManager", quietly = TRUE)) {
# #     install.packages("BiocManager", repos = "https://cloud.r-project.org")
# #   }
# #   if (length(need_bio)) {
# #     msg("Installing Bioconductor packages: ", paste(need_bio, collapse=", "))
# #     BiocManager::install(need_bio, update = FALSE, ask = FALSE)
# #   }
# # } else if (length(need_cran) || length(need_bio)) {
# #   stop("Missing R packages: ", paste(c(need_cran, need_bio), collapse=", "),
# #        "\nRun again with auto_install=TRUE or install manually.")
# # }

# # suppressPackageStartupMessages({
# #   lapply(c(cran_pkgs, bio_pkgs), library, character.only = TRUE)
# # })
# # capture.output(sessionInfo(), file = file.path(OUT_DIR, "R_sessionInfo.txt"))

# # ## ======================= LOAD SINGLE-CELL DATA ==============================
# # stopifnot(file.exists(SC_COUNTS_FILE), file.exists(SC_META_FILE))
# # msg("Loading single-cell counts: ", SC_COUNTS_FILE)
# # cnt_full <- data.table::fread(SC_COUNTS_FILE, sep="\t", header=TRUE, data.table=FALSE)

# # # First column must be 'gene'; if not, treat the first as gene column anyway.
# # genes <- as.character(cnt_full[[1]]); cnt_full[[1]] <- NULL
# # m_sc <- ensure_double_matrix(as.matrix(cnt_full))
# # rownames(m_sc) <- genes
# # counts_bc_raw <- colnames(cnt_full)

# # msg("Loading metadata: ", SC_META_FILE)
# # md <- data.table::fread(SC_META_FILE, sep="\t", header=TRUE, data.table=FALSE)
# # if (!"cell_type" %in% names(md))     md$cell_type     <- NA_character_
# # if (!"individual_id" %in% names(md)) md$individual_id <- "ONE_SUBJECT"
# # if (!"cell_barcode" %in% names(md))  md$cell_barcode  <- NA_character_

# # counts_bc <- norm_bc(counts_bc_raw); meta_bc <- norm_bc(md$cell_barcode)
# # if (nrow(md) != length(counts_bc)) {
# #   stop(sprintf("Metadata rows (%d) != number of cells in counts (%d). Re-export so sizes match.",
# #                nrow(md), length(counts_bc)))
# # }
# # if (length(na.omit(unique(meta_bc))) <= 2 || is_all_numeric_vec(na.omit(meta_bc))) {
# #   md$cell_barcode <- counts_bc; meta_bc <- counts_bc
# # }
# # if (is_all_numeric_vec(counts_bc) && is_all_numeric_vec(meta_bc)) {
# #   msg("Counts & metadata both numeric; generating synthetic barcodes (in-memory).")
# #   syn <- sprintf("C%06d", seq_len(length(counts_bc)))
# #   colnames(m_sc) <- syn; md$cell_barcode <- syn
# #   counts_bc <- syn; meta_bc <- syn
# # } else {
# #   colnames(m_sc) <- counts_bc
# # }

# # # Align (transpose rescue + synthetic fallback)
# # ov <- intersect(colnames(m_sc), md$cell_barcode)
# # if (!length(ov) && any(rownames(m_sc) %in% md$cell_barcode)) {
# #   msg("Counts appear transposed; transposing ...")
# #   m_sc <- ensure_double_matrix(t(m_sc))
# #   ov <- intersect(colnames(m_sc), md$cell_barcode)
# # }
# # if (!length(ov)) {
# #   syn <- sprintf("C%06d", seq_len(ncol(m_sc)))
# #   colnames(m_sc) <- syn; md$cell_barcode <- syn; ov <- syn
# # }
# # m_sc <- m_sc[, ov, drop=FALSE]
# # md <- md[match(ov, md$cell_barcode), , drop=FALSE]
# # rownames(md) <- md$cell_barcode

# # # Build ExpressionSet for sc
# # pdat <- new("AnnotatedDataFrame", data = data.frame(
# #   cellType    = md$cell_type,
# #   SubjectName = md$individual_id,
# #   row.names   = rownames(md),
# #   check.names = FALSE
# # ))
# # sc.eset <- Biobase::ExpressionSet(assayData = ensure_double_matrix(m_sc), phenoData = pdat)

# # ## ============================== LOAD BULK ===================================
# # stopifnot(file.exists(BULK_COUNTS_FILE))
# # read_bulk <- function(path) {
# #   dt <- data.table::fread(path, sep="\t")
# #   cand <- c("gene","genes","gene_id","geneid","ensembl","ensembl_id","symbol",
# #             "gene_symbol","hgnc","Gene","GeneID","Gene_Symbol","Gene.Name","ID","Name")
# #   gcol <- intersect(cand, names(dt))[1]
# #   if (is.na(gcol)) {
# #     if (!is.numeric(dt[[1]]) && length(unique(dt[[1]])) == nrow(dt)) gcol <- names(dt)[1]
# #     else stop("Cannot detect gene column in bulk file.")
# #   }
# #   genes <- as.character(dt[[gcol]]); dt[[gcol]] <- NULL
# #   m <- ensure_double_matrix(as.matrix(dt))
# #   rownames(m) <- genes
# #   msg("Bulk matrix dims (genes x samples): ", nrow(m), " x ", ncol(m))
# #   m
# # }
# # bulk.mat <- read_bulk(BULK_COUNTS_FILE)

# # ## ============================ SANITY + FILTERS ==============================
# # stopifnot(ncol(Biobase::exprs(sc.eset)) == nrow(Biobase::pData(sc.eset)))

# # if (any(Biobase::exprs(sc.eset) < 0, na.rm=TRUE)) stop("Negative values in single-cell counts.")
# # if (any(bulk.mat < 0, na.rm=TRUE))                stop("Negative values in bulk counts.")

# # if (ALLOW_DUPLICATE_GENES) {
# #   msg("Collapsing duplicate genes (safe rebuild)...")
# #   expr_new <- collapse_dups(Biobase::exprs(sc.eset))
# #   sc.eset <- Biobase::ExpressionSet(
# #     assayData = expr_new,
# #     phenoData = Biobase::phenoData(sc.eset)
# #   )
# #   bulk.mat <- collapse_dups(bulk.mat)
# # }

# # if (DROP_MITO_GENES) {
# #   keep_sc   <- drop_mito(rownames(sc.eset));    sc.eset  <- sc.eset[keep_sc, ]
# #   keep_bulk <- drop_mito(rownames(bulk.mat));   bulk.mat <- bulk.mat[keep_bulk, , drop=FALSE]
# # }

# # zv <- zero_var_rows(Biobase::exprs(sc.eset))
# # if (length(zv)) {
# #   sc.eset <- sc.eset[setdiff(rownames(Biobase::exprs(sc.eset)), zv), ]
# #   msg("Removed ", length(zv), " zero-variance genes from sc.")
# # }

# # bulk_zero_rows <- which(rowSums(bulk.mat == 0, na.rm=TRUE) == ncol(bulk.mat))
# # if (length(bulk_zero_rows)) {
# #   bulk.mat <- bulk.mat[-bulk_zero_rows, , drop=FALSE]
# #   msg("Removed ", length(bulk_zero_rows), " unexpressed genes from bulk (rows).")
# # }

# # # Ensure unique & ordered gene set
# # gene_set <- intersect(rownames(Biobase::exprs(sc.eset)), rownames(bulk.mat))
# # gene_set <- unique(gene_set)

# # if (length(gene_set) < WARN_MIN_OVERLAP) {
# #   msg("WARNING: low overlap gene count (", length(gene_set), "). Results may be less stable.")
# # }
# # if (length(gene_set) < 2L) {
# #   stop(sprintf("Too few overlapping genes after prefilters (|genes| = %d).", length(gene_set)))
# # }

# # # ExpressionSet-aware subsetting
# # sc.eset  <- sc.eset[gene_set, ]
# # bulk.mat <- ensure_double_matrix(bulk.mat[gene_set, , drop=FALSE])

# # # Dim sanity checks
# # msg(sprintf("Dims after alignment: sc.eset %d genes × %d cells; bulk %d genes × %d samples",
# #             nrow(Biobase::exprs(sc.eset)), ncol(Biobase::exprs(sc.eset)),
# #             nrow(bulk.mat), ncol(bulk.mat)))

# # # Drop bulk samples that are zero over the selected genes
# # bulk.mat <- drop_zero_expr_samples(bulk.mat, label="bulk")

# # ## ============ DONOR-LEVEL PSEUDOBULK (CPM) & SIGNATURE MATRIX ===============
# # msg("Building donor-level pseudobulk (CPM) ...")
# # expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))     # genes × cells
# # ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# # subj_vec<- as.character(Biobase::pData(sc.eset)$SubjectName)
# # key <- paste(subj_vec, ct_vec, sep="__")
# # idx <- split(seq_len(ncol(expr_sc)), key)
# # PB_counts <- sapply(idx, function(cols) Matrix::rowSums(expr_sc[, cols, drop=FALSE]))
# # PB_counts <- ensure_double_matrix(as.matrix(PB_counts))
# # PB_cpm <- cpm_mat(PB_counts)
# # pb_path <- file.path(OUT_DIR, "pseudobulk_donor_level.tsv")
# # data.table::fwrite(data.frame(gene = rownames(PB_cpm), PB_cpm, check.names = FALSE),
# #                    pb_path, sep = "\t")
# # msg("Saved donor-level pseudobulk → ", pb_path)

# # keep_cells <- !is.na(ct_vec) & nzchar(ct_vec)
# # if (any(!keep_cells)) {
# #   msg("Dropping ", sum(!keep_cells), " cells with missing/empty cell_type.")
# #   sc.eset <- sc.eset[, keep_cells]
# #   expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))
# #   ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# # }

# # expr <- ensure_double_matrix(Biobase::exprs(sc.eset))        # genes × cells
# # lib  <- colSums(expr); lib[lib == 0] <- 1
# # expr_cpm <- ensure_double_matrix(t(1e6 * t(expr) / lib))
# # idxs <- split(seq_along(ct_vec), ct_vec)
# # Z <- sapply(idxs, function(cols) rowMeans(expr_cpm[, cols, drop=FALSE], na.rm=TRUE))
# # Z <- ensure_double_matrix(as.matrix(Z))
# # colnames(Z) <- make.names(colnames(Z), unique=TRUE)
# # Z[!is.finite(Z)] <- 0
# # keep_gene_sig <- rowSums(Z) > 0
# # if (any(!keep_gene_sig)) {
# #   msg("Signature: dropping ", sum(!keep_gene_sig), " genes with all-zero across cell types.")
# #   Z <- Z[keep_gene_sig, , drop=FALSE]
# # }
# # sig_df <- data.frame(gene = rownames(Z), Z, check.names = FALSE)
# # sig_path <- file.path(OUT_DIR, "signature_matrix.tsv")
# # data.table::fwrite(sig_df, sig_path, sep="\t")
# # msg("Signature matrix saved: ", sig_path)

# # ## ===================== BISQUE (>=2 subjects) or NNLS (1) ====================
# # n_subjects <- length(unique(Biobase::pData(sc.eset)$SubjectName))
# # msg("Single-cell unique subjects detected: ", n_subjects)

# # if (n_subjects >= 2) {
# #   genes_use <- intersect(rownames(sc.eset), rownames(bulk.mat))
# #   bulk_use  <- ensure_double_matrix(bulk.mat[genes_use, , drop=FALSE])
# #   bulk_use  <- drop_zero_expr_samples(bulk_use, label="bulk (pre-Bisque)")
# #   bulk.eset <- Biobase::ExpressionSet(assayData = bulk_use)
# #   msg("Running BisqueRNA::ReferenceBasedDecomposition ...")
# #   res <- BisqueRNA::ReferenceBasedDecomposition(
# #     bulk.eset     = bulk.eset,
# #     sc.eset       = sc.eset,
# #     markers       = NULL,
# #     cell.types    = "cellType",
# #     subject.names = "SubjectName",
# #     use.overlap   = FALSE,
# #     verbose       = TRUE,
# #     old.cpm       = TRUE
# #   )
# #   bulk_props <- ensure_double_matrix(res$bulk.props)
# # } else {
# #   msg("Fallback: only ONE subject in scRNA-seq; running NNLS on signature matrix.")
# #   common_genes <- intersect(rownames(Z), rownames(bulk.mat))
# #   if (length(common_genes) < WARN_MIN_OVERLAP)
# #     msg("WARNING: low overlap gene count for NNLS (", length(common_genes), ").")
# #   A <- ensure_double_matrix(Z[common_genes, , drop=FALSE])        # genes × CT
# #   B <- ensure_double_matrix(bulk.mat[common_genes, , drop=FALSE]) # genes × samples
# #   A[!is.finite(A)] <- 0; B[!is.finite(B)] <- 0
# #   keep_gene <- rowSums(A) > 0
# #   if (any(!keep_gene)) { msg("NNLS: dropping ", sum(!keep_gene), " genes with zero signature.");
# #     A <- A[keep_gene, , drop=FALSE]; B <- B[keep_gene, , drop=FALSE] }
# #   keep_ct <- colSums(A) > 0
# #   if (any(!keep_ct)) { msg("NNLS: dropping ", sum(!keep_ct), " CT with zero signature.");
# #     A <- A[, keep_ct, drop=FALSE] }
# #   if (ncol(A) == 0) stop("NNLS: no non-zero cell types remain after filtering.")
# #   P_nnls <- matrix(NA_real_, nrow=ncol(A), ncol=ncol(B),
# #                    dimnames=list(colnames(A), colnames(B)))
# #   for (j in seq_len(ncol(B))) {
# #     fit <- nnls::nnls(A, B[, j]); p <- as.numeric(fit$x); p[!is.finite(p)] <- 0
# #     s <- sum(p); if (is.finite(s) && s > 0) p <- p / s
# #     P_nnls[, j] <- p
# #   }
# #   bulk_props <- ensure_double_matrix(P_nnls)
# #   res <- list(bulk.props = bulk_props, genes.used = rownames(A))
# # }

# # ## =============================== OUTPUTS ====================================
# # prop_sums  <- colSums(bulk_props)
# # bad_sum    <- which(abs(prop_sums - 1) > MAX_PROP_DEVIATION)
# # if (length(bad_sum))
# #   msg("WARNING: |sum(props)-1| >", MAX_PROP_DEVIATION, " for samples: ",
# #       paste(colnames(bulk_props)[bad_sum], collapse=", "))

# # # TSV (canonical)
# # data.table::fwrite(
# #   data.frame(cell_type = rownames(bulk_props), bulk_props, check.names = FALSE),
# #   file.path(OUT_DIR, "bisque_bulk_proportions.tsv"), sep = "\t"
# # )
# # # CSV (for older Python code that expects it)
# # utils::write.csv(
# #   data.frame(bulk_props, check.names = FALSE),
# #   file = file.path(OUT_DIR, "Bisque_proportions.csv"),
# #   row.names = TRUE
# # )

# # if (!is.null(res$sc.props)) {
# #   data.table::fwrite(
# #     data.frame(cell_type = rownames(res$sc.props), res$sc.props, check.names = FALSE),
# #     file.path(OUT_DIR, "bisque_sc_proportions.tsv"), sep = "\t"
# #   )
# # }

# # if (!is.null(res$genes.used)) {
# #   writeLines(res$genes.used, file.path(OUT_DIR, "bisque_genes_used.txt"))
# # } else {
# #   writeLines(rownames(bulk.mat), file.path(OUT_DIR, "bisque_genes_used.txt"))
# # }

# # # Bar of sums (open an explicit device)
# # png(filename = file.path(OUT_DIR, "prop_sums.png"), width = 1200, height = 400)
# # par(mar=c(6,4,2,1))
# # barplot(prop_sums, las = 2, main = "Sum of estimated proportions per sample")
# # abline(h = 1, col = "red", lty = 2)
# # safe_dev_off()

# # ## ======================= POST-HOC QC + VISUALS ==============================
# # PROP_FILE <- file.path(OUT_DIR, "bisque_bulk_proportions.tsv")
# # SIG_FILE  <- file.path(OUT_DIR, "signature_matrix.tsv")
# # stopifnot(file.exists(PROP_FILE), file.exists(SIG_FILE))

# # props_dt <- data.table::fread(PROP_FILE)
# # stopifnot("cell_type" %in% names(props_dt))
# # P_raw <- as.matrix(props_dt[, -1, with=FALSE]); rownames(P_raw) <- props_dt$cell_type
# # P_raw <- ensure_double_matrix(P_raw)
# # samples <- colnames(P_raw)

# # sig_dt <- data.table::fread(SIG_FILE)
# # gcol <- if ("gene" %in% names(sig_dt)) "gene" else names(sig_dt)[1]
# # genes <- sig_dt[[gcol]]; sig_dt[[gcol]] <- NULL
# # Z_raw <- ensure_double_matrix(as.matrix(sig_dt)); rownames(Z_raw) <- genes

# # # Harmonize CT labels
# # colnames(Z_raw) <- harmonize_labels(colnames(Z_raw))
# # rownames(P_raw) <- harmonize_labels(rownames(P_raw))
# # if (any(duplicated(colnames(Z_raw))))  stop("Duplicate CTs in signature (Z).")
# # if (any(duplicated(rownames(P_raw))))  stop("Duplicate CTs in proportions (P).")
# # CT <- intersect(colnames(Z_raw), rownames(P_raw))
# # if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d) after harmonization.", length(CT)))
# # Z_use <- ensure_double_matrix(Z_raw[, CT, drop=FALSE])   # genes x CT
# # P_use <- ensure_double_matrix(P_raw[CT,  , drop=FALSE])  # CT x samples

# # # Tidy for plots
# # props_dt_use <- data.frame(cell_type = rownames(P_use), P_use, check.names = FALSE)
# # props_long <- data.table::melt(data.table::as.data.table(props_dt_use), id.vars="cell_type",
# #                                variable.name="sample", value.name="prop")
# # wide <- data.table::dcast(props_long, sample ~ cell_type, value.var="prop")
# # prop_mat <- as.matrix(wide[, -1, with=FALSE]); rownames(prop_mat) <- wide$sample
# # prop_mat <- ensure_double_matrix(prop_mat)

# # has_sample_meta <- !is.null(SAMPLE_META_FILE) && file.exists(SAMPLE_META_FILE)
# # if (has_sample_meta) meta <- data.table::fread(SAMPLE_META_FILE)

# # have_bulk <- file.exists(BULK_COUNTS_FILE)
# # if (have_bulk) {
# #   bulk_dt <- data.table::fread(BULK_COUNTS_FILE)
# #   gcol2 <- intersect(c("gene","symbol","Gene","Gene_Symbol","ID","Name","GeneID","genes"), names(bulk_dt))[1]
# #   if (is.na(gcol2)) gcol2 <- names(bulk_dt)[1]
# #   bgenes <- bulk_dt[[gcol2]]; bulk_dt[[gcol2]] <- NULL
# #   X_bulk <- ensure_double_matrix(as.matrix(bulk_dt)); rownames(X_bulk) <- as.character(bgenes)
# #   X_cpm  <- cpm_mat(X_bulk)
# # }

# # have_sc_props <- !is.null(res$sc.props)
# # if (have_sc_props) {
# #   Ct_raw <- ensure_double_matrix(as.matrix(res$sc.props))
# #   rownames(Ct_raw) <- harmonize_labels(rownames(res$sc.props))
# # }

# # # --- QC: sums table + hist
# # prop_sums <- colSums(P_use, na.rm=TRUE)
# # qc_df <- data.frame(sample=names(prop_sums), sum_props=as.numeric(prop_sums),
# #                     abs_dev=abs(as.numeric(prop_sums)-1))
# # data.table::fwrite(qc_df, file.path(TAB_DIR,"qc_prop_sums.tsv"), sep="\t")

# # g_hist <- ggplot2::ggplot(qc_df, ggplot2::aes(sum_props)) +
# #   ggplot2::geom_histogram(bins=30, alpha=.9) +
# #   ggplot2::geom_vline(xintercept=1, linetype=2, color="red") +
# #   theme_pub() + ggplot2::labs(title="Distribution of proportion sums", x="Sum of proportions", y="Count")
# # ggplot2::ggsave(file.path(PLOT_DIR,"02_qc_sum_hist.png"), g_hist, width=6.2, height=4.5, dpi=200)

# # # --- Stacked composition
# # hc <- hclust(dist(prop_mat), method = "complete")
# # sample_order <- rownames(prop_mat)[hc$order]
# # props_long$sample <- factor(props_long$sample, levels = sample_order)

# # gg1 <- ggplot2::ggplot(props_long, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
# #   ggplot2::geom_bar(stat = "identity", width = 0.95) +
# #   ggplot2::labs(title = "Cell-type composition per sample (cluster-ordered)", x = "Sample", y = "Proportion") +
# #   ggplot2::theme_bw() + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=90, vjust=0.5, hjust=1))

# # ggplot2::ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), gg1, width = 16, height = 7, dpi = 200)

# # # --- Heatmap of proportions
# # pheatmap::pheatmap(prop_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
# #                    filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
# #                    main="Proportions heatmap", width=10, height=13)

# # # --- Violin/box of CT distributions
# # g_boxv <- ggplot2::ggplot(props_long, ggplot2::aes(cell_type, prop, fill=cell_type)) +
# #   ggplot2::geom_violin(trim=FALSE, alpha=.85) + ggplot2::geom_boxplot(width=.12, outlier.size=.4) +
# #   viridis::scale_fill_viridis(discrete=TRUE, guide="none") + theme_pub() +
# #   ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45, hjust=1)) +
# #   ggplot2::labs(title="Distribution of Bisque-estimated proportions", x="Cell type", y="Proportion")

# # ggplot2::ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), g_boxv, width=11, height=6.5, dpi=200)

# # ## =================== OPTIONAL MARKER CHECKS & EXPORTS =======================
# # eps <- 1e-8
# # other_mean <- function(mat, ct) { if (ncol(mat)==1) return(rep(0, nrow(mat))); rowMeans(mat[, setdiff(colnames(mat), ct), drop=FALSE], na.rm=TRUE) }
# # spec_list <- lapply(colnames(Z_use), function(ct){
# #   spec <- log2((Z_use[, ct] + eps) / (other_mean(Z_use, ct) + eps))
# #   data.frame(gene = rownames(Z_use), cell_type = ct, specificity = spec, expr_ct = Z_use[, ct], stringsAsFactors = FALSE)
# # })
# # spec_df <- data.table::rbindlist(spec_list)
# # markers_sel <- spec_df[order(cell_type, -specificity), .SD[1:min(.N, TOP_MARKERS_PER_CT)], by = cell_type]
# # data.table::fwrite(markers_sel, file.path(TAB_DIR, "top_markers_per_celltype.tsv"), sep="\t")

# # if (!is.null(MARKER_FILE) && file.exists(MARKER_FILE) && have_bulk) {
# #   mk <- data.table::fread(MARKER_FILE)
# #   if (nrow(mk)) {
# #     data.table::setnames(mk, old = names(mk), new = tolower(names(mk)))
# #     gene_col <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","feature","id","geneid","name"), names(mk))[1]
# #     ct_col   <- intersect(c("cell_type","celltype","cell_type_label","celltype_label","cluster","label","cell","ct"), names(mk))[1]
# #     if (!is.na(gene_col) && !is.na(ct_col)) {
# #       markers <- mk[, .(gene = get(gene_col), cell_type = harmonize_labels(get(ct_col)))]
# #       markers <- unique(markers[!is.na(gene) & nzchar(gene) & !is.na(cell_type) & nzchar(cell_type)])
# #       bulk_dt0 <- data.table::fread(BULK_COUNTS_FILE)
# #       gcol3 <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","id","name","geneid","genes"), names(bulk_dt0))[1]
# #       if (is.na(gcol3)) gcol3 <- names(bulk_dt0)[1]
# #       bgenes <- bulk_dt0[[gcol3]]; bulk_dt0[[gcol3]] <- NULL
# #       bulk_mat <- ensure_double_matrix(as.matrix(bulk_dt0)); rownames(bulk_mat) <- as.character(bgenes)
# #       bulk_cpm <- cpm_mat(bulk_mat)
# #       ct_ok <- intersect(unique(markers$cell_type), rownames(P_use))
# #       ctab <- lapply(ct_ok, function(ct) {
# #         gs <- intersect(markers[cell_type == ct, gene], rownames(bulk_cpm))
# #         if (!length(gs)) return(NULL)
# #         score <- colMeans(bulk_cpm[gs, , drop = FALSE])
# #         common <- intersect(names(score), colnames(P_use))
# #         if (length(common) < 3) return(NULL)
# #         rho <- suppressWarnings(cor(score[common], P_use[ct, common], method = "spearman"))
# #         data.frame(cell_type = ct, n_genes = length(gs), n_samples = length(common), spearman_rho = rho)
# #       })
# #       ctab <- data.table::rbindlist(ctab, fill = TRUE)
# #       if (!is.null(ctab) && nrow(ctab)) data.table::fwrite(ctab, file.path(TAB_DIR, "marker_prop_spearman.tsv"), sep = "\t")
# #     }
# #   }
# # }

# # # ====================== S3 ADD-ONS: OVERVIEW + SAMPLE CARDS ===================
# # # (Safe: only runs if objects exist; otherwise writes placeholders.)
# # safe_ggsave <- function(path, plot=NULL, width=8, height=5, dpi=200, title_when_empty=NULL){
# #   if (!is.null(plot)) {
# #     ggplot2::ggsave(path, plot, width=width, height=height, dpi=dpi)
# #   } else {
# #     png(path, width=width*100, height=height*100)
# #     par(mar=c(2,2,2,2)); plot.new()
# #     ttl <- if (is.null(title_when_empty)) basename(path) else title_when_empty
# #     title(ttl); mtext("No data to plot", side=3, line=-1, col="gray40")
# #     dev.off()
# #   }
# # }

# # dir.create(file.path(PLOT_DIR, "S3_sample_cards"), showWarnings = FALSE, recursive = TRUE)

# # # ---------- S3 overview: overall composition pie ----------
# # overall_pie <- NULL
# # if (exists("props_long") && nrow(props_long)) {
# #   dfpie <- props_long[, .(prop = mean(prop, na.rm=TRUE)), by=cell_type]
# #   dfpie <- dfpie[prop > 0]
# #   if (nrow(dfpie)) {
# #     overall_pie <- ggplot2::ggplot(dfpie, ggplot2::aes(x="", y=prop, fill=cell_type)) +
# #       ggplot2::geom_bar(stat="identity", width=0.9) +
# #       ggplot2::coord_polar(theta="y") +
# #       viridis::scale_fill_viridis(discrete=TRUE, guide=NULL) +
# #       ggplot2::labs(title="Overall mean composition (across samples)", y=NULL, x=NULL) +
# #       ggplot2::theme_void() +
# #       ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold"))
# #   }
# # }
# # safe_ggsave(file.path(PLOT_DIR,"S3_overview_composition_pie.png"), overall_pie, 7, 7,
# #             title_when_empty="Overall mean composition (pie)")

# # # ---------- S3 overview: CT prevalence across samples ----------
# # ct_prev_plot <- NULL
# # if (exists("P_use") && ncol(P_use) > 0) {
# #   prev <- data.table::as.data.table(P_use)
# #   prev[, cell_type := rownames(P_use)]
# #   prev_long <- data.table::melt(prev, id.vars="cell_type", variable.name="sample", value.name="prop")
# #   prev_ct <- prev_long[, .(prevalence = sum(prop > 0, na.rm=TRUE)), by=cell_type]
# #   if (nrow(prev_ct)) {
# #     ct_prev_plot <- ggplot2::ggplot(prev_ct, ggplot2::aes(reorder(cell_type, prevalence), prevalence)) +
# #       ggplot2::geom_col() + ggplot2::coord_flip() +
# #       ggplot2::labs(title="CT prevalence (#samples with prop>0)", x="Cell type", y="#Samples") +
# #       ggplot2::theme_bw()
# #   }
# # }
# # safe_ggsave(file.path(PLOT_DIR,"S3_celltype_prevalence.png"), ct_prev_plot, 8, 7,
# #             title_when_empty="CT prevalence")

# # # ---------- S3 overview: sample × sample correlation of compositions ----------
# # corr_plot_path <- file.path(PLOT_DIR, "S3_props_correlation_heatmap.png")
# # if (exists("prop_mat") && nrow(prop_mat) > 1 && ncol(prop_mat) > 1) {
# #   # rows = samples, cols = CTs
# #   cm <- suppressWarnings(cor(t(prop_mat), method="spearman", use="pairwise.complete.obs"))
# #   cm[!is.finite(cm)] <- 0
# #   tryCatch({
# #     pheatmap::pheatmap(cm, filename=corr_plot_path, main="Sample×Sample Spearman correlation (compositions)",
# #                        cluster_rows=TRUE, cluster_cols=TRUE, width=10, height=10)
# #   }, error=function(e){
# #     safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty=paste("Correlation heatmap:", e$message))
# #   })
# # } else {
# #   safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty="Correlation heatmap (insufficient data)")
# # }

# # # ---------- S3 overview: PCA of compositions ----------
# # pca_plot <- NULL
# # if (exists("prop_mat") && nrow(prop_mat) >= 2 && ncol(prop_mat) >= 2) {
# #   X <- prop_mat
# #   X <- X[, colSums(is.finite(X)) == nrow(X), drop=FALSE]
# #   if (ncol(X) >= 2) {
# #     pc <- tryCatch(prcomp(X, center=TRUE, scale.=TRUE), error=function(e) NULL)
# #     if (!is.null(pc)) {
# #       sc <- data.frame(PC1=pc$x[,1], PC2=pc$x[,2], sample=rownames(pc$x))
# #       pca_plot <- ggplot2::ggplot(sc, ggplot2::aes(PC1, PC2, label=sample)) +
# #         ggplot2::geom_point() + ggrepel::geom_text_repel(size=3, max.overlaps=40) +
# #         ggplot2::theme_bw() + ggplot2::labs(title="PCA of sample compositions")
# #     }
# #   }
# # }
# # safe_ggsave(file.path(PLOT_DIR,"S3_props_PCA.png"), pca_plot, 8, 6,
# #             title_when_empty="PCA of compositions (insufficient data)")

# # # ---------- S1 sanity: mean signature per CT ----------
# # sig_mean_plot <- NULL
# # if (exists("Z_use") && ncol(Z_use) >= 1) {
# #   means <- colMeans(Z_use, na.rm=TRUE)
# #   dfm <- data.frame(cell_type=names(means), mean_expr=as.numeric(means))
# #   sig_mean_plot <- ggplot2::ggplot(dfm, ggplot2::aes(reorder(cell_type, mean_expr), mean_expr)) +
# #     ggplot2::geom_col() + ggplot2::coord_flip() +
# #     ggplot2::theme_bw() + ggplot2::labs(title="Mean signature per CT (CPM)", x="Cell type", y="Mean CPM")
# # }
# # safe_ggsave(file.path(PLOT_DIR,"S1_signature_ct_means.png"), sig_mean_plot, 7, 6,
# #             title_when_empty="Mean signature per CT")

# # # ---------- S3: per-sample QC TSV + SAMPLE CARDS (PNG) ----------
# # qc_out <- file.path(TAB_DIR, "S3_sample_qc.tsv")
# # qc_df2 <- data.frame()
# # if (exists("P_use") && ncol(P_use) > 0) {
# #   prop_sums_local <- colSums(P_use, na.rm=TRUE)
# #   abs_dev <- abs(prop_sums_local - 1)
# #   qc_df2 <- data.frame(sample = names(prop_sums_local),
# #                        sum_props = as.numeric(prop_sums_local),
# #                        abs_dev = as.numeric(abs_dev))
# #   data.table::fwrite(qc_df2, qc_out, sep="\t")
# # } else {
# #   data.table::fwrite(data.frame(), qc_out, sep="\t")
# # }

# # # Helper: top genes plot for a sample (from X_cpm, if present)
# # top_genes_plot <- function(sample_id, topN=10){
# #   if (!exists("X_cpm") || !is.matrix(X_cpm) || !(sample_id %in% colnames(X_cpm))) return(NULL)
# #   v <- X_cpm[, sample_id]; if (all(!is.finite(v))) return(NULL)
# #   ord <- order(v, decreasing=TRUE); k <- min(topN, length(ord))
# #   if (k < 1) return(NULL)
# #   sel <- ord[seq_len(k)]
# #   df <- data.frame(gene=names(v)[sel], CPM=as.numeric(v[sel]))
# #   ggplot2::ggplot(df, ggplot2::aes(reorder(gene, CPM), CPM)) +
# #     ggplot2::geom_col() + ggplot2::coord_flip() +
# #     ggplot2::theme_bw() + ggplot2::labs(title=paste0("Top ", k, " genes in ", sample_id), x=NULL, y="CPM")
# # }

# # # Helper: per-sample composition stacked bar (single sample)
# # sample_comp_plot <- function(sample_id){
# #   if (!exists("props_long") || !nrow(props_long)) return(NULL)
# #   dl <- props_long[sample == sample_id]
# #   if (!nrow(dl)) return(NULL)
# #   ggplot2::ggplot(dl, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
# #     ggplot2::geom_bar(stat="identity", width=0.8) +
# #     viridis::scale_fill_viridis(discrete=TRUE, guide="none") +
# #     ggplot2::theme_bw() + ggplot2::labs(title=paste0("Composition – ", sample_id), x=NULL, y="Proportion")
# # }

# # # Helper: sample QC tile
# # sample_qc_tile <- function(sample_id){
# #   sp <- tryCatch(qc_df2[qc_df2$sample == sample_id, , drop=FALSE], error=function(e) NULL)
# #   if (is.null(sp) || !nrow(sp)) return(NULL)
# #   lbl <- sprintf("sum(props)=%.3f\n|sum-1|=%.3f", sp$sum_props[1], sp$abs_dev[1])
# #   ggplot2::ggplot(data.frame(x=1,y=1,label=lbl), ggplot2::aes(x,y,label=label)) +
# #     ggplot2::geom_text(size=5) + ggplot2::theme_void() +
# #     ggplot2::labs(title="QC")
# # }

# # # Render sample cards
# # if (exists("prop_mat") && nrow(prop_mat) >= 1) {
# #   smpls <- rownames(prop_mat)
# #   for (s in smpls) {
# #     p1 <- sample_comp_plot(s)
# #     p2 <- top_genes_plot(s, topN = max(5, min(10, if (exists("TOP_GENES_PER_SAMPLE")) TOP_GENES_PER_SAMPLE else 10)))
# #     p3 <- sample_qc_tile(s)
# #     # Arrange with patchwork if at least one plot exists
# #     if (!is.null(p1) || !is.null(p2) || !is.null(p3)) {
# #       card <- NULL
# #       if (!is.null(p1) && !is.null(p2) && !is.null(p3)) card <- p1 + p2 + p3 + patchwork::plot_layout(ncol=3, widths=c(1.2,1,0.6))
# #       else if (!is.null(p1) && !is.null(p2)) card <- p1 + p2 + patchwork::plot_layout(ncol=2, widths=c(1.2,1))
# #       else if (!is.null(p1)) card <- p1
# #       else if (!is.null(p2)) card <- p2
# #       else card <- p3
# #       outp <- file.path(PLOT_DIR, "S3_sample_cards", paste0(s, ".png"))
# #       safe_ggsave(outp, card, width=15, height=5, dpi=200, title_when_empty=paste("Sample card:", s))
# #     } else {
# #       safe_ggsave(file.path(PLOT_DIR, "S3_sample_cards", paste0(s, ".png")), NULL, 12, 4,
# #                   title_when_empty=paste("Sample card:", s))
# #     }
# #   }
# # } else {
# #   # No compositions: still produce a note
# #   safe_ggsave(file.path(PLOT_DIR, "S3_sample_cards", "no_samples.png"), NULL, 10, 3,
# #               title_when_empty="No sample compositions available for cards")
# # }
# # # ==================== END S3 ADD-ONS =========================================

# # msg("DONE. Outputs written to: ", OUT_DIR)
# # msg("Tables → ", TAB_DIR)
# # msg("Plots  → ", PLOT_DIR)


# #!/usr/bin/env Rscript

# options(stringsAsFactors = FALSE)
# set.seed(42L)

# # ============================ OPTIONS =======================================
# # Environment variables set by Python
# get_bool  <- function(x, default = FALSE) {
#   v <- toupper(Sys.getenv(x, ifelse(default, "TRUE", "FALSE"))); v %in% c("1","T","TRUE","YES","Y")
# }
# get_int   <- function(x, default = NA_integer_) {
#   v <- suppressWarnings(as.integer(Sys.getenv(x, ""))); if (is.na(v)) default else v
# }
# get_num   <- function(x, default = NA_real_) {
#   v <- suppressWarnings(as.numeric(Sys.getenv(x, ""))); if (is.na(v)) default else v
# }
# get_str   <- function(x, default = "") {
#   v <- Sys.getenv(x, ""); if (!nzchar(v)) default else v
# }

# REF_DIR_PY            <- get_str("REF_DIR_PY", "")
# OUT_DIR_PY            <- get_str("OUT_DIR_PY", "")
# DROP_MITO_PY          <- get_bool("DROP_MITO_PY", FALSE)
# ALLOW_DUP_PY          <- get_bool("ALLOW_DUP_PY", FALSE)
# WARN_MIN_OVERLAP_PY   <- get_int ("WARN_MIN_OVERLAP_PY", 100L)
# MAX_PROP_DEV_PY       <- get_num ("MAX_PROP_DEV_PY", 0.2)
# TOP_MARKERS_PER_CT_PY <- get_int ("TOP_MARKERS_PER_CT_PY", 10L)
# TOP_GENES_PER_SAMPLE_PY <- get_int("TOP_GENES_PER_SAMPLE_PY", 50L)
# AUTO_INSTALL_PY       <- get_bool("AUTO_INSTALL_PY", TRUE)

# if (!nzchar(REF_DIR_PY)) stop("Environment variable REF_DIR_PY not set; Python must pass it to R.")

# # ============================ UTILITIES =====================================
# msg <- function(...) cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] "), ..., "\n")

# safe_dev_off <- function() {
#   if (names(dev.cur()) != "null device") try(dev.off(), silent = TRUE)
# }

# ensure_double_matrix <- function(m) {
#   if (isS4(m)) m <- as.matrix(m)
#   if (!is.matrix(m)) m <- as.matrix(m)
#   storage.mode(m) <- "double"; m
# }

# is_all_numeric_vec <- function(x) all(grepl("^[0-9]+$", as.character(x)))
# norm_bc <- function(x){ x <- trimws(as.character(x)); x <- gsub("^X(?=\\d)","",x,perl=TRUE); x <- gsub("\\.", "-", x); x }
# drop_mito <- function(genes) genes[!grepl("^MT-", toupper(genes))]
# zero_var_rows <- function(m) { v <- apply(m, 1, function(x) var(x, na.rm=TRUE)); names(which(is.na(v) | v == 0)) }

# collapse_dups <- function(m) {
#   m <- ensure_double_matrix(m)
#   if (is.null(rownames(m))) stop("Matrix lacks rownames.")
#   if (!any(duplicated(rownames(m)))) return(m)
#   msg("Duplicate gene IDs detected; collapsing by sum.")
#   ux <- unique(rownames(m)); idx <- match(rownames(m), ux)
#   sp <- Matrix::sparseMatrix(i = idx, j = seq_len(nrow(m)), x = 1,
#                              dims = c(length(ux), nrow(m)))
#   out <- ensure_double_matrix(as.matrix(sp %*% m))
#   rownames(out) <- ux; colnames(out) <- colnames(m)
#   out
# }

# drop_zero_expr_samples <- function(m, label="bulk") {
#   m <- ensure_double_matrix(m)
#   cs <- colSums(m, na.rm=TRUE); zc <- which(!is.finite(cs) | cs <= 0)
#   if (length(zc)) {
#     msg("Dropping ", length(zc), " ", label, " sample(s) with zero/NA expr: ",
#         paste(colnames(m)[zc], collapse=", "))
#     m <- m[, -zc, drop=FALSE]
#   }
#   if (ncol(m) == 0) stop("All ", label, " samples dropped; cannot proceed.")
#   m
# }

# theme_pub <- function(){
#   ggplot2::theme_bw(base_size=12) +
#     ggplot2::theme(plot.title=ggplot2::element_text(face="bold"),
#                    panel.grid.minor=ggplot2::element_blank())
# }

# cpm_mat <- function(m){
#   m <- ensure_double_matrix(m)
#   lib <- colSums(m,na.rm=TRUE); lib[!is.finite(lib)|lib==0] <- 1
#   ensure_double_matrix(t(1e6*t(m)/lib))
# }

# harmonize_labels <- function(x) make.names(trimws(as.character(x)), unique = FALSE)

# # =========================== PATHS & PARAMS =================================
# REF_DIR <- REF_DIR_PY
# OUT_DIR_DEFAULT <- file.path(REF_DIR, "bisque_deconv-results")
# OUT_DIR <- if (!is.null(OUT_DIR_PY) && nzchar(OUT_DIR_PY)) OUT_DIR_PY else OUT_DIR_DEFAULT
# dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# DROP_MITO_GENES       <- isTRUE(DROP_MITO_PY)
# ALLOW_DUPLICATE_GENES <- isTRUE(ALLOW_DUP_PY)
# WARN_MIN_OVERLAP      <- as.integer(WARN_MIN_OVERLAP_PY)
# MAX_PROP_DEVIATION    <- as.numeric(MAX_PROP_DEV_PY)
# TOP_MARKERS_PER_CT    <- as.integer(TOP_MARKERS_PER_CT_PY)
# TOP_GENES_PER_SAMPLE  <- as.integer(TOP_GENES_PER_SAMPLE_PY)
# AUTO_INSTALL          <- isTRUE(AUTO_INSTALL_PY)

# SC_COUNTS_FILE   <- file.path(REF_DIR, "sc_counts.tsv")
# SC_META_FILE     <- file.path(REF_DIR, "sc_metadata.tsv")
# BULK_COUNTS_FILE <- file.path(REF_DIR, "bulk_counts_overlap.tsv")

# SAMPLE_META_FILE <- file.path(REF_DIR, "sample_metadata_patientAKI.tsv")  # optional
# MARKER_FILE      <- file.path(REF_DIR, "marker_gene_table.tsv")           # optional

# PLOT_DIR <- file.path(OUT_DIR, "deconv_visual_reports"); dir.create(PLOT_DIR, FALSE, TRUE)
# TAB_DIR  <- file.path(OUT_DIR, "QC & Performance Metrics"); dir.create(TAB_DIR,  FALSE, TRUE)

# # ============================ PACKAGES ======================================
# cran_pkgs <- c("data.table","Matrix","tools","nnls","ggplot2","viridis",
#                "pheatmap","corrplot","dplyr","tidyr","patchwork","ggrepel")
# bio_pkgs  <- c("Biobase","BisqueRNA")

# need_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
# need_bio  <- bio_pkgs[!vapply(bio_pkgs,  requireNamespace, logical(1), quietly = TRUE)]

# if (AUTO_INSTALL) {
#   if (length(need_cran)) {
#     msg("Installing CRAN packages: ", paste(need_cran, collapse=", "))
#     install.packages(need_cran, repos = "https://cloud.r-project.org")
#   }
#   if (!requireNamespace("BiocManager", quietly = TRUE)) {
#     install.packages("BiocManager", repos = "https://cloud.r-project.org")
#   }
#   if (length(need_bio)) {
#     msg("Installing Bioconductor packages: ", paste(need_bio, collapse=", "))
#     BiocManager::install(need_bio, update = FALSE, ask = FALSE)
#   }
# } else if (length(need_cran) || length(need_bio)) {
#   stop("Missing R packages: ", paste(c(need_cran, need_bio), collapse=", "),
#        "\nRun again with auto_install=TRUE or install manually.")
# }

# suppressPackageStartupMessages({
#   lapply(c(cran_pkgs, bio_pkgs), library, character.only = TRUE)
# })
# capture.output(sessionInfo(), file = file.path(OUT_DIR, "R_sessionInfo.txt"))

# # ======================= LOAD SINGLE-CELL DATA ==============================
# stopifnot(file.exists(SC_COUNTS_FILE), file.exists(SC_META_FILE))
# msg("Loading single-cell counts: ", SC_COUNTS_FILE)
# cnt_full <- data.table::fread(SC_COUNTS_FILE, sep="\t", header=TRUE, data.table=FALSE)

# # First column must be 'gene'; if not, treat the first as gene column anyway.
# genes <- as.character(cnt_full[[1]]); cnt_full[[1]] <- NULL
# m_sc <- ensure_double_matrix(as.matrix(cnt_full))
# rownames(m_sc) <- genes
# counts_bc_raw <- colnames(cnt_full)

# msg("Loading metadata: ", SC_META_FILE)
# md <- data.table::fread(SC_META_FILE, sep="\t", header=TRUE, data.table=FALSE)
# if (!"cell_type" %in% names(md))     md$cell_type     <- NA_character_
# if (!"individual_id" %in% names(md)) md$individual_id <- "ONE_SUBJECT"
# if (!"cell_barcode" %in% names(md))  md$cell_barcode  <- NA_character_

# counts_bc <- norm_bc(counts_bc_raw); meta_bc <- norm_bc(md$cell_barcode)
# if (nrow(md) != length(counts_bc)) {
#   stop(sprintf("Metadata rows (%d) != number of cells in counts (%d). Re-export so sizes match.",
#                nrow(md), length(counts_bc)))
# }
# if (length(na.omit(unique(meta_bc))) <= 2 || is_all_numeric_vec(na.omit(meta_bc))) {
#   md$cell_barcode <- counts_bc; meta_bc <- counts_bc
# }
# if (is_all_numeric_vec(counts_bc) && is_all_numeric_vec(meta_bc)) {
#   msg("Counts & metadata both numeric; generating synthetic barcodes (in-memory).")
#   syn <- sprintf("C%06d", seq_len(length(counts_bc)))
#   colnames(m_sc) <- syn; md$cell_barcode <- syn
#   counts_bc <- syn; meta_bc <- syn
# } else {
#   colnames(m_sc) <- counts_bc
# }

# # Align (transpose rescue + synthetic fallback)
# ov <- intersect(colnames(m_sc), md$cell_barcode)
# if (!length(ov) && any(rownames(m_sc) %in% md$cell_barcode)) {
#   msg("Counts appear transposed; transposing ...")
#   m_sc <- ensure_double_matrix(t(m_sc))
#   ov <- intersect(colnames(m_sc), md$cell_barcode)
# }
# if (!length(ov)) {
#   syn <- sprintf("C%06d", seq_len(ncol(m_sc)))
#   colnames(m_sc) <- syn; md$cell_barcode <- syn; ov <- syn
# }
# m_sc <- m_sc[, ov, drop=FALSE]
# md <- md[match(ov, md$cell_barcode), , drop=FALSE]
# rownames(md) <- md$cell_barcode

# # Build ExpressionSet for sc
# pdat <- new("AnnotatedDataFrame", data = data.frame(
#   cellType    = md$cell_type,
#   SubjectName = md$individual_id,
#   row.names   = rownames(md),
#   check.names = FALSE
# ))
# sc.eset <- Biobase::ExpressionSet(assayData = ensure_double_matrix(m_sc), phenoData = pdat)

# # ============================== LOAD BULK ===================================
# stopifnot(file.exists(BULK_COUNTS_FILE))
# read_bulk <- function(path) {
#   dt <- data.table::fread(path, sep="\t")
#   cand <- c("gene","genes","gene_id","geneid","ensembl","ensembl_id","symbol",
#             "gene_symbol","hgnc","Gene","GeneID","Gene_Symbol","Gene.Name","ID","Name")
#   gcol <- intersect(cand, names(dt))[1]
#   if (is.na(gcol)) {
#     if (!is.numeric(dt[[1]]) && length(unique(dt[[1]])) == nrow(dt)) gcol <- names(dt)[1]
#     else stop("Cannot detect gene column in bulk file.")
#   }
#   genes <- as.character(dt[[gcol]]); dt[[gcol]] <- NULL
#   m <- ensure_double_matrix(as.matrix(dt))
#   rownames(m) <- genes
#   msg("Bulk matrix dims (genes x samples): ", nrow(m), " x ", ncol(m))
#   m
# }
# bulk.mat <- read_bulk(BULK_COUNTS_FILE)

# # ============================ SANITY + FILTERS ==============================
# stopifnot(ncol(Biobase::exprs(sc.eset)) == nrow(Biobase::pData(sc.eset)))

# if (any(Biobase::exprs(sc.eset) < 0, na.rm=TRUE)) stop("Negative values in single-cell counts.")
# if (any(bulk.mat < 0, na.rm=TRUE))                stop("Negative values in bulk counts.")

# if (ALLOW_DUPLICATE_GENES) {
#   msg("Collapsing duplicate genes (safe rebuild)...")
#   expr_new <- collapse_dups(Biobase::exprs(sc.eset))
#   sc.eset <- Biobase::ExpressionSet(
#     assayData = expr_new,
#     phenoData = Biobase::phenoData(sc.eset)
#   )
#   bulk.mat <- collapse_dups(bulk.mat)
# }

# if (DROP_MITO_GENES) {
#   keep_sc   <- drop_mito(rownames(sc.eset));    sc.eset  <- sc.eset[keep_sc, ]
#   keep_bulk <- drop_mito(rownames(bulk.mat));   bulk.mat <- bulk.mat[keep_bulk, , drop=FALSE]
# }

# zv <- zero_var_rows(Biobase::exprs(sc.eset))
# if (length(zv)) {
#   sc.eset <- sc.eset[setdiff(rownames(Biobase::exprs(sc.eset)), zv), ]
#   msg("Removed ", length(zv), " zero-variance genes from sc.")
# }

# bulk_zero_rows <- which(rowSums(bulk.mat == 0, na.rm=TRUE) == ncol(bulk.mat))
# if (length(bulk_zero_rows)) {
#   bulk.mat <- bulk.mat[-bulk_zero_rows, , drop=FALSE]
#   msg("Removed ", length(bulk_zero_rows), " unexpressed genes from bulk (rows).")
# }

# # Ensure unique & ordered gene set
# gene_set <- intersect(rownames(Biobase::exprs(sc.eset)), rownames(bulk.mat))
# gene_set <- unique(gene_set)

# if (length(gene_set) < WARN_MIN_OVERLAP) {
#   msg("WARNING: low overlap gene count (", length(gene_set), "). Results may be less stable.")
# }
# if (length(gene_set) < 2L) {
#   stop(sprintf("Too few overlapping genes after prefilters (|genes| = %d).", length(gene_set)))
# }

# # ExpressionSet-aware subsetting
# sc.eset  <- sc.eset[gene_set, ]
# bulk.mat <- ensure_double_matrix(bulk.mat[gene_set, , drop=FALSE])

# # Dim sanity checks
# msg(sprintf("Dims after alignment: sc.eset %d genes × %d cells; bulk %d genes × %d samples",
#             nrow(Biobase::exprs(sc.eset)), ncol(Biobase::exprs(sc.eset)),
#             nrow(bulk.mat), ncol(bulk.mat)))

# # Drop bulk samples that are zero over the selected genes
# bulk.mat <- drop_zero_expr_samples(bulk.mat, label="bulk")

# # ============ DONOR-LEVEL PSEUDOBULK (CPM) & SIGNATURE MATRIX ===============
# msg("Building donor-level pseudobulk (CPM) ...")
# expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))     # genes × cells
# ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# subj_vec<- as.character(Biobase::pData(sc.eset)$SubjectName)
# key <- paste(subj_vec, ct_vec, sep="__")
# idx <- split(seq_len(ncol(expr_sc)), key)
# PB_counts <- sapply(idx, function(cols) Matrix::rowSums(expr_sc[, cols, drop=FALSE]))
# PB_counts <- ensure_double_matrix(as.matrix(PB_counts))
# PB_cpm <- cpm_mat(PB_counts)
# pb_path <- file.path(OUT_DIR, "pseudobulk_donor_level.tsv")
# data.table::fwrite(data.frame(gene = rownames(PB_cpm), PB_cpm, check.names = FALSE),
#                    pb_path, sep = "\t")
# msg("Saved donor-level pseudobulk → ", pb_path)

# keep_cells <- !is.na(ct_vec) & nzchar(ct_vec)
# if (any(!keep_cells)) {
#   msg("Dropping ", sum(!keep_cells), " cells with missing/empty cell_type.")
#   sc.eset <- sc.eset[, keep_cells]
#   expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))
#   ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# }

# expr <- ensure_double_matrix(Biobase::exprs(sc.eset))        # genes × cells
# lib  <- colSums(expr); lib[lib == 0] <- 1
# expr_cpm <- ensure_double_matrix(t(1e6 * t(expr) / lib))
# idxs <- split(seq_along(ct_vec), ct_vec)
# Z <- sapply(idxs, function(cols) rowMeans(expr_cpm[, cols, drop=FALSE], na.rm=TRUE))
# Z <- ensure_double_matrix(as.matrix(Z))
# colnames(Z) <- make.names(colnames(Z), unique=TRUE)
# Z[!is.finite(Z)] <- 0
# keep_gene_sig <- rowSums(Z) > 0
# if (any(!keep_gene_sig)) {
#   msg("Signature: dropping ", sum(!keep_gene_sig), " genes with all-zero across cell types.")
#   Z <- Z[keep_gene_sig, , drop=FALSE]
# }
# sig_df <- data.frame(gene = rownames(Z), Z, check.names = FALSE)
# sig_path <- file.path(OUT_DIR, "signature_matrix.tsv")
# data.table::fwrite(sig_df, sig_path, sep="\t")
# msg("Signature matrix saved: ", sig_path)

# # ===================== BISQUE (>=2 subjects) or NNLS (1) ====================
# n_subjects <- length(unique(Biobase::pData(sc.eset)$SubjectName))
# msg("Single-cell unique subjects detected: ", n_subjects)

# if (n_subjects >= 2) {
#   genes_use <- intersect(rownames(sc.eset), rownames(bulk.mat))
#   bulk_use  <- ensure_double_matrix(bulk.mat[genes_use, , drop=FALSE])
#   bulk_use  <- drop_zero_expr_samples(bulk_use, label="bulk (pre-Bisque)")
#   bulk.eset <- Biobase::ExpressionSet(assayData = bulk_use)
#   msg("Running BisqueRNA::ReferenceBasedDecomposition ...")
#   res <- BisqueRNA::ReferenceBasedDecomposition(
#     bulk.eset     = bulk.eset,
#     sc.eset       = sc.eset,
#     markers       = NULL,
#     cell.types    = "cellType",
#     subject.names = "SubjectName",
#     use.overlap   = FALSE,
#     verbose       = TRUE,
#     old.cpm       = TRUE
#   )
#   bulk_props <- ensure_double_matrix(res$bulk.props)
# } else {
#   msg("Fallback: only ONE subject in scRNA-seq; running NNLS on signature matrix.")
#   common_genes <- intersect(rownames(Z), rownames(bulk.mat))
#   if (length(common_genes) < WARN_MIN_OVERLAP)
#     msg("WARNING: low overlap gene count for NNLS (", length(common_genes), ").")
#   A <- ensure_double_matrix(Z[common_genes, , drop=FALSE])        # genes × CT
#   B <- ensure_double_matrix(bulk.mat[common_genes, , drop=FALSE]) # genes × samples
#   A[!is.finite(A)] <- 0; B[!is.finite(B)] <- 0
#   keep_gene <- rowSums(A) > 0
#   if (any(!keep_gene)) { msg("NNLS: dropping ", sum(!keep_gene), " genes with zero signature.");
#     A <- A[keep_gene, , drop=FALSE]; B <- B[keep_gene, , drop=FALSE] }
#   keep_ct <- colSums(A) > 0
#   if (any(!keep_ct)) { msg("NNLS: dropping ", sum(!keep_ct), " CT with zero signature.");
#     A <- A[, keep_ct, drop=FALSE] }
#   if (ncol(A) == 0) stop("NNLS: no non-zero cell types remain after filtering.")
#   P_nnls <- matrix(NA_real_, nrow=ncol(A), ncol=ncol(B),
#                    dimnames=list(colnames(A), colnames(B)))
#   for (j in seq_len(ncol(B))) {
#     fit <- nnls::nnls(A, B[, j]); p <- as.numeric(fit$x); p[!is.finite(p)] <- 0
#     s <- sum(p); if (is.finite(s) && s > 0) p <- p / s
#     P_nnls[, j] <- p
#   }
#   bulk_props <- ensure_double_matrix(P_nnls)
#   res <- list(bulk.props = bulk_props, genes.used = rownames(A))
# }

# # =============================== OUTPUTS ====================================
# prop_sums  <- colSums(bulk_props)
# bad_sum    <- which(abs(prop_sums - 1) > MAX_PROP_DEVIATION)
# if (length(bad_sum))
#   msg("WARNING: |sum(props)-1| >", MAX_PROP_DEVIATION, " for samples: ",
#       paste(colnames(bulk_props)[bad_sum], collapse=", "))

# # TSV (canonical)
# data.table::fwrite(
#   data.frame(cell_type = rownames(bulk_props), bulk_props, check.names = FALSE),
#   file.path(OUT_DIR, "bisque_bulk_proportions.tsv"), sep = "\t"
# )
# # CSV (legacy)
# utils::write.csv(
#   data.frame(bulk_props, check.names = FALSE),
#   file = file.path(OUT_DIR, "Bisque_proportions.csv"),
#   row.names = TRUE
# )

# if (!is.null(res$sc.props)) {
#   data.table::fwrite(
#     data.frame(cell_type = rownames(res$sc.props), res$sc.props, check.names = FALSE),
#     file.path(OUT_DIR, "bisque_sc_proportions.tsv"), sep = "\t"
#   )
# }

# if (!is.null(res$genes.used)) {
#   writeLines(res$genes.used, file.path(OUT_DIR, "bisque_genes_used.txt"))
# } else {
#   writeLines(rownames(bulk.mat), file.path(OUT_DIR, "bisque_genes_used.txt"))
# }

# # Bar of sums (open an explicit device)
# png(filename = file.path(OUT_DIR, "prop_sums.png"), width = 1200, height = 400)
# par(mar=c(6,4,2,1))
# barplot(prop_sums, las = 2, main = "Sum of estimated proportions per sample")
# abline(h = 1, col = "red", lty = 2)
# safe_dev_off()  # SAFE CLOSE

# # ======================= POST-HOC QC + VISUALS ==============================
# PROP_FILE <- file.path(OUT_DIR, "bisque_bulk_proportions.tsv")
# SIG_FILE  <- file.path(OUT_DIR, "signature_matrix.tsv")
# stopifnot(file.exists(PROP_FILE), file.exists(SIG_FILE))

# props_dt <- data.table::fread(PROP_FILE)
# stopifnot("cell_type" %in% names(props_dt))
# P_raw <- as.matrix(props_dt[, -1, with=FALSE]); rownames(P_raw) <- props_dt$cell_type
# P_raw <- ensure_double_matrix(P_raw)
# samples <- colnames(P_raw)

# sig_dt <- data.table::fread(SIG_FILE)
# gcol <- if ("gene" %in% names(sig_dt)) "gene" else names(sig_dt)[1]
# genes <- sig_dt[[gcol]]; sig_dt[[gcol]] <- NULL
# Z_raw <- ensure_double_matrix(as.matrix(sig_dt)); rownames(Z_raw) <- genes

# # Harmonize CT labels
# colnames(Z_raw) <- harmonize_labels(colnames(Z_raw))
# rownames(P_raw) <- harmonize_labels(rownames(P_raw))
# if (any(duplicated(colnames(Z_raw))))  stop("Duplicate CTs in signature (Z).")
# if (any(duplicated(rownames(P_raw))))  stop("Duplicate CTs in proportions (P).")
# CT <- intersect(colnames(Z_raw), rownames(P_raw))
# if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d) after harmonization.", length(CT)))
# Z_use <- ensure_double_matrix(Z_raw[, CT, drop=FALSE])   # genes x CT
# P_use <- ensure_double_matrix(P_raw[CT,  , drop=FALSE])  # CT x samples

# # Tidy for plots
# props_dt_use <- data.frame(cell_type = rownames(P_use), P_use, check.names = FALSE)
# props_long <- data.table::melt(data.table::as.data.table(props_dt_use), id.vars="cell_type",
#                                variable.name="sample", value.name="prop")
# wide <- data.table::dcast(props_long, sample ~ cell_type, value.var="prop")
# prop_mat <- as.matrix(wide[, -1, with=FALSE]); rownames(prop_mat) <- wide$sample
# prop_mat <- ensure_double_matrix(prop_mat)

# has_sample_meta <- !is.null(SAMPLE_META_FILE) && file.exists(SAMPLE_META_FILE)
# if (has_sample_meta) meta <- data.table::fread(SAMPLE_META_FILE)

# have_bulk <- file.exists(BULK_COUNTS_FILE)
# if (have_bulk) {
#   bulk_dt <- data.table::fread(BULK_COUNTS_FILE)
#   gcol2 <- intersect(c("gene","symbol","Gene","Gene_Symbol","ID","Name","GeneID","genes"), names(bulk_dt))[1]
#   if (is.na(gcol2)) gcol2 <- names(bulk_dt)[1]
#   bgenes <- bulk_dt[[gcol2]]; bulk_dt[[gcol2]] <- NULL
#   X_bulk <- ensure_double_matrix(as.matrix(bulk_dt)); rownames(X_bulk) <- as.character(bgenes)
#   X_cpm  <- cpm_mat(X_bulk)
# }

# have_sc_props <- !is.null(res$sc.props)
# if (have_sc_props) {
#   Ct_raw <- ensure_double_matrix(as.matrix(res$sc.props))
#   rownames(Ct_raw) <- harmonize_labels(rownames(res$sc.props))
# }

# # --- QC: sums table + hist
# prop_sums2 <- colSums(P_use, na.rm=TRUE)
# qc_df <- data.frame(sample=names(prop_sums2), sum_props=as.numeric(prop_sums2),
#                     abs_dev=abs(as.numeric(prop_sums2)-1))
# data.table::fwrite(qc_df, file.path(TAB_DIR,"qc_prop_sums.tsv"), sep="\t")

# g_hist <- ggplot2::ggplot(qc_df, ggplot2::aes(sum_props)) +
#   ggplot2::geom_histogram(bins=30, alpha=.9) +
#   ggplot2::geom_vline(xintercept=1, linetype=2, color="red") +
#   theme_pub() + ggplot2::labs(title="Distribution of proportion sums", x="Sum of proportions", y="Count")
# ggplot2::ggsave(file.path(PLOT_DIR,"02_qc_sum_hist.png"), g_hist, width=6.2, height=4.5, dpi=200)

# # --- Stacked composition
# hc <- hclust(dist(prop_mat), method = "complete")
# sample_order <- rownames(prop_mat)[hc$order]
# props_long$sample <- factor(props_long$sample, levels = sample_order)
# gg1 <- ggplot2::ggplot(props_long, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
#   ggplot2::geom_bar(stat = "identity", width = 0.95) +
#   ggplot2::labs(title = "Cell-type composition per sample (cluster-ordered)", x = "Sample", y = "Proportion") +
#   ggplot2::theme_bw() + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=90, vjust=0.5, hjust=1))
# ggplot2::ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), gg1, width = 16, height = 7, dpi = 200)

# # --- Heatmap of proportions
# pheatmap::pheatmap(prop_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
#                    filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
#                    main="Proportions heatmap", width=10, height=13)

# # --- Violin/box of CT distributions
# g_boxv <- ggplot2::ggplot(props_long, ggplot2::aes(cell_type, prop, fill=cell_type)) +
#   ggplot2::geom_violin(trim=FALSE, alpha=.85) + ggplot2::geom_boxplot(width=.12, outlier.size=.4) +
#   viridis::scale_fill_viridis(discrete=TRUE, guide="none") + theme_pub() +
#   ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45, hjust=1)) +
#   ggplot2::labs(title="Distribution of Bisque-estimated proportions", x="Cell type", y="Proportion")
# ggplot2::ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), g_boxv, width=11, height=6.5, dpi=200)

# # =================== OPTIONAL MARKER CHECKS & EXPORTS =======================
# eps <- 1e-8
# other_mean <- function(mat, ct) { if (ncol(mat)==1) return(rep(0, nrow(mat))); rowMeans(mat[, setdiff(colnames(mat), ct), drop=FALSE], na.rm=TRUE) }
# spec_list <- lapply(colnames(Z_use), function(ct){
#   spec <- log2((Z_use[, ct] + eps) / (other_mean(Z_use, ct) + eps))
#   data.frame(gene = rownames(Z_use), cell_type = ct, specificity = spec, expr_ct = Z_use[, ct], stringsAsFactors = FALSE)
# })
# spec_df <- data.table::rbindlist(spec_list)
# markers_sel <- spec_df[order(cell_type, -specificity), .SD[1:min(.N, TOP_MARKERS_PER_CT)], by = cell_type]
# data.table::fwrite(markers_sel, file.path(TAB_DIR, "top_markers_per_celltype.tsv"), sep="\t")

# if (!is.null(MARKER_FILE) && file.exists(MARKER_FILE) && have_bulk) {
#   mk <- data.table::fread(MARKER_FILE)
#   if (nrow(mk)) {
#     data.table::setnames(mk, old = names(mk), new = tolower(names(mk)))
#     gene_col <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","feature","id","geneid","name"), names(mk))[1]
#     ct_col   <- intersect(c("cell_type","celltype","cell_type_label","celltype_label","cluster","label","cell","ct"), names(mk))[1]
#     if (!is.na(gene_col) && !is.na(ct_col)) {
#       markers <- mk[, .(gene = get(gene_col), cell_type = harmonize_labels(get(ct_col)))]
#       markers <- unique(markers[!is.na(gene) & nzchar(gene) & !is.na(cell_type) & nzchar(cell_type)])
#       bulk_dt0 <- data.table::fread(BULK_COUNTS_FILE)
#       gcol3 <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","id","name","geneid","genes"), names(bulk_dt0))[1]
#       if (is.na(gcol3)) gcol3 <- names(bulk_dt0)[1]
#       bgenes <- bulk_dt0[[gcol3]]; bulk_dt0[[gcol3]] <- NULL
#       bulk_mat <- ensure_double_matrix(as.matrix(bulk_dt0)); rownames(bulk_mat) <- as.character(bgenes)
#       bulk_cpm <- cpm_mat(bulk_mat)
#       ct_ok <- intersect(unique(markers$cell_type), rownames(P_use))
#       ctab <- lapply(ct_ok, function(ct) {
#         gs <- intersect(markers[cell_type == ct, gene], rownames(bulk_cpm))
#         if (!length(gs)) return(NULL)
#         score <- colMeans(bulk_cpm[gs, , drop = FALSE])
#         common <- intersect(names(score), colnames(P_use))
#         if (length(common) < 3) return(NULL)
#         rho <- suppressWarnings(cor(score[common], P_use[ct, common], method = "spearman"))
#         data.frame(cell_type = ct, n_genes = length(gs), n_samples = length(common), spearman_rho = rho)
#       })
#       ctab <- data.table::rbindlist(ctab, fill = TRUE)
#       if (!is.null(ctab) && nrow(ctab)) data.table::fwrite(ctab, file.path(TAB_DIR, "marker_prop_spearman.tsv"), sep = "\t")
#     }
#   }
# }

# # ====================== S3 ADD-ONS: OVERVIEW PLOTS ==========================
# safe_ggsave <- function(path, plot=NULL, width=8, height=5, dpi=200, title_when_empty=NULL){
#   if (!is.null(plot)) {
#     ggplot2::ggsave(path, plot, width=width, height=height, dpi=dpi)
#   } else {
#     png(path, width=width*100, height=height*100)
#     par(mar=c(2,2,2,2)); plot.new()
#     ttl <- if (is.null(title_when_empty)) basename(path) else title_when_empty
#     title(ttl); mtext("No data to plot", side=3, line=-1, col="gray40")
#     safe_dev_off()
#   }
# }

# dir.create(file.path(PLOT_DIR, "S3_sample_cards"), showWarnings = FALSE, recursive = TRUE)

# # ---------- S3 overview: overall composition pie ----------
# overall_pie <- NULL
# if (exists("props_long") && nrow(props_long)) {
#   dfpie <- props_long[, .(prop = mean(prop, na.rm=TRUE)), by=cell_type]
#   dfpie <- dfpie[prop > 0]
#   if (nrow(dfpie)) {
#     overall_pie <- ggplot2::ggplot(dfpie, ggplot2::aes(x="", y=prop, fill=cell_type)) +
#       ggplot2::geom_bar(stat="identity", width=0.9) +
#       ggplot2::coord_polar(theta="y") +
#       viridis::scale_fill_viridis(discrete=TRUE, guide=NULL) +
#       ggplot2::labs(title="Overall mean composition (across samples)", y=NULL, x=NULL) +
#       ggplot2::theme_void() +
#       ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold"))
#   }
# }
# safe_ggsave(file.path(PLOT_DIR,"S3_overview_composition_pie.png"), overall_pie, 7, 7,
#             title_when_empty="Overall mean composition (pie)")

# # ---------- S3 overview: CT prevalence across samples ----------
# ct_prev_plot <- NULL
# if (exists("P_use") && ncol(P_use) > 0) {
#   prev <- data.table::as.data.table(P_use)
#   prev[, cell_type := rownames(P_use)]
#   prev_long <- data.table::melt(prev, id.vars="cell_type", variable.name="sample", value.name="prop")
#   prev_ct <- prev_long[, .(prevalence = sum(prop > 0, na.rm=TRUE)), by=cell_type]
#   if (nrow(prev_ct)) {
#     ct_prev_plot <- ggplot2::ggplot(prev_ct, ggplot2::aes(reorder(cell_type, prevalence), prevalence)) +
#       ggplot2::geom_col() + ggplot2::coord_flip() +
#       ggplot2::labs(title="CT prevalence (#samples with prop>0)", x="Cell type", y="#Samples") +
#       ggplot2::theme_bw()
#   }
# }
# safe_ggsave(file.path(PLOT_DIR,"S3_celltype_prevalence.png"), ct_prev_plot, 8, 7,
#             title_when_empty="CT prevalence")

# # ---------- S3 overview: sample × sample correlation of compositions ----------
# corr_plot_path <- file.path(PLOT_DIR, "S3_props_correlation_heatmap.png")
# if (exists("prop_mat") && nrow(prop_mat) > 1 && ncol(prop_mat) > 1) {
#   cm <- suppressWarnings(cor(t(prop_mat), method="spearman", use="pairwise.complete.obs"))
#   cm[!is.finite(cm)] <- 0
#   tryCatch({
#     pheatmap::pheatmap(cm, filename=corr_plot_path, main="Sample×Sample Spearman correlation (compositions)",
#                        cluster_rows=TRUE, cluster_cols=TRUE, width=10, height=10)
#   }, error=function(e){
#     safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty=paste("Correlation heatmap:", e$message))
#   })
# } else {
#   safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty="Correlation heatmap (insufficient data)")
# }

# # ---------- S3 overview: PCA of compositions ----------
# pca_plot <- NULL
# if (exists("prop_mat") && nrow(prop_mat) >= 2 && ncol(prop_mat) >= 2) {
#   X <- prop_mat
#   X <- X[, colSums(is.finite(X)) == nrow(X), drop=FALSE]
#   if (ncol(X) >= 2) {
#     pc <- tryCatch(prcomp(X, center=TRUE, scale.=TRUE), error=function(e) NULL)
#     if (!is.null(pc)) {
#       scp <- data.frame(PC1=pc$x[,1], PC2=pc$x[,2], sample=rownames(pc$x))
#       pca_plot <- ggplot2::ggplot(scp, ggplot2::aes(PC1, PC2, label=sample)) +
#         ggplot2::geom_point() + ggrepel::geom_text_repel(size=3, max.overlaps=40) +
#         ggplot2::theme_bw() + ggplot2::labs(title="PCA of sample compositions")
#     }
#   }
# }
# safe_ggsave(file.path(PLOT_DIR,"S3_props_PCA.png"), pca_plot, 8, 6,
#             title_when_empty="PCA of compositions (insufficient data)")

# # ---------- S1 sanity: mean signature per CT ----------
# sig_mean_plot <- NULL
# if (exists("Z_use") && ncol(Z_use) >= 1) {
#   means <- colMeans(Z_use, na.rm=TRUE)
#   dfm <- data.frame(cell_type=names(means), mean_expr=as.numeric(means))
#   sig_mean_plot <- ggplot2::ggplot(dfm, ggplot2::aes(reorder(cell_type, mean_expr), mean_expr)) +
#     ggplot2::geom_col() + ggplot2::coord_flip() +
#     ggplot2::theme_bw() + ggplot2::labs(title="Mean signature per CT (CPM)", x="Cell type", y="Mean CPM")
# }
# safe_ggsave(file.path(PLOT_DIR,"S1_signature_ct_means.png"), sig_mean_plot, 7, 6,
#             title_when_empty="Mean signature per CT")

# # =================== Sample cards (CT-aligned!) ===================
# # Master switch: disable Sample Cards by default (flip to TRUE to enable)
# GENERATE_SAMPLE_CARDS <- FALSE

# if (GENERATE_SAMPLE_CARDS) {
#   CARD_DIR <- file.path(PLOT_DIR, "sample_cards"); dir.create(CARD_DIR, FALSE, TRUE)
#   Yhat_all <- Z_use %*% P_use
#   sanitize <- function(x){ x <- gsub("[^A-Za-z0-9._-]+", "_", x); substr(x, 1, 80) }
#   all_rows <- list()
#   for (s in colnames(P_use)) {
#     y <- Yhat_all[, s]; y[!is.finite(y)] <- 0
#     ord <- order(y, decreasing = TRUE); g_ord <- names(y[ord]); g_ord <- g_ord[y[ord] > 0]
#     if (!length(g_ord)) next
#     n_take <- min(TOP_GENES_PER_SAMPLE, length(g_ord)); g_top <- g_ord[seq_len(n_take)]
#     Pvec <- as.numeric(P_use[, s]); names(Pvec) <- rownames(P_use)
#     Cmat <- sweep(Z_use[g_top, , drop = FALSE], 2, Pvec[colnames(Z_use)], "*"); Cmat[!is.finite(Cmat)] <- 0
#     dom_ix <- max.col(Cmat, ties.method = "first"); dom_ct <- colnames(Cmat)[dom_ix]
#     dom_val <- apply(Cmat, 1, max); totals <- rowSums(Cmat)
#     dom_share <- ifelse(totals > 0, dom_val / totals, 0)
#     df <- data.frame(sample = s, gene = g_top,
#                      predicted_signal = as.numeric(y[g_top]),
#                      dominant_ct = dom_ct, dominant_share = round(dom_share,3))
#     all_rows[[s]] <- df

#     rng <- max(df$predicted_signal) - min(df$predicted_signal)
#     size_scaled <- if (rng <= 0) rep(3.5, nrow(df)) else 1 + 4 * (df$predicted_signal - min(df$predicted_signal)) / (rng + 1e-12)
#     p <- ggplot2::ggplot(df, ggplot2::aes(x = predicted_signal, y = reorder(gene, predicted_signal), color = dominant_ct)) +
#       ggplot2::geom_segment(aes(x = 0, xend = predicted_signal, yend = reorder(gene, predicted_signal)), linewidth = 0.6, alpha = 0.7) +
#       ggplot2::geom_point(aes(size = size_scaled), alpha = 0.9) +
#       ggplot2::scale_size_continuous(range = c(2.5, 7), guide = "none") +
#       ggplot2::labs(title = paste0("Top genes in sample: ", s),
#            subtitle = "Color = dominant cell type | Size/bar = predicted gene signal (Z × p)",
#            x = "Predicted gene signal (Z × p)", y = "Gene") +
#       ggplot2::theme_bw(base_size = 12) + ggplot2::theme(plot.title = ggplot2::element_text(face = "bold"))
#     ggplot2::ggsave(file.path(CARD_DIR, paste0("sample_", sanitize(s), "_top_genes.png")),
#            p, width = 9.5, height = 6.5, dpi = 220)
#   }
#   if (length(all_rows)) data.table::fwrite(data.table::rbindlist(all_rows, use.names = TRUE, fill = TRUE),
#                                            file.path(TAB_DIR, "top_genes_per_sample.tsv"), sep="\t")
# }

# # Final safety: close any stray device if open
# if (names(dev.cur()) != "null device") dev.off()

# msg("DONE. Outputs written to: ", OUT_DIR)
# msg("Tables → ", TAB_DIR)
# msg("Plots  → ", PLOT_DIR)


#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)
set.seed(42L)

# ## ============================ OPTIONS =======================================
# # Environment variables set by Python
# get_bool  <- function(x, default = FALSE) {
#   v <- toupper(Sys.getenv(x, ifelse(default, "TRUE", "FALSE"))); v %in% c("1","T","TRUE","YES","Y")
# }
# get_int   <- function(x, default = NA_integer_) {
#   v <- suppressWarnings(as.integer(Sys.getenv(x, ""))); if (is.na(v)) default else v
# }
# get_num   <- function(x, default = NA_real_) {
#   v <- suppressWarnings(as.numeric(Sys.getenv(x, ""))); if (is.na(v)) default else v
# }
# get_str   <- function(x, default = "") {
#   v <- Sys.getenv(x, ""); if (!nzchar(v)) default else v
# }

# REF_DIR_PY              <- get_str("REF_DIR_PY", "")
# OUT_DIR_PY              <- get_str("OUT_DIR_PY", "")
# DROP_MITO_PY            <- get_bool("DROP_MITO_PY", FALSE)
# ALLOW_DUP_PY            <- get_bool("ALLOW_DUP_PY", FALSE)
# WARN_MIN_OVERLAP_PY     <- get_int ("WARN_MIN_OVERLAP_PY", 100L)
# MAX_PROP_DEV_PY         <- get_num ("MAX_PROP_DEV_PY", 0.2)
# TOP_MARKERS_PER_CT_PY   <- get_int ("TOP_MARKERS_PER_CT_PY", 10L)
# TOP_GENES_PER_SAMPLE_PY <- get_int ("TOP_GENES_PER_SAMPLE_PY", 50L)
# AUTO_INSTALL_PY         <- get_bool("AUTO_INSTALL_PY", TRUE)

# if (!nzchar(REF_DIR_PY)) stop("Environment variable REF_DIR_PY not set; Python must pass it to R.")

# ## ============================ UTILITIES =====================================
# msg <- function(...) cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] "), ..., "\n")

# safe_dev_off <- function() {
#   if (names(dev.cur()) != "null device") try(dev.off(), silent = TRUE)
# }

# ensure_double_matrix <- function(m) {
#   if (isS4(m)) m <- as.matrix(m)
#   if (!is.matrix(m)) m <- as.matrix(m)
#   storage.mode(m) <- "double"; m[!is.finite(m)] <- 0; m
# }

# is_all_numeric_vec <- function(x) all(grepl("^[0-9]+$", as.character(x)))
# norm_bc <- function(x){ x <- trimws(as.character(x)); x <- gsub("^X(?=\\d)","",x,perl=TRUE); x <- gsub("\\.", "-", x); x }
# drop_mito <- function(genes) genes[!grepl("^MT-", toupper(genes))]
# zero_var_rows <- function(m) { v <- apply(m, 1, function(x) var(x, na.rm=TRUE)); names(which(is.na(v) | v == 0)) }

# collapse_dups <- function(m) {
#   m <- ensure_double_matrix(m)
#   if (is.null(rownames(m))) stop("Matrix lacks rownames.")
#   if (!any(duplicated(rownames(m)))) return(m)
#   msg("Duplicate gene IDs detected; collapsing by sum.")
#   ux <- unique(rownames(m)); idx <- match(rownames(m), ux)
#   sp <- Matrix::sparseMatrix(i = idx, j = seq_len(nrow(m)), x = 1,
#                              dims = c(length(ux), nrow(m)))
#   out <- ensure_double_matrix(as.matrix(sp %*% m))
#   rownames(out) <- ux; colnames(out) <- colnames(m)
#   out
# }

# drop_zero_expr_samples <- function(m, label="bulk") {
#   m <- ensure_double_matrix(m)
#   cs <- colSums(m, na.rm=TRUE); zc <- which(!is.finite(cs) | cs <= 0)
#   if (length(zc)) {
#     msg("Dropping ", length(zc), " ", label, " sample(s) with zero/NA expr: ",
#         paste(colnames(m)[zc], collapse=", "))
#     m <- m[, -zc, drop=FALSE]
#   }
#   if (ncol(m) == 0) stop("All ", label, " samples dropped; cannot proceed.")
#   m
# }

# theme_pub <- function(){
#   ggplot2::theme_bw(base_size=12) +
#     ggplot2::theme(plot.title=ggplot2::element_text(face="bold"),
#                    panel.grid.minor=ggplot2::element_blank())
# }

# cpm_mat <- function(m){
#   m <- ensure_double_matrix(m)
#   lib <- colSums(m,na.rm=TRUE); lib[!is.finite(lib)|lib==0] <- 1
#   ensure_double_matrix(t(1e6*t(m)/lib))
# }

# harmonize_labels <- function(x) make.names(trimws(as.character(x)), unique = FALSE)

# ## =========================== PATHS & PARAMS =================================
# REF_DIR <- REF_DIR_PY
# OUT_DIR_DEFAULT <- file.path(REF_DIR, "bisque_deconv-results")
# OUT_DIR <- if (!is.null(OUT_DIR_PY) && nzchar(OUT_DIR_PY)) OUT_DIR_PY else OUT_DIR_DEFAULT
# dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# DROP_MITO_GENES       <- isTRUE(DROP_MITO_PY)
# ALLOW_DUPLICATE_GENES <- isTRUE(ALLOW_DUP_PY)
# WARN_MIN_OVERLAP      <- as.integer(WARN_MIN_OVERLAP_PY)
# MAX_PROP_DEVIATION    <- as.numeric(MAX_PROP_DEV_PY)
# TOP_MARKERS_PER_CT    <- as.integer(TOP_MARKERS_PER_CT_PY)
# TOP_GENES_PER_SAMPLE  <- as.integer(TOP_GENES_PER_SAMPLE_PY)
# AUTO_INSTALL          <- isTRUE(AUTO_INSTALL_PY)

# SC_COUNTS_FILE   <- file.path(REF_DIR, "sc_counts.tsv")
# SC_META_FILE     <- file.path(REF_DIR, "sc_metadata.tsv")
# BULK_COUNTS_FILE <- file.path(REF_DIR, "bulk_counts_overlap.tsv")

# SAMPLE_META_FILE <- file.path(REF_DIR, "sample_metadata_patientAKI.tsv")  # optional
# MARKER_FILE      <- file.path(REF_DIR, "marker_gene_table.tsv")           # optional

# PLOT_DIR <- file.path(OUT_DIR, "deconv_visual_reports"); dir.create(PLOT_DIR, FALSE, TRUE)
# TAB_DIR  <- file.path(OUT_DIR, "QC & Performance Metrics"); dir.create(TAB_DIR,  FALSE, TRUE)

# ## ============================ PACKAGES ======================================
# cran_pkgs <- c("data.table","Matrix","tools","nnls","ggplot2","viridis",
#                "pheatmap","corrplot","dplyr","tidyr","patchwork","ggrepel")
# bio_pkgs  <- c("Biobase","BisqueRNA")

# need_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace, logical(1), quietly = TRUE)]
# need_bio  <- bio_pkgs[!vapply(bio_pkgs,  requireNamespace, logical(1), quietly = TRUE)]

# if (AUTO_INSTALL) {
#   if (length(need_cran)) {
#     msg("Installing CRAN packages: ", paste(need_cran, collapse=", "))
#     install.packages(need_cran, repos = "https://cloud.r-project.org")
#   }
#   if (!requireNamespace("BiocManager", quietly = TRUE)) {
#     install.packages("BiocManager", repos = "https://cloud.r-project.org")
#   }
#   if (length(need_bio)) {
#     msg("Installing Bioconductor packages: ", paste(need_bio, collapse=", "))
#     BiocManager::install(need_bio, update = FALSE, ask = FALSE)
#   }
# } else if (length(need_cran) || length(need_bio)) {
#   stop("Missing R packages: ", paste(c(need_cran, need_bio), collapse=", "),
#        "\nRun again with auto_install=TRUE or install manually.")
# }

# suppressPackageStartupMessages({
#   lapply(c(cran_pkgs, bio_pkgs), library, character.only = TRUE)
# })
# capture.output(sessionInfo(), file = file.path(OUT_DIR, "R_sessionInfo.txt"))

# ## ======================= LOAD SINGLE-CELL DATA ==============================
# stopifnot(file.exists(SC_COUNTS_FILE), file.exists(SC_META_FILE))
# msg("Loading single-cell counts: ", SC_COUNTS_FILE)
# cnt_full <- data.table::fread(SC_COUNTS_FILE, sep="\t", header=TRUE, data.table=FALSE)

# # First column must be 'gene'; if not, treat the first as gene column anyway.
# genes <- as.character(cnt_full[[1]]); cnt_full[[1]] <- NULL
# m_sc <- ensure_double_matrix(as.matrix(cnt_full))
# rownames(m_sc) <- genes
# counts_bc_raw <- colnames(cnt_full)

# msg("Loading metadata: ", SC_META_FILE)
# md <- data.table::fread(SC_META_FILE, sep="\t", header=TRUE, data.table=FALSE)
# if (!"cell_type" %in% names(md))     md$cell_type     <- NA_character_
# if (!"individual_id" %in% names(md)) md$individual_id <- "ONE_SUBJECT"
# if (!"cell_barcode" %in% names(md))  md$cell_barcode  <- NA_character_

# counts_bc <- norm_bc(counts_bc_raw); meta_bc <- norm_bc(md$cell_barcode)
# if (nrow(md) != length(counts_bc)) {
#   stop(sprintf("Metadata rows (%d) != number of cells in counts (%d). Re-export so sizes match.",
#                nrow(md), length(counts_bc)))
# }
# if (length(na.omit(unique(meta_bc))) <= 2 || is_all_numeric_vec(na.omit(meta_bc))) {
#   md$cell_barcode <- counts_bc; meta_bc <- counts_bc
# }
# if (is_all_numeric_vec(counts_bc) && is_all_numeric_vec(meta_bc)) {
#   msg("Counts & metadata both numeric; generating synthetic barcodes (in-memory).")
#   syn <- sprintf("C%06d", seq_len(length(counts_bc)))
#   colnames(m_sc) <- syn; md$cell_barcode <- syn
#   counts_bc <- syn; meta_bc <- syn
# } else {
#   colnames(m_sc) <- counts_bc
# }

# # Align (transpose rescue + synthetic fallback)
# ov <- intersect(colnames(m_sc), md$cell_barcode)
# if (!length(ov) && any(rownames(m_sc) %in% md$cell_barcode)) {
#   msg("Counts appear transposed; transposing ...")
#   m_sc <- ensure_double_matrix(t(m_sc))
#   ov <- intersect(colnames(m_sc), md$cell_barcode)
# }
# if (!length(ov)) {
#   syn <- sprintf("C%06d", seq_len(ncol(m_sc)))
#   colnames(m_sc) <- syn; md$cell_barcode <- syn; ov <- syn
# }
# m_sc <- m_sc[, ov, drop=FALSE]
# md <- md[match(ov, md$cell_barcode), , drop=FALSE]
# rownames(md) <- md$cell_barcode

# # Build ExpressionSet for sc
# pdat <- new("AnnotatedDataFrame", data = data.frame(
#   cellType    = md$cell_type,
#   SubjectName = md$individual_id,
#   row.names   = rownames(md),
#   check.names = FALSE
# ))
# sc.eset <- Biobase::ExpressionSet(assayData = ensure_double_matrix(m_sc), phenoData = pdat)

# ## ============================== LOAD BULK ===================================
# stopifnot(file.exists(BULK_COUNTS_FILE))
# read_bulk <- function(path) {
#   dt <- data.table::fread(path, sep="\t")
#   cand <- c("gene","genes","gene_id","geneid","ensembl","ensembl_id","symbol",
#             "gene_symbol","hgnc","Gene","GeneID","Gene_Symbol","Gene.Name","ID","Name")
#   gcol <- intersect(cand, names(dt))[1]
#   if (is.na(gcol)) {
#     if (!is.numeric(dt[[1]]) && length(unique(dt[[1]])) == nrow(dt)) gcol <- names(dt)[1]
#     else stop("Cannot detect gene column in bulk file.")
#   }
#   genes <- as.character(dt[[gcol]]); dt[[gcol]] <- NULL
#   m <- ensure_double_matrix(as.matrix(dt))
#   rownames(m) <- genes
#   msg("Bulk matrix dims (genes x samples): ", nrow(m), " x ", ncol(m))
#   m
# }
# bulk.mat <- read_bulk(BULK_COUNTS_FILE)

# ## ============================ SANITY + FILTERS ==============================
# stopifnot(ncol(Biobase::exprs(sc.eset)) == nrow(Biobase::pData(sc.eset)))

# if (any(Biobase::exprs(sc.eset) < 0, na.rm=TRUE)) stop("Negative values in single-cell counts.")
# if (any(bulk.mat < 0, na.rm=TRUE))                stop("Negative values in bulk counts.")

# if (ALLOW_DUPLICATE_GENES) {
#   msg("Collapsing duplicate genes (safe rebuild)...")
#   expr_new <- collapse_dups(Biobase::exprs(sc.eset))
#   sc.eset <- Biobase::ExpressionSet(
#     assayData = expr_new,
#     phenoData = Biobase::phenoData(sc.eset)
#   )
#   bulk.mat <- collapse_dups(bulk.mat)
# }

# if (DROP_MITO_GENES) {
#   keep_sc   <- drop_mito(rownames(sc.eset));    sc.eset  <- sc.eset[keep_sc, ]
#   keep_bulk <- drop_mito(rownames(bulk.mat));   bulk.mat <- bulk.mat[keep_bulk, , drop=FALSE]
# }

# zv <- zero_var_rows(Biobase::exprs(sc.eset))
# if (length(zv)) {
#   sc.eset <- sc.eset[setdiff(rownames(Biobase::exprs(sc.eset)), zv), ]
#   msg("Removed ", length(zv), " zero-variance genes from sc.")
# }

# bulk_zero_rows <- which(rowSums(bulk.mat == 0, na.rm=TRUE) == ncol(bulk.mat))
# if (length(bulk_zero_rows)) {
#   bulk.mat <- bulk.mat[-bulk_zero_rows, , drop=FALSE]
#   msg("Removed ", length(bulk_zero_rows), " unexpressed genes from bulk (rows).")
# }

# # Ensure unique & ordered gene set
# gene_set <- intersect(rownames(Biobase::exprs(sc.eset)), rownames(bulk.mat))
# gene_set <- unique(gene_set)

# if (length(gene_set) < WARN_MIN_OVERLAP) {
#   msg("WARNING: low overlap gene count (", length(gene_set), "). Results may be less stable.")
# }
# if (length(gene_set) < 2L) {
#   stop(sprintf("Too few overlapping genes after prefilters (|genes| = %d).", length(gene_set)))
# }

# # ExpressionSet-aware subsetting
# sc.eset  <- sc.eset[gene_set, ]
# bulk.mat <- ensure_double_matrix(bulk.mat[gene_set, , drop=FALSE])

# # Dim sanity checks
# msg(sprintf("Dims after alignment: sc.eset %d genes × %d cells; bulk %d genes × %d samples",
#             nrow(Biobase::exprs(sc.eset)), ncol(Biobase::exprs(sc.eset)),
#             nrow(bulk.mat), ncol(bulk.mat)))

# # Drop bulk samples that are zero over the selected genes
# bulk.mat <- drop_zero_expr_samples(bulk.mat, label="bulk")

# ## ============ DONOR-LEVEL PSEUDOBULK (CPM) & SIGNATURE MATRIX ===============
# msg("Building donor-level pseudobulk (CPM) ...")
# expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))     # genes × cells
# ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# subj_vec<- as.character(Biobase::pData(sc.eset)$SubjectName)
# key <- paste(subj_vec, ct_vec, sep="__")
# idx <- split(seq_len(ncol(expr_sc)), key)
# PB_counts <- sapply(idx, function(cols) Matrix::rowSums(expr_sc[, cols, drop=FALSE]))
# PB_counts <- ensure_double_matrix(as.matrix(PB_counts))
# PB_cpm <- cpm_mat(PB_counts)
# pb_path <- file.path(OUT_DIR, "pseudobulk_donor_level.tsv")
# data.table::fwrite(data.frame(gene = rownames(PB_cpm), PB_cpm, check.names = FALSE),
#                    pb_path, sep = "\t")
# msg("Saved donor-level pseudobulk → ", pb_path)

# keep_cells <- !is.na(ct_vec) & nzchar(ct_vec)
# if (any(!keep_cells)) {
#   msg("Dropping ", sum(!keep_cells), " cells with missing/empty cell_type.")
#   sc.eset <- sc.eset[, keep_cells]
#   expr_sc <- ensure_double_matrix(Biobase::exprs(sc.eset))
#   ct_vec  <- as.character(Biobase::pData(sc.eset)$cellType)
# }

# expr <- ensure_double_matrix(Biobase::exprs(sc.eset))        # genes × cells
# lib  <- colSums(expr); lib[lib == 0] <- 1
# expr_cpm <- ensure_double_matrix(t(1e6 * t(expr) / lib))
# idxs <- split(seq_along(ct_vec), ct_vec)
# Z <- sapply(idxs, function(cols) rowMeans(expr_cpm[, cols, drop=FALSE], na.rm=TRUE))
# Z <- ensure_double_matrix(as.matrix(Z))
# colnames(Z) <- make.names(colnames(Z), unique=TRUE)
# Z[!is.finite(Z)] <- 0
# keep_gene_sig <- rowSums(Z) > 0
# if (any(!keep_gene_sig)) {
#   msg("Signature: dropping ", sum(!keep_gene_sig), " genes with all-zero across cell types.")
#   Z <- Z[keep_gene_sig, , drop=FALSE]
# }
# sig_df <- data.frame(gene = rownames(Z), Z, check.names = FALSE)
# sig_path <- file.path(OUT_DIR, "signature_matrix.tsv")
# data.table::fwrite(sig_df, sig_path, sep="\t")
# msg("Signature matrix saved: ", sig_path)

# ## ===================== BISQUE (>=2 subjects) or NNLS (1) ====================
# n_subjects <- length(unique(Biobase::pData(sc.eset)$SubjectName))
# msg("Single-cell unique subjects detected: ", n_subjects)

# if (n_subjects >= 2) {
#   genes_use <- intersect(rownames(sc.eset), rownames(bulk.mat))
#   bulk_use  <- ensure_double_matrix(bulk.mat[genes_use, , drop=FALSE])
#   bulk_use  <- drop_zero_expr_samples(bulk_use, label="bulk (pre-Bisque)")
#   bulk.eset <- Biobase::ExpressionSet(assayData = bulk_use)
#   msg("Running BisqueRNA::ReferenceBasedDecomposition ...")
#   res <- BisqueRNA::ReferenceBasedDecomposition(
#     bulk.eset     = bulk.eset,
#     sc.eset       = sc.eset,
#     markers       = NULL,
#     cell.types    = "cellType",
#     subject.names = "SubjectName",
#     use.overlap   = FALSE,
#     verbose       = TRUE,
#     old.cpm       = TRUE
#   )
#   bulk_props <- ensure_double_matrix(res$bulk.props)
# } else {
#   msg("Fallback: only ONE subject in scRNA-seq; running NNLS on signature matrix.")
#   common_genes <- intersect(rownames(Z), rownames(bulk.mat))
#   if (length(common_genes) < WARN_MIN_OVERLAP)
#     msg("WARNING: low overlap gene count for NNLS (", length(common_genes), ").")
#   A <- ensure_double_matrix(Z[common_genes, , drop=FALSE])        # genes × CT
#   B <- ensure_double_matrix(bulk.mat[common_genes, , drop=FALSE]) # genes × samples
#   A[!is.finite(A)] <- 0; B[!is.finite(B)] <- 0
#   keep_gene <- rowSums(A) > 0
#   if (any(!keep_gene)) { msg("NNLS: dropping ", sum(!keep_gene), " genes with zero signature.");
#     A <- A[keep_gene, , drop=FALSE]; B <- B[keep_gene, , drop=FALSE] }
#   keep_ct <- colSums(A) > 0
#   if (any(!keep_ct)) { msg("NNLS: dropping ", sum(!keep_ct), " CT with zero signature.");
#     A <- A[, keep_ct, drop=FALSE] }
#   if (ncol(A) == 0) stop("NNLS: no non-zero cell types remain after filtering.")
#   P_nnls <- matrix(NA_real_, nrow=ncol(A), ncol=ncol(B),
#                    dimnames=list(colnames(A), colnames(B)))
#   for (j in seq_len(ncol(B))) {
#     fit <- nnls::nnls(A, B[, j]); p <- as.numeric(fit$x); p[!is.finite(p)] <- 0
#     s <- sum(p); if (is.finite(s) && s > 0) p <- p / s
#     P_nnls[, j] <- p
#   }
#   bulk_props <- ensure_double_matrix(P_nnls)
#   res <- list(bulk.props = bulk_props, genes.used = rownames(A))
# }

# ## =============================== OUTPUTS ====================================
# prop_sums  <- colSums(bulk_props)
# bad_sum    <- which(abs(prop_sums - 1) > MAX_PROP_DEVIATION)
# if (length(bad_sum))
#   msg("WARNING: |sum(props)-1| >", MAX_PROP_DEVIATION, " for samples: ",
#       paste(colnames(bulk_props)[bad_sum], collapse=", "))

# # TSV (canonical)
# data.table::fwrite(
#   data.frame(cell_type = rownames(bulk_props), bulk_props, check.names = FALSE),
#   file.path(OUT_DIR, "bisque_bulk_proportions.tsv"), sep = "\t"
# )
# # CSV (legacy)
# utils::write.csv(
#   data.frame(bulk_props, check.names = FALSE),
#   file = file.path(OUT_DIR, "Bisque_proportions.csv"),
#   row.names = TRUE
# )

# if (!is.null(res$sc.props)) {
#   data.table::fwrite(
#     data.frame(cell_type = rownames(res$sc.props), res$sc.props, check.names = FALSE),
#     file.path(OUT_DIR, "bisque_sc_proportions.tsv"), sep = "\t"
#   )
# }

# if (!is.null(res$genes.used)) {
#   writeLines(res$genes.used, file.path(OUT_DIR, "bisque_genes_used.txt"))
# } else {
#   writeLines(rownames(bulk.mat), file.path(OUT_DIR, "bisque_genes_used.txt"))
# }

# # Bar of sums (explicit device)
# png(filename = file.path(OUT_DIR, "prop_sums.png"), width = 1200, height = 400)
# par(mar=c(6,4,2,1))
# barplot(prop_sums, las = 2, main = "Sum of estimated proportions per sample")
# abline(h = 1, col = "red", lty = 2)
# safe_dev_off()

# ## ======================= POST-HOC QC + VISUALS ==============================
# PROP_FILE <- file.path(OUT_DIR, "bisque_bulk_proportions.tsv")
# SIG_FILE  <- file.path(OUT_DIR, "signature_matrix.tsv")
# stopifnot(file.exists(PROP_FILE), file.exists(SIG_FILE))

# props_dt <- data.table::fread(PROP_FILE)
# stopifnot("cell_type" %in% names(props_dt))
# P_raw <- as.matrix(props_dt[, -1, with=FALSE]); rownames(P_raw) <- props_dt$cell_type
# P_raw <- ensure_double_matrix(P_raw)

# sig_dt <- data.table::fread(SIG_FILE)
# gcol <- if ("gene" %in% names(sig_dt)) "gene" else names(sig_dt)[1]
# genes <- sig_dt[[gcol]]; sig_dt[[gcol]] <- NULL
# Z_raw <- ensure_double_matrix(as.matrix(sig_dt)); rownames(Z_raw) <- genes

# # Harmonize CT labels
# colnames(Z_raw) <- harmonize_labels(colnames(Z_raw))
# rownames(P_raw) <- harmonize_labels(rownames(P_raw))
# if (any(duplicated(colnames(Z_raw))))  stop("Duplicate CTs in signature (Z).")
# if (any(duplicated(rownames(P_raw))))  stop("Duplicate CTs in proportions (P).")
# CT <- intersect(colnames(Z_raw), rownames(P_raw))
# if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d) after harmonization.", length(CT)))
# Z_use <- ensure_double_matrix(Z_raw[, CT, drop=FALSE])   # genes x CT
# P_use <- ensure_double_matrix(P_raw[CT,  , drop=FALSE])  # CT x samples

# # Tidy for plots
# props_dt_use <- data.frame(cell_type = rownames(P_use), P_use, check.names = FALSE)
# props_long <- data.table::melt(data.table::as.data.table(props_dt_use), id.vars="cell_type",
#                                variable.name="sample", value.name="prop")
# wide <- data.table::dcast(props_long, sample ~ cell_type, value.var="prop")
# prop_mat <- as.matrix(wide[, -1, with=FALSE]); rownames(prop_mat) <- wide$sample
# prop_mat <- ensure_double_matrix(prop_mat)

# has_sample_meta <- !is.null(SAMPLE_META_FILE) && file.exists(SAMPLE_META_FILE)
# if (has_sample_meta) meta <- data.table::fread(SAMPLE_META_FILE)

# have_bulk <- file.exists(BULK_COUNTS_FILE)
# if (have_bulk) {
#   bulk_dt <- data.table::fread(BULK_COUNTS_FILE)
#   gcol2 <- intersect(c("gene","symbol","Gene","Gene_Symbol","ID","Name","GeneID","genes"), names(bulk_dt))[1]
#   if (is.na(gcol2)) gcol2 <- names(bulk_dt)[1]
#   bgenes <- bulk_dt[[gcol2]]; bulk_dt[[gcol2]] <- NULL
#   X_bulk <- ensure_double_matrix(as.matrix(bulk_dt)); rownames(X_bulk) <- as.character(bgenes)
#   X_cpm  <- cpm_mat(X_bulk)
# }

# have_sc_props <- !is.null(res$sc.props)
# if (have_sc_props) {
#   Ct_raw <- ensure_double_matrix(as.matrix(res$sc.props))
#   rownames(Ct_raw) <- harmonize_labels(rownames(res$sc.props))
# }

# # --- QC: sums table + hist
# prop_sums <- colSums(P_use, na.rm=TRUE)
# qc_df <- data.frame(sample=names(prop_sums), sum_props=as.numeric(prop_sums),
#                     abs_dev=abs(as.numeric(prop_sums)-1))
# data.table::fwrite(qc_df, file.path(TAB_DIR,"qc_prop_sums.tsv"), sep="\t")

# g_hist <- ggplot2::ggplot(qc_df, ggplot2::aes(sum_props)) +
#   ggplot2::geom_histogram(bins=30, alpha=.9) +
#   ggplot2::geom_vline(xintercept=1, linetype=2, color="red") +
#   theme_pub() + ggplot2::labs(title="Distribution of proportion sums", x="Sum of proportions", y="Count")
# ggplot2::ggsave(file.path(PLOT_DIR,"02_qc_sum_hist.png"), g_hist, width=6.2, height=4.5, dpi=200)

# # --- Stacked composition
# hc <- hclust(dist(prop_mat), method = "complete")
# sample_order <- rownames(prop_mat)[hc$order]
# props_long$sample <- factor(props_long$sample, levels = sample_order)
# gg1 <- ggplot2::ggplot(props_long, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
#   ggplot2::geom_bar(stat = "identity", width = 0.95) +
#   ggplot2::labs(title = "Cell-type composition per sample (cluster-ordered)", x = "Sample", y = "Proportion") +
#   ggplot2::theme_bw() + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=90, vjust=0.5, hjust=1))
# ggplot2::ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), gg1, width = 16, height = 7, dpi = 200)

# # --- Heatmap of proportions
# pheatmap::pheatmap(prop_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
#                    filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
#                    main="Proportions heatmap", width=10, height=13)

# # --- Violin/box of CT distributions
# g_boxv <- ggplot2::ggplot(props_long, ggplot2::aes(cell_type, prop, fill=cell_type)) +
#   ggplot2::geom_violin(trim=FALSE, alpha=.85) + ggplot2::geom_boxplot(width=.12, outlier.size=.4) +
#   viridis::scale_fill_viridis(discrete=TRUE, guide="none") + theme_pub() +
#   ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45, hjust=1)) +
#   ggplot2::labs(title="Distribution of Bisque-estimated proportions", x="Cell type", y="Proportion")
# ggplot2::ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), g_boxv, width=11, height=6.5, dpi=200)

# ## =================== OPTIONAL MARKER CHECKS & EXPORTS =======================
# eps <- 1e-8
# other_mean <- function(mat, ct) { if (ncol(mat)==1) return(rep(0, nrow(mat))); rowMeans(mat[, setdiff(colnames(mat), ct), drop=FALSE], na.rm=TRUE) }
# spec_list <- lapply(colnames(Z_use), function(ct){
#   spec <- log2((Z_use[, ct] + eps) / (other_mean(Z_use, ct) + eps))
#   data.frame(gene = rownames(Z_use), cell_type = ct, specificity = spec, expr_ct = Z_use[, ct], stringsAsFactors = FALSE)
# })
# spec_df <- data.table::rbindlist(spec_list)
# markers_sel <- spec_df[order(cell_type, -specificity), .SD[1:min(.N, TOP_MARKERS_PER_CT)], by = cell_type]
# data.table::fwrite(markers_sel, file.path(TAB_DIR, "top_markers_per_celltype.tsv"), sep="\t")

# if (!is.null(MARKER_FILE) && file.exists(MARKER_FILE) && have_bulk) {
#   mk <- data.table::fread(MARKER_FILE)
#   if (nrow(mk)) {
#     data.table::setnames(mk, old = names(mk), new = tolower(names(mk)))
#     gene_col <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","feature","id","geneid","name"), names(mk))[1]
#     ct_col   <- intersect(c("cell_type","celltype","cell_type_label","celltype_label","cluster","label","cell","ct"), names(mk))[1]
#     if (!is.na(gene_col) && !is.na(ct_col)) {
#       markers <- mk[, .(gene = get(gene_col), cell_type = harmonize_labels(get(ct_col)))]
#       markers <- unique(markers[!is.na(gene) & nzchar(gene) & !is.na(cell_type) & nzchar(cell_type)])
#       bulk_dt0 <- data.table::fread(BULK_COUNTS_FILE)
#       gcol3 <- intersect(c("gene","symbol","gene_symbol","hgnc_symbol","id","name","geneid","genes"), names(bulk_dt0))[1]
#       if (is.na(gcol3)) gcol3 <- names(bulk_dt0)[1]
#       bgenes <- bulk_dt0[[gcol3]]; bulk_dt0[[gcol3]] <- NULL
#       bulk_mat2 <- ensure_double_matrix(as.matrix(bulk_dt0)); rownames(bulk_mat2) <- as.character(bgenes)
#       bulk_cpm <- cpm_mat(bulk_mat2)
#       ct_ok <- intersect(unique(markers$cell_type), rownames(P_use))
#       ctab <- lapply(ct_ok, function(ct) {
#         gs <- intersect(markers[cell_type == ct, gene], rownames(bulk_cpm))
#         if (!length(gs)) return(NULL)
#         score <- colMeans(bulk_cpm[gs, , drop = FALSE])
#         common <- intersect(names(score), colnames(P_use))
#         if (length(common) < 3) return(NULL)
#         rho <- suppressWarnings(cor(score[common], P_use[ct, common], method = "spearman"))
#         data.frame(cell_type = ct, n_genes = length(gs), n_samples = length(common), spearman_rho = rho)
#       })
#       ctab <- data.table::rbindlist(ctab, fill = TRUE)
#       if (!is.null(ctab) && nrow(ctab)) data.table::fwrite(ctab, file.path(TAB_DIR, "marker_prop_spearman.tsv"), sep = "\t")
#     }
#   }
# }

# # ====================== S3 ADD-ONS: OVERVIEW + SAMPLE CARDS ===================
# safe_ggsave <- function(path, plot=NULL, width=8, height=5, dpi=200, title_when_empty=NULL){
#   if (!is.null(plot)) {
#     ggplot2::ggsave(path, plot, width=width, height=height, dpi=dpi)
#   } else {
#     png(path, width=width*100, height=height*100)
#     par(mar=c(2,2,2,2)); plot.new()
#     ttl <- if (is.null(title_when_empty)) basename(path) else title_when_empty
#     title(ttl); mtext("No data to plot", side=3, line=-1, col="gray40")
#     dev.off()
#   }
# }

# dir.create(file.path(PLOT_DIR, "S3_sample_cards"), showWarnings = FALSE, recursive = TRUE)

# # ---------- S3 overview: overall composition pie ----------
# overall_pie <- NULL
# if (exists("props_long") && nrow(props_long)) {
#   dfpie <- props_long[, .(prop = mean(prop, na.rm=TRUE)), by=cell_type]
#   dfpie <- dfpie[prop > 0]
#   if (nrow(dfpie)) {
#     overall_pie <- ggplot2::ggplot(dfpie, ggplot2::aes(x="", y=prop, fill=cell_type)) +
#       ggplot2::geom_bar(stat="identity", width=0.9) +
#       ggplot2::coord_polar(theta="y") +
#       viridis::scale_fill_viridis(discrete=TRUE, guide=NULL) +
#       ggplot2::labs(title="Overall mean composition (across samples)", y=NULL, x=NULL) +
#       ggplot2::theme_void() +
#       ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold"))
#   }
# }
# safe_ggsave(file.path(PLOT_DIR,"S3_overview_composition_pie.png"), overall_pie, 7, 7,
#             title_when_empty="Overall mean composition (pie)")

# # ---------- S3 overview: CT prevalence across samples ----------
# ct_prev_plot <- NULL
# if (exists("P_use") && ncol(P_use) > 0) {
#   prev <- data.table::as.data.table(P_use)
#   prev[, cell_type := rownames(P_use)]
#   prev_long <- data.table::melt(prev, id.vars="cell_type", variable.name="sample", value.name="prop")
#   prev_ct <- prev_long[, .(prevalence = sum(prop > 0, na.rm=TRUE)), by=cell_type]
#   if (nrow(prev_ct)) {
#     ct_prev_plot <- ggplot2::ggplot(prev_ct, ggplot2::aes(reorder(cell_type, prevalence), prevalence)) +
#       ggplot2::geom_col() + ggplot2::coord_flip() +
#       ggplot2::labs(title="CT prevalence (#samples with prop>0)", x="Cell type", y="#Samples") +
#       ggplot2::theme_bw()
#   }
# }
# safe_ggsave(file.path(PLOT_DIR,"S3_celltype_prevalence.png"), ct_prev_plot, 8, 7,
#             title_when_empty="CT prevalence")

# # ---------- S3 overview: sample × sample correlation of compositions ----------
# corr_plot_path <- file.path(PLOT_DIR, "S3_props_correlation_heatmap.png")
# if (exists("prop_mat") && nrow(prop_mat) > 1 && ncol(prop_mat) > 1) {
#   cm <- suppressWarnings(cor(t(prop_mat), method="spearman", use="pairwise.complete.obs"))
#   cm[!is.finite(cm)] <- 0
#   tryCatch({
#     pheatmap::pheatmap(cm, filename=corr_plot_path, main="Sample×Sample Spearman correlation (compositions)",
#                        cluster_rows=TRUE, cluster_cols=TRUE, width=10, height=10)
#   }, error=function(e){
#     safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty=paste("Correlation heatmap:", e$message))
#   })
# } else {
#   safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty="Correlation heatmap (insufficient data)")
# }

# # ---------- S3 overview: PCA of compositions ----------
# pca_plot <- NULL
# if (exists("prop_mat") && nrow(prop_mat) >= 2 && ncol(prop_mat) >= 2) {
#   X <- prop_mat
#   X <- X[, colSums(is.finite(X)) == nrow(X), drop=FALSE]
#   if (ncol(X) >= 2) {
#     pc <- tryCatch(prcomp(X, center=TRUE, scale.=TRUE), error=function(e) NULL)
#     if (!is.null(pc)) {
#       sc <- data.frame(PC1=pc$x[,1], PC2=pc$x[,2], sample=rownames(pc$x))
#       pca_plot <- ggplot2::ggplot(sc, ggplot2::aes(PC1, PC2, label=sample)) +
#         ggplot2::geom_point() + ggrepel::geom_text_repel(size=3, max.overlaps=40) +
#         ggplot2::theme_bw() + ggplot2::labs(title="PCA of sample compositions")
#     }
#   }
# }
# safe_ggsave(file.path(PLOT_DIR,"S3_props_PCA.png"), pca_plot, 8, 6,
#             title_when_empty="PCA of compositions (insufficient data)")

# # ---------- S1 sanity: mean signature per CT ----------
# sig_mean_plot <- NULL
# if (exists("Z_use") && ncol(Z_use) >= 1) {
#   means <- colMeans(Z_use, na.rm=TRUE)
#   dfm <- data.frame(cell_type=names(means), mean_expr=as.numeric(means))
#   sig_mean_plot <- ggplot2::ggplot(dfm, ggplot2::aes(reorder(cell_type, mean_expr), mean_expr)) +
#     ggplot2::geom_col() + ggplot2::coord_flip() +
#     ggplot2::theme_bw() + ggplot2::labs(title="Mean signature per CT (CPM)", x="Cell type", y="Mean CPM")
# }
# safe_ggsave(file.path(PLOT_DIR,"S1_signature_ct_means.png"), sig_mean_plot, 7, 6,
#             title_when_empty="Mean signature per CT")

# # ---------- S3: per-sample QC TSV ----------
# qc_out <- file.path(TAB_DIR, "S3_sample_qc.tsv")
# qc_df2 <- data.frame(sample = names(prop_sums),
#                      sum_props = as.numeric(prop_sums),
#                      abs_dev = abs(as.numeric(prop_sums) - 1))
# if (nrow(qc_df2)) {
#   data.table::fwrite(qc_df2, qc_out, sep="\t")
# } else {
#   data.table::fwrite(data.frame(), qc_out, sep="\t")
# }

# # ---------- NEW: robust sample-card generator (integrated) ----------
# make_sample_cards <- function(Z_use,
#                               P_use,
#                               out_dir,
#                               top_genes_per_sample = 40) {
#   # ---- Hard checks & coercions ----
#   to_double_matrix <- function(x) {
#     if (is.data.frame(x)) x <- as.matrix(x)
#     if (!is.matrix(x))   x <- as.matrix(x)
#     storage.mode(x) <- "double"
#     x[!is.finite(x)] <- 0
#     x
#   }
#   Z_use <- to_double_matrix(Z_use)
#   P_use <- to_double_matrix(P_use)

#   if (nrow(Z_use) < 1 || ncol(Z_use) < 1) stop("Z_use has no rows/cols.")
#   if (nrow(P_use) < 1 || ncol(P_use) < 1) stop("P_use has no rows/cols.")

#   # ---- Align CTs ----
#   CT <- intersect(colnames(Z_use), rownames(P_use))
#   if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d).", length(CT)))
#   Z_use <- Z_use[, CT, drop = FALSE]
#   P_use <- P_use[CT,  , drop = FALSE]

#   # ---- Output dirs ----
#   plot_root <- file.path(out_dir, "deconv_visual_reports")
#   card_dir  <- file.path(plot_root, "sample_cards")
#   if (!dir.exists(plot_root)) dir.create(plot_root, recursive = TRUE, showWarnings = FALSE)
#   if (!dir.exists(card_dir))  dir.create(card_dir,  recursive = TRUE, showWarnings = FALSE)

#   sanitize <- function(x) {
#     x <- gsub("[^A-Za-z0-9._-]+", "_", x)
#     substr(x, 1, 80)
#   }

#   # clamp negatives
#   Z_use[Z_use < 0 | !is.finite(Z_use)] <- 0
#   P_use[P_use < 0 | !is.finite(P_use)] <- 0

#   # ---- Predicted signals ----
#   Yhat_all <- Z_use %*% P_use            # genes × samples
#   rownames(Yhat_all) <- rownames(Z_use)

#   if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required.")
#   library(ggplot2)

#   message(sprintf("[Sample Cards] %d genes × %d CTs; %d samples",
#                   nrow(Z_use), ncol(Z_use), ncol(P_use)))

#   all_rows <- list()

#   # Helper: save placeholder
#   save_placeholder <- function(path, title_txt) {
#     png(path, width = 950, height = 650)
#     par(mar=c(1,1,3,1)); plot.new()
#     title(title_txt, cex.main=1.2, font.main=2)
#     mtext("No positive or finite predicted gene signal available", side=3, line=-1, col="gray40")
#     dev.off()
#   }

#   for (s in colnames(P_use)) {
#     y <- as.numeric(Yhat_all[, s]); names(y) <- rownames(Yhat_all)
#     y[!is.finite(y)] <- 0; y[y < 0] <- 0

#     # Prefer genes with positive signal
#     ord  <- order(y, decreasing = TRUE)
#     gord <- names(y)[ord]
#     gord <- gord[y[ord] > 0]

#     # Fallback if nothing > 0
#     if (!length(gord)) {
#       Pvec <- as.numeric(P_use[, s]); names(Pvec) <- rownames(P_use)
#       if (sum(Pvec, na.rm=TRUE) <= 0) {
#         out_png <- file.path(card_dir, paste0("sample_", sanitize(s), "_top_genes.png"))
#         save_placeholder(out_png, paste0("Top genes in sample: ", s))
#         next
#       }
#       pseudo_signal <- as.numeric(Z_use %*% (Pvec / max(sum(Pvec), 1e-12)))
#       names(pseudo_signal) <- rownames(Z_use)
#       ord2 <- order(pseudo_signal, decreasing = TRUE)
#       gord <- names(pseudo_signal)[ord2]
#       gord <- gord[pseudo_signal[ord2] > 0]
#       if (!length(gord)) {
#         out_png <- file.path(card_dir, paste0("sample_", sanitize(s), "_top_genes.png"))
#         save_placeholder(out_png, paste0("Top genes in sample: ", s))
#         next
#       }
#       y <- pseudo_signal
#     }

#     n_take <- min(top_genes_per_sample, length(gord))
#     g_top  <- gord[seq_len(n_take)]

#     # Gene × CT contributions
#     Pvec <- as.numeric(P_use[, s]); names(Pvec) <- rownames(P_use)
#     Cmat <- sweep(Z_use[g_top, , drop = FALSE], 2, Pvec[colnames(Z_use)], "*")
#     Cmat[!is.finite(Cmat)] <- 0
#     Cmat[Cmat < 0] <- 0

#     dom_ix    <- max.col(Cmat, ties.method = "first")
#     dom_ct    <- colnames(Cmat)[dom_ix]
#     dom_val   <- apply(Cmat, 1, max)
#     totals    <- rowSums(Cmat)
#     dom_share <- ifelse(totals > 0, dom_val / totals, 0)

#     df <- data.frame(
#       sample           = s,
#       gene             = g_top,
#       predicted_signal = as.numeric(y[g_top]),
#       dominant_ct      = dom_ct,
#       dominant_share   = round(dom_share, 3),
#       stringsAsFactors = FALSE
#     )
#     all_rows[[s]] <- df

#     ps <- df$predicted_signal; ps[!is.finite(ps)] <- 0
#     rng <- max(ps) - min(ps)
#     size_scaled <- if (rng <= 0) rep(3.5, nrow(df)) else 1 + 4 * (ps - min(ps)) / (rng + 1e-12)

#     xmax <- max(ps, na.rm=TRUE); if (!is.finite(xmax) || xmax <= 0) xmax <- 1

#     p <- ggplot(df, aes(x = predicted_signal,
#                         y = reorder(gene, predicted_signal),
#                         color = dominant_ct)) +
#       geom_segment(aes(x = 0,
#                        xend = predicted_signal,
#                        yend = reorder(gene, predicted_signal)),
#                    linewidth = 0.6, alpha = 0.7) +
#       geom_point(aes(size = size_scaled), alpha = 0.9) +
#       scale_size_continuous(range = c(2.5, 7), guide = "none") +
#       coord_cartesian(xlim = c(0, xmax * 1.05)) +
#       labs(
#         title    = paste0("Top genes in sample: ", s),
#         subtitle = "Color = dominant cell type | Size/bar = predicted gene signal (Z × p)",
#         x        = "Predicted gene signal (Z × p)",
#         y        = "Gene"
#       ) +
#       theme_bw(base_size = 12) +
#       theme(plot.title = element_text(face = "bold"))

#     out_png <- file.path(card_dir, paste0("sample_", sanitize(s), "_top_genes.png"))
#     ggsave(out_png, p, width = 9.5, height = 6.5, dpi = 220)
#   }

#   out_tsv <- file.path(plot_root, "top_genes_per_sample.tsv")
#   if (length(all_rows)) {
#     df_all <- data.table::rbindlist(all_rows, use.names = TRUE, fill = TRUE)
#     data.table::fwrite(df_all, out_tsv, sep = "\t")
#     message("[Sample Cards] Wrote: ", out_tsv)
#   } else {
#     data.table::fwrite(data.table::as.data.table(data.frame(
#       sample=character(), gene=character(), predicted_signal=double(),
#       dominant_ct=character(), dominant_share=double()
#     )), out_tsv, sep = "\t")
#     message("[Sample Cards] No rows; wrote empty TSV: ", out_tsv)
#   }

#   invisible(list(
#     card_dir = file.path(plot_root, "sample_cards"),
#     combined_tsv = out_tsv
#   ))
# }

# # ---------- CALL: generate robust sample cards ----------
# try({
#   make_sample_cards(
#     Z_use = Z_use,
#     P_use = P_use,
#     out_dir = OUT_DIR,
#     top_genes_per_sample = max(5, min(40, TOP_GENES_PER_SAMPLE))
#   )
# }, silent = TRUE)

# # ==================== END S3 ADD-ONS =========================================

# msg("DONE. Outputs written to: ", OUT_DIR)
# msg("Tables → ", TAB_DIR)
# msg("Plots  → ", PLOT_DIR)





#!/usr/bin/env Rscript
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

REF_DIR_PY              <- get_str("REF_DIR_PY", "")
OUT_DIR_PY              <- get_str("OUT_DIR_PY", "")
DROP_MITO_PY            <- get_bool("DROP_MITO_PY", FALSE)
ALLOW_DUP_PY            <- get_bool("ALLOW_DUP_PY", FALSE)
WARN_MIN_OVERLAP_PY     <- get_int ("WARN_MIN_OVERLAP_PY", 100L)
MAX_PROP_DEV_PY         <- get_num ("MAX_PROP_DEV_PY", 0.2)
TOP_MARKERS_PER_CT_PY   <- get_int ("TOP_MARKERS_PER_CT_PY", 10L)
TOP_GENES_PER_SAMPLE_PY <- get_int ("TOP_GENES_PER_SAMPLE_PY", 50L)
AUTO_INSTALL_PY         <- get_bool("AUTO_INSTALL_PY", TRUE)

if (!nzchar(REF_DIR_PY)) stop("Environment variable REF_DIR_PY not set; Python must pass it to R.")

## ============================ UTILITIES =====================================
msg <- function(...) cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] "), ..., "\n")

safe_dev_off <- function() {
  if (names(dev.cur()) != "null device") try(dev.off(), silent = TRUE)
}

ensure_double_matrix <- function(m) {
  if (isS4(m)) m <- as.matrix(m)
  if (!is.matrix(m)) m <- as.matrix(m)
  storage.mode(m) <- "double"; m[!is.finite(m)] <- 0; m
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
# CSV (legacy)
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

# Bar of sums (explicit device)
# png(filename = file.path(OUT_DIR, "prop_sums.png"), width = 1200, height = 400)
# par(mar=c(6,4,2,1))
# barplot(prop_sums, las = 2, main = "Sum of estimated proportions per sample")
# abline(h = 1, col = "red", lty = 2)
# safe_dev_off()

## ======================= POST-HOC QC + VISUALS ==============================
# Helper function for safe plot saving (defined early for use in error handling)
safe_ggsave <- function(path, plot=NULL, width=8, height=5, dpi=200, title_when_empty=NULL){
  if (!is.null(plot)) {
    tryCatch({
      ggplot2::ggsave(path, plot, width=width, height=height, dpi=dpi)
    }, error = function(e) {
      msg("Failed to save plot to ", path, ": ", e$message)
      # Create placeholder
      png(path, width=width*100, height=height*100)
      par(mar=c(2,2,2,2)); plot.new()
      ttl <- if (is.null(title_when_empty)) basename(path) else title_when_empty
      title(ttl); mtext(paste("Plot generation failed:", e$message), side=3, line=-1, col="gray40")
      dev.off()
    })
  } else {
    png(path, width=width*100, height=height*100)
    par(mar=c(2,2,2,2)); plot.new()
    ttl <- if (is.null(title_when_empty)) basename(path) else title_when_empty
    title(ttl); mtext("No data to plot", side=3, line=-1, col="gray40")
    dev.off()
  }
}

PROP_FILE <- file.path(OUT_DIR, "bisque_bulk_proportions.tsv")
SIG_FILE  <- file.path(OUT_DIR, "signature_matrix.tsv")
stopifnot(file.exists(PROP_FILE), file.exists(SIG_FILE))

props_dt <- data.table::fread(PROP_FILE)
stopifnot("cell_type" %in% names(props_dt))
P_raw <- as.matrix(props_dt[, -1, with=FALSE]); rownames(P_raw) <- props_dt$cell_type
P_raw <- ensure_double_matrix(P_raw)

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

# # --- QC: sums table + hist
prop_sums <- colSums(P_use, na.rm=TRUE)
qc_df <- data.frame(sample=names(prop_sums), sum_props=as.numeric(prop_sums),
                    abs_dev=abs(as.numeric(prop_sums)-1))
#data.table::fwrite(qc_df, file.path(TAB_DIR,"qc_prop_sums.tsv"), sep="\t")

g_hist <- ggplot2::ggplot(qc_df, ggplot2::aes(sum_props)) +
  ggplot2::geom_histogram(bins=30, alpha=.9) +
  ggplot2::geom_vline(xintercept=1, linetype=2, color="red") +
  theme_pub() + ggplot2::labs(title="Distribution of proportion sums", x="Sum of proportions", y="Count")
#ggplot2::ggsave(file.path(PLOT_DIR,"02_qc_sum_hist.png"), g_hist, width=6.2, height=4.5, dpi=200)

# --- Stacked composition (with error handling for large datasets)
tryCatch({
  # For very large datasets, limit clustering to avoid memory issues
  if (nrow(prop_mat) > 500) {
    msg("Large dataset detected (", nrow(prop_mat), " samples). Using random sampling for clustering.")
    set.seed(42L)
    sample_idx <- sample(seq_len(nrow(prop_mat)), min(500, nrow(prop_mat)))
    prop_mat_clust <- prop_mat[sample_idx, , drop=FALSE]
    hc <- hclust(dist(prop_mat_clust), method = "complete")
    sample_order_clust <- rownames(prop_mat_clust)[hc$order]
    # Use full data but order by clustered subset
    sample_order <- c(sample_order_clust, setdiff(rownames(prop_mat), sample_order_clust))
  } else {
hc <- hclust(dist(prop_mat), method = "complete")
sample_order <- rownames(prop_mat)[hc$order]
  }
props_long$sample <- factor(props_long$sample, levels = sample_order)
gg1 <- ggplot2::ggplot(props_long, ggplot2::aes(x = sample, y = prop, fill = cell_type)) +
  ggplot2::geom_bar(stat = "identity", width = 0.95) +
  ggplot2::labs(title = "Cell-type composition per sample (cluster-ordered)", x = "Sample", y = "Proportion") +
  ggplot2::theme_bw() + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=90, vjust=0.5, hjust=1))
ggplot2::ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), gg1, width = 16, height = 7, dpi = 200)
  msg("✓ Saved stacked bar plot")
}, error = function(e) {
  msg("✗ Failed to create stacked bar plot: ", e$message)
  safe_ggsave(file.path(PLOT_DIR, "plot_stacked_bar_by_sample_clustered.png"), NULL, 16, 7,
              title_when_empty=paste("Stacked bar plot failed:", e$message))
})

# --- Heatmap of proportions (with error handling and memory optimization)
tryCatch({
  # For very large datasets, downsample or use lower resolution
  if (nrow(prop_mat) > 200 || ncol(prop_mat) > 50) {
    msg("Large dataset detected. Using memory-efficient heatmap generation.")
    # Downsample rows if too many samples
    if (nrow(prop_mat) > 200) {
      set.seed(42L)
      sample_idx <- sample(seq_len(nrow(prop_mat)), 200)
      prop_mat_plot <- prop_mat[sample_idx, , drop=FALSE]
      msg("Downsampled to 200 samples for heatmap visualization")
    } else {
      prop_mat_plot <- prop_mat
    }
    # Use lower DPI and smaller dimensions for large datasets
    pheatmap::pheatmap(prop_mat_plot, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
                       filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
                       main="Proportions heatmap (downsampled for large dataset)", 
                       width=10, height=13, dpi=150)
  } else {
pheatmap::pheatmap(prop_mat, cluster_rows=TRUE, cluster_cols=TRUE, scale="none",
                   filename=file.path(PLOT_DIR,"F3_heatmap_proportions.png"),
                   main="Proportions heatmap", width=10, height=13)
  }
  msg("✓ Saved proportions heatmap")
}, error = function(e) {
  msg("✗ Failed to create proportions heatmap: ", e$message)
  safe_ggsave(file.path(PLOT_DIR,"F3_heatmap_proportions.png"), NULL, 10, 13,
              title_when_empty=paste("Heatmap failed:", e$message))
})

# --- Violin/box of CT distributions (with error handling)
tryCatch({
g_boxv <- ggplot2::ggplot(props_long, ggplot2::aes(cell_type, prop, fill=cell_type)) +
  ggplot2::geom_violin(trim=FALSE, alpha=.85) + ggplot2::geom_boxplot(width=.12, outlier.size=.4) +
  viridis::scale_fill_viridis(discrete=TRUE, guide="none") + theme_pub() +
  ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45, hjust=1)) +
  ggplot2::labs(title="Distribution of Bisque-estimated proportions", x="Cell type", y="Proportion")
ggplot2::ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), g_boxv, width=11, height=6.5, dpi=200)
  msg("✓ Saved box/violin plot")
}, error = function(e) {
  msg("✗ Failed to create box/violin plot: ", e$message)
  safe_ggsave(file.path(PLOT_DIR,"F3_box_violin_cohort.png"), NULL, 11, 6.5,
              title_when_empty=paste("Box/violin plot failed:", e$message))
})

## =================== ADVANCED RECONSTRUCTION & QC (from your full code) =====
# Top marker helper (used optionally below)
get_top_markers <- function(Z, n=5){
  out <- c()
  for (ct in colnames(Z)) {
    rest <- if (ncol(Z)>1) rowMeans(Z[, setdiff(colnames(Z),ct), drop=FALSE]) else 0
    fc <- (Z[, ct]+1e-6)/(rest+1e-6)
    out <- c(out, head(names(sort(fc, decreasing=TRUE)), n))
  }
  unique(out)
}

# Bulk transform helpers: regression vs shrinkage to align X_cpm scale to Yhat
transform_bulk <- function(X_cpm, Z_ct_aligned, P_ct_aligned=NULL, overlaps=NULL) {
  if (!is.null(P_ct_aligned) && !is.null(overlaps) && nrow(overlaps)>0) {
    common_ct <- intersect(colnames(Z_ct_aligned), rownames(P_ct_aligned))
    Y <- Z_ct_aligned[, common_ct, drop=FALSE] %*% P_ct_aligned[common_ct, overlaps$sc_subject, drop=FALSE]
    Xo <- X_cpm[rownames(Z_ct_aligned), overlaps$bulk_sample, drop=FALSE]
    betas <- rep(NA_real_, nrow(Z_ct_aligned)); alphas <- rep(NA_real_, nrow(Z_ct_aligned))
    for (j in seq_len(nrow(Z_ct_aligned))) {
      x <- as.numeric(Xo[j, ]); y <- as.numeric(Y[j, ])
      ok <- is.finite(x) & is.finite(y)
      if (sum(ok) >= 3) { fit <- lm(y[ok] ~ x[ok]); betas[j] <- coef(fit)[2]; alphas[j] <- coef(fit)[1] }
    }
    list(mode="regression", beta=betas, alpha=alphas, genes=rownames(Z_ct_aligned))
  } else {
    Yhat <- Z_ct_aligned %*% P_use
    muY <- rowMeans(Yhat, na.rm=TRUE)
    sdY <- apply(Yhat, 1, sd, na.rm=TRUE); sdY[!is.finite(sdY)|sdY==0] <- 1
    muX <- rowMeans(X_cpm[rownames(Z_ct_aligned), , drop=FALSE], na.rm=TRUE)
    sdX <- apply(X_cpm[rownames(Z_ct_aligned), , drop=FALSE], 1, sd, na.rm=TRUE); sdX[!is.finite(sdX)|sdX==0] <- 1
    list(mode="shrinkage", muX=muX, sdX=sdX, muY=muY, sdY=sdY, genes=rownames(Z_ct_aligned))
  }
}
apply_transform <- function(X_cpm, tf) {
  Xg <- X_cpm[tf$genes, , drop=FALSE]
  if (tf$mode == "regression") {
    A <- matrix(tf$alpha, nrow=length(tf$alpha), ncol=ncol(Xg))
    B <- matrix(tf$beta,  nrow=length(tf$beta),  ncol=ncol(Xg))
    A + B * Xg
  } else {
    Xstd <- sweep(Xg, 1, tf$muX, "-"); Xstd <- sweep(Xstd, 1, tf$sdX, "/")
    Xtr  <- sweep(Xstd, 1, tf$sdY, "*"); Xtr <- sweep(Xtr, 1, tf$muY, "+"); Xtr
  }
}

if (have_bulk) {
  Ct_use <- NULL
  if (have_sc_props) {
    ct_overlap <- intersect(rownames(Ct_raw), colnames(Z_use))
    if (length(ct_overlap)) Ct_use <- Ct_raw[ct_overlap, , drop=FALSE]
  }

  tf   <- transform_bulk(X_cpm, Z_use, P_ct_aligned = Ct_use, overlaps = NULL)
  X_tr <- apply_transform(X_cpm, tf)

  G  <- intersect(rownames(Z_use), rownames(X_cpm))
  A  <- Z_use[G, , drop=FALSE]      # genes x CT
  Yhat <- A %*% P_use               # genes x samples
  X_use <- X_tr[G, colnames(P_use), drop=FALSE]

  # Residuals
  res_norm <- sapply(colnames(P_use), function(s) sqrt(sum((X_use[, s] - Yhat[, s])^2, na.rm=TRUE)))
  res_df <- data.frame(sample=names(res_norm), residual_norm=as.numeric(res_norm))
  data.table::fwrite(res_df, file.path(TAB_DIR,"residual_norms.tsv"), sep="\t")

  # Reconstruction fit per sample
  fit_tab <- lapply(colnames(P_use), function(s) {
    obs <- X_use[, s]; pred <- Yhat[, s]
    ok <- is.finite(obs) & is.finite(pred)
    pr <- suppressWarnings(cor(obs[ok], pred[ok], method="pearson"))
    sr <- suppressWarnings(cor(obs[ok], pred[ok], method="spearman"))
    data.frame(sample=s, pearson_r=pr, R2=pr^2, spearman_rho=sr)
  })
  fit_tab <- data.table::rbindlist(fit_tab, fill=TRUE)
  data.table::fwrite(fit_tab, file.path(TAB_DIR,"reconstruction_fit.tsv"), sep="\t")

  # R² / sums / residuals dashboard + verdict
  median_R2   <- round(median(fit_tab$R2, na.rm=TRUE), 3)
  pct_strong  <- round(mean(fit_tab$R2 > 0.70, na.rm=TRUE) * 100, 1)
  pct_sum_ok  <- round(mean(qc_df$abs_dev <= 0.05, na.rm=TRUE) * 100, 1)
  res_q95     <- round(quantile(res_df$residual_norm, 0.95, na.rm=TRUE), 3)
  res_q50     <- round(quantile(res_df$residual_norm, 0.50, na.rm=TRUE), 3)
  grade <- if (median_R2 >= 0.70 && pct_strong >= 90 && pct_sum_ok >= 95) "PASS"
           else if (median_R2 >= 0.50 && pct_strong >= 70 && pct_sum_ok >= 85) "WARN" else "FAIL"
  qc_summary <- data.frame(
    Metric = c("Median R2 (reconstruction)", "% samples with R2>0.70",
               "% samples with |sum(props)-1|<=0.05", "Residual norm 95th percentile", "Residual norm median",
               "Overall verdict"),
    Value  = c(median_R2, pct_strong, pct_sum_ok, res_q95, res_q50, grade)
  )
  data.table::fwrite(qc_summary, file.path(TAB_DIR, "QC_summary.tsv"), sep="\t")

  # QC dashboard plots (with error handling)
  tryCatch({
  p1 <- ggplot2::ggplot(fit_tab, ggplot2::aes(R2)) + ggplot2::geom_histogram(bins=30, alpha=.95) +
    ggplot2::geom_vline(xintercept=0.70, linetype=2) +
    ggplot2::labs(title="Reconstruction R² across samples", x="R²", y="Count") + theme_pub()
  p2 <- ggplot2::ggplot(qc_df, ggplot2::aes(abs_dev)) + ggplot2::geom_histogram(bins=30, alpha=.95) +
    ggplot2::geom_vline(xintercept=0.05, linetype=2) +
    ggplot2::labs(title="Sum-to-one deviation per sample", x="|sum(props) − 1|", y="Count") + theme_pub()
  p3 <- ggplot2::ggplot(res_df, ggplot2::aes(residual_norm)) + ggplot2::geom_histogram(bins=30, alpha=.95) +
    ggplot2::geom_vline(xintercept=res_q50, linetype=2) + ggplot2::geom_vline(xintercept=res_q95, linetype=2) +
    ggplot2::labs(title="Residual size per sample  ||X_trans − Z×p||", x="Residual norm", y="Count") + theme_pub()
  dash <- p1 / p2 / p3 + patchwork::plot_layout(heights=c(1,1,1))
    # Save QC dashboard (optional, commented out by default to save space)
    # ggplot2::ggsave(file.path(PLOT_DIR, "QC_dashboard_simple.png"), dash, width=10, height=12, dpi=220)
    msg("✓ Created QC dashboard plots")
  }, error = function(e) {
    msg("✗ Failed to create QC dashboard plots: ", e$message)
  })
}

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
      bulk_mat2 <- ensure_double_matrix(as.matrix(bulk_dt0)); rownames(bulk_mat2) <- as.character(bgenes)
      bulk_cpm <- cpm_mat(bulk_mat2)
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

## ====================== S3 ADD-ONS: OVERVIEW + SAMPLE CARDS ==================
# Note: safe_ggsave is defined earlier in the script (before POST-HOC QC section)

#dir.create(file.path(PLOT_DIR, "S3_sample_cards"), showWarnings = FALSE, recursive = TRUE)

# ---------- S3 overview: overall composition pie ----------
overall_pie <- NULL
if (exists("props_long") && nrow(props_long)) {
  dfpie <- props_long[, .(prop = mean(prop, na.rm=TRUE)), by=cell_type]
  dfpie <- dfpie[prop > 0]
  if (nrow(dfpie)) {
    overall_pie <- ggplot2::ggplot(dfpie, ggplot2::aes(x="", y=prop, fill=cell_type)) +
      ggplot2::geom_bar(stat="identity", width=0.9) +
      ggplot2::coord_polar(theta="y") +
      viridis::scale_fill_viridis(discrete=TRUE, guide=NULL) +
      ggplot2::labs(title="Overall mean composition (across samples)", y=NULL, x=NULL) +
      ggplot2::theme_void() +
      ggplot2::theme(plot.title=ggplot2::element_text(hjust=0.5, face="bold"))
  }
}
# safe_ggsave(file.path(PLOT_DIR,"S3_overview_composition_pie.png"), overall_pie, 7, 7,
#             title_when_empty="Overall mean composition (pie)")

# ---------- S3 overview: CT prevalence across samples ----------
ct_prev_plot <- NULL
if (exists("P_use") && ncol(P_use) > 0) {
  prev <- data.table::as.data.table(P_use)
  prev[, cell_type := rownames(P_use)]
  prev_long <- data.table::melt(prev, id.vars="cell_type", variable.name="sample", value.name="prop")
  prev_ct <- prev_long[, .(prevalence = sum(prop > 0, na.rm=TRUE)), by=cell_type]
  if (nrow(prev_ct)) {
    ct_prev_plot <- ggplot2::ggplot(prev_ct, ggplot2::aes(reorder(cell_type, prevalence), prevalence)) +
      ggplot2::geom_col() + ggplot2::coord_flip() +
      ggplot2::labs(title="CT prevalence (#samples with prop>0)", x="Cell type", y="#Samples") +
      ggplot2::theme_bw()
  }
}
# safe_ggsave(file.path(PLOT_DIR,"S3_celltype_prevalence.png"), ct_prev_plot, 8, 7,
#             title_when_empty="CT prevalence")

# ---------- S3 overview: sample × sample correlation of compositions ----------
# corr_plot_path <- file.path(PLOT_DIR, "S3_props_correlation_heatmap.png")
# if (exists("prop_mat") && nrow(prop_mat) > 1 && ncol(prop_mat) > 1) {
#   cm <- suppressWarnings(cor(t(prop_mat), method="spearman", use="pairwise.complete.obs"))
#   cm[!is.finite(cm)] <- 0
#   tryCatch({
#     pheatmap::pheatmap(cm, filename=corr_plot_path, main="Sample×Sample Spearman correlation (compositions)",
#                        cluster_rows=TRUE, cluster_cols=TRUE, width=10, height=10)
#   }, error=function(e){
#     safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty=paste("Correlation heatmap:", e$message))
#   })
# } else {
#   safe_ggsave(corr_plot_path, NULL, 8, 6, title_when_empty="Correlation heatmap (insufficient data)")
# }

# ---------- S3 overview: PCA of compositions ----------
pca_plot <- NULL
if (exists("prop_mat") && nrow(prop_mat) >= 2 && ncol(prop_mat) >= 2) {
  X <- prop_mat
  X <- X[, colSums(is.finite(X)) == nrow(X), drop=FALSE]
  if (ncol(X) >= 2) {
    pc <- tryCatch(prcomp(X, center=TRUE, scale.=TRUE), error=function(e) NULL)
    if (!is.null(pc)) {
      sc <- data.frame(PC1=pc$x[,1], PC2=pc$x[,2], sample=rownames(pc$x))
      pca_plot <- ggplot2::ggplot(sc, ggplot2::aes(PC1, PC2, label=sample)) +
        ggplot2::geom_point() + ggrepel::geom_text_repel(size=3, max.overlaps=40) +
        ggplot2::theme_bw() + ggplot2::labs(title="PCA of sample compositions")
    }
  }
}
# safe_ggsave(file.path(PLOT_DIR,"S3_props_PCA.png"), pca_plot, 8, 6,
#             title_when_empty="PCA of compositions (insufficient data)")

# ---------- S1 sanity: mean signature per CT ----------
sig_mean_plot <- NULL
if (exists("Z_use") && ncol(Z_use) >= 1) {
  means <- colMeans(Z_use, na.rm=TRUE)
  dfm <- data.frame(cell_type=names(means), mean_expr=as.numeric(means))
  sig_mean_plot <- ggplot2::ggplot(dfm, ggplot2::aes(reorder(cell_type, mean_expr), mean_expr)) +
    ggplot2::geom_col() + ggplot2::coord_flip() +
    ggplot2::theme_bw() + ggplot2::labs(title="Mean signature per CT (CPM)", x="Cell type", y="Mean CPM")
}
# safe_ggsave(file.path(PLOT_DIR,"S1_signature_ct_means.png"), sig_mean_plot, 7, 6,
#             title_when_empty="Mean signature per CT")

# # ---------- S3: per-sample QC TSV ----------
# qc_out <- file.path(TAB_DIR, "S3_sample_qc.tsv")
# qc_df2 <- data.frame(sample = names(prop_sums),
#                      sum_props = as.numeric(prop_sums),
#                      abs_dev = abs(as.numeric(prop_sums) - 1))
# if (nrow(qc_df2)) {
#   data.table::fwrite(qc_df2, qc_out, sep="\t")
# } else {
#   data.table::fwrite(data.frame(), qc_out, sep="\t")
# }
# --- QC check (in-memory only; do NOT write to file) ---
qc_df2 <- data.frame(
  sample   = names(prop_sums),
  sum_props = as.numeric(prop_sums),
  abs_dev   = abs(as.numeric(prop_sums) - 1)
)

if (nrow(qc_df2)) {
  message("[QC] Computed sample QC table (", nrow(qc_df2), " samples).")
  print(head(qc_df2))  # optional preview
} else {
  message("[QC] No samples found; QC table is empty.")
  qc_df2 <- data.frame(sample = character(),
                       sum_props = numeric(),
                       abs_dev = numeric(),
                       stringsAsFactors = FALSE)
}


# ---------- NEW: robust sample-card generator (integrated) ----------
make_sample_cards <- function(Z_use,
                              P_use,
                              out_dir,
                              top_genes_per_sample = 40) {
  # ---- Hard checks & coercions ----
  to_double_matrix <- function(x) {
    if (is.data.frame(x)) x <- as.matrix(x)
    if (!is.matrix(x))   x <- as.matrix(x)
    storage.mode(x) <- "double"
    x[!is.finite(x)] <- 0
    x
  }
  Z_use <- to_double_matrix(Z_use)
  P_use <- to_double_matrix(P_use)

  if (nrow(Z_use) < 1 || ncol(Z_use) < 1) stop("Z_use has no rows/cols.")
  if (nrow(P_use) < 1 || ncol(P_use) < 1) stop("P_use has no rows/cols.")

  # ---- Align CTs ----
  CT <- intersect(colnames(Z_use), rownames(P_use))
  if (length(CT) < 2L) stop(sprintf("Too few overlapping CTs (|CT|=%d).", length(CT)))
  Z_use <- Z_use[, CT, drop = FALSE]
  P_use <- P_use[CT,  , drop = FALSE]

  # ---- Output dirs ----
  plot_root <- file.path(out_dir, "deconv_visual_reports")
  card_dir  <- file.path(plot_root, "sample_cards")
  if (!dir.exists(plot_root)) dir.create(plot_root, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(card_dir))  dir.create(card_dir,  recursive = TRUE, showWarnings = FALSE)

  sanitize <- function(x) {
    x <- gsub("[^A-Za-z0-9._-]+", "_", x)
    substr(x, 1, 80)
  }

  # clamp negatives
  Z_use[Z_use < 0 | !is.finite(Z_use)] <- 0
  P_use[P_use < 0 | !is.finite(P_use)] <- 0

  # ---- Predicted signals ----
  Yhat_all <- Z_use %*% P_use            # genes × samples
  rownames(Yhat_all) <- rownames(Z_use)

  if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required.")
  library(ggplot2)

  message(sprintf("[Sample Cards] %d genes × %d CTs; %d samples",
                  nrow(Z_use), ncol(Z_use), ncol(P_use)))

  all_rows <- list()

  # Helper: save placeholder
  save_placeholder <- function(path, title_txt) {
    png(path, width = 950, height = 650)
    par(mar=c(1,1,3,1)); plot.new()
    title(title_txt, cex.main=1.2, font.main=2)
    mtext("No positive or finite predicted gene signal available", side=3, line=-1, col="gray40")
    dev.off()
  }

  for (s in colnames(P_use)) {
    y <- as.numeric(Yhat_all[, s]); names(y) <- rownames(Yhat_all)
    y[!is.finite(y)] <- 0; y[y < 0] <- 0

    # Prefer genes with positive signal
    ord  <- order(y, decreasing = TRUE)
    gord <- names(y)[ord]
    gord <- gord[y[ord] > 0]

    # Fallback if nothing > 0
    if (!length(gord)) {
      Pvec <- as.numeric(P_use[, s]); names(Pvec) <- rownames(P_use)
      if (sum(Pvec, na.rm=TRUE) <= 0) {
        out_png <- file.path(card_dir, paste0("sample_", sanitize(s), "_top_genes.png"))
        save_placeholder(out_png, paste0("Top genes in sample: ", s))
        next
      }
      pseudo_signal <- as.numeric(Z_use %*% (Pvec / max(sum(Pvec), 1e-12)))
      names(pseudo_signal) <- rownames(Z_use)
      ord2 <- order(pseudo_signal, decreasing = TRUE)
      gord <- names(pseudo_signal)[ord2]
      gord <- gord[pseudo_signal[ord2] > 0]
      if (!length(gord)) {
        out_png <- file.path(card_dir, paste0("sample_", sanitize(s), "_top_genes.png"))
        save_placeholder(out_png, paste0("Top genes in sample: ", s))
        next
      }
      y <- pseudo_signal
    }

    n_take <- min(top_genes_per_sample, length(gord))
    g_top  <- gord[seq_len(n_take)]

    # Gene × CT contributions
    Pvec <- as.numeric(P_use[, s]); names(Pvec) <- rownames(P_use)
    Cmat <- sweep(Z_use[g_top, , drop = FALSE], 2, Pvec[colnames(Z_use)], "*")
    Cmat[!is.finite(Cmat)] <- 0
    Cmat[Cmat < 0] <- 0

    dom_ix    <- max.col(Cmat, ties.method = "first")
    dom_ct    <- colnames(Cmat)[dom_ix]
    dom_val   <- apply(Cmat, 1, max)
    totals    <- rowSums(Cmat)
    dom_share <- ifelse(totals > 0, dom_val / totals, 0)

    df <- data.frame(
      sample           = s,
      gene             = g_top,
      predicted_signal = as.numeric(y[g_top]),
      dominant_ct      = dom_ct,
      dominant_share   = round(dom_share, 3),
      stringsAsFactors = FALSE
    )
    all_rows[[s]] <- df

    ps <- df$predicted_signal; ps[!is.finite(ps)] <- 0
    rng <- max(ps) - min(ps)
    size_scaled <- if (rng <= 0) rep(3.5, nrow(df)) else 1 + 4 * (ps - min(ps)) / (rng + 1e-12)

    xmax <- max(ps, na.rm=TRUE); if (!is.finite(xmax) || xmax <= 0) xmax <- 1

    p <- ggplot(df, aes(x = predicted_signal,
                        y = reorder(gene, predicted_signal),
                        color = dominant_ct)) +
      geom_segment(aes(x = 0,
                       xend = predicted_signal,
                       yend = reorder(gene, predicted_signal)),
                   linewidth = 0.6, alpha = 0.7) +
      geom_point(aes(size = size_scaled), alpha = 0.9) +
      scale_size_continuous(range = c(2.5, 7), guide = "none") +
      coord_cartesian(xlim = c(0, xmax * 1.05)) +
      labs(
        title    = paste0("Top genes in sample: ", s),
        subtitle = "Color = dominant cell type | Size/bar = predicted gene signal (Z × p)",
        x        = "Predicted gene signal (Z × p)",
        y        = "Gene"
      ) +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(face = "bold"))

    out_png <- file.path(card_dir, paste0("sample_", sanitize(s), "_top_genes.png"))
    ggsave(out_png, p, width = 9.5, height = 6.5, dpi = 220)
  }

  out_tsv <- file.path(plot_root, "top_genes_per_sample.tsv")
  if (length(all_rows)) {
    df_all <- data.table::rbindlist(all_rows, use.names = TRUE, fill = TRUE)
    data.table::fwrite(df_all, out_tsv, sep = "\t")
    message("[Sample Cards] Wrote: ", out_tsv)
  } else {
    data.table::fwrite(data.table::as.data.table(data.frame(
      sample=character(), gene=character(), predicted_signal=double(),
      dominant_ct=character(), dominant_share=double()
    )), out_tsv, sep = "\t")
    message("[Sample Cards] No rows; wrote empty TSV: ", out_tsv)
  }

  invisible(list(
    card_dir = file.path(plot_root, "sample_cards"),
    combined_tsv = out_tsv
  ))
}

# ---------- CALL: generate robust sample cards ----------
try({
  make_sample_cards(
    Z_use = Z_use,
    P_use = P_use,
    out_dir = OUT_DIR,
    top_genes_per_sample = max(5, min(40, TOP_GENES_PER_SAMPLE))
  )
}, silent = TRUE)

# ==================== END S3 ADD-ONS =========================================

msg("DONE. Outputs written to: ", OUT_DIR)
msg("Tables → ", TAB_DIR)
msg("Plots  → ", PLOT_DIR)
