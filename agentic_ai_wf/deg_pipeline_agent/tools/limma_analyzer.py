"""
limma-voom/trend analyzer tool for DEG Pipeline Agent.
"""
import subprocess
import tempfile
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional

from .base_tool import BaseTool
from ..exceptions import AnalysisError, ValidationError


class LimmaAnalyzerTool(BaseTool):
    """Tool for running limma-voom/trend differential expression analysis."""
    
    @property
    def name(self) -> str:
        return "LimmaAnalyzer"
    
    @property
    def description(self) -> str:
        return "Run limma-voom/trend differential expression analysis"
    
    def _find_rscript(self) -> Optional[str]:
        """Find Rscript executable path."""
        # First try to find in PATH
        rscript_path = shutil.which("Rscript")
        if rscript_path:
            return rscript_path
        
        # Try common Windows R installation paths
        if os.name == 'nt':  # Windows
            common_paths = [
                r"C:\Program Files\R\R-*\bin\Rscript.exe",
                r"C:\Program Files (x86)\R\R-*\bin\Rscript.exe",
            ]
            import glob
            for pattern in common_paths:
                matches = glob.glob(pattern)
                if matches:
                    # Get the latest version
                    matches.sort(reverse=True)
                    return matches[0]
        
        return None
    
    def execute(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], 
                output_file: Union[str, Path], **kwargs) -> Dict:
        """
        Run limma analysis (auto-detects voom vs trend).
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            output_file: Path to output file
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        counts_file = Path(counts_file)
        metadata_file = Path(metadata_file)
        output_file = Path(output_file)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rscript_path = os.path.join(tmpdir, "run_limma.R")
            result_csv = os.path.join(tmpdir, "limma_output.csv")

            r_code = r"""
cat("Reading counts and metadata...\n")

# Read counts WITHOUT row.names so we can collapse duplicates safely.
counts_df <- read.csv("<<COUNTS_FILE>>", check.names=FALSE)
meta      <- read.csv("<<META_FILE>>",  check.names=FALSE)

suppressPackageStartupMessages(library(limma))
suppressPackageStartupMessages(library(edgeR))

# ---- Find/construct CONDITION column robustly ----
# Common aliases for condition/group column in biomedical metadata
condition_aliases <- c(
  "condition", "Condition", "CONDITION",
  "group", "Group", "GROUP", "group_id", "GroupID",
  "treatment", "Treatment", "TREATMENT", "treatment_group",
  "disease", "Disease", "DISEASE", "disease_status", "disease_state",
  "status", "Status", "STATUS", "sample_status",
  "class", "Class", "CLASS", "sample_class",
  "type", "Type", "TYPE", "sample_type",
  "phenotype", "Phenotype", "PHENOTYPE",
  "outcome", "Outcome", "OUTCOME",
  "label", "Label", "LABEL",
  "category", "Category", "CATEGORY",
  "cohort", "Cohort", "COHORT",
  "case_control", "Case_Control", "case_control_status",
  "control", "Control", "CONTROL",
  "disease_group", "Disease_Group"
)

# Find condition column using aliases (case-insensitive)
cond_candidates <- which(tolower(colnames(meta)) %in% tolower(condition_aliases))

if (length(cond_candidates) >= 1) {
  # Use first matching column
  cond_col_name <- colnames(meta)[cond_candidates[1]]
  meta$condition <- as.character(meta[[cond_candidates[1]]])
  cat(sprintf("✓ Found condition column: '%s'\n", cond_col_name))
} else {
  # Fallback: Intelligent detection based on column content
  # Look for columns with categorical values that might represent conditions
  cat("⚠️  No standard condition column found. Attempting intelligent detection...\n")
  
  # Strategy 1: Check for columns with few unique values (likely categorical/condition)
  unique_counts <- sapply(seq_len(ncol(meta)), function(i) {
    length(unique(meta[[i]]))
  })
  
  # Prefer columns with 2-10 unique values (typical for condition groups)
  # but exclude columns that look like IDs (too many unique values)
  n_samples <- nrow(meta)
  reasonable_range <- unique_counts >= 2 & unique_counts <= min(10, n_samples * 0.5)
  
  if (any(reasonable_range)) {
    # Among reasonable candidates, prefer columns with common condition-like values
    cond_keywords <- c("control", "disease", "case", "normal", "tumor", "healthy", 
                       "treated", "untreated", "wild", "mutant", "wt", "ko")
    
    scores <- sapply(which(reasonable_range), function(i) {
      col_vals <- tolower(as.character(meta[[i]]))
      # Count how many values contain condition keywords
      keyword_matches <- sum(sapply(cond_keywords, function(kw) any(grepl(kw, col_vals))))
      # Prefer columns with more keyword matches and fewer unique values
      keyword_matches - (unique_counts[i] / n_samples)
    })
    
    best_cond <- which(reasonable_range)[which.max(scores)]
    if (length(best_cond) == 1 && scores[which.max(scores)] > 0) {
      cond_col_name <- colnames(meta)[best_cond]
      meta$condition <- as.character(meta[[best_cond]])
      cat(sprintf("✓ Detected condition column: '%s' (intelligent detection)\n", cond_col_name))
    } else {
      # Strategy 2: Use first non-sample, non-ID column
      sample_cols <- which(tolower(colnames(meta)) %in% tolower(c("sample", "id", "sample_id", "SampleID")))
      other_cols <- setdiff(seq_len(ncol(meta)), sample_cols)
      
      if (length(other_cols) >= 1) {
        cond_col_name <- colnames(meta)[other_cols[1]]
        meta$condition <- as.character(meta[[other_cols[1]]])
        cat(sprintf("⚠️  Using first available column as condition: '%s'\n", cond_col_name))
      } else {
        stop("Unable to detect a 'condition' column from metadata. Please ensure metadata has a column indicating sample groups (e.g., 'condition', 'group', 'treatment', 'disease').")
      }
    }
  } else {
    # Last resort: Use first non-sample column
    sample_cols <- which(tolower(colnames(meta)) %in% tolower(c("sample", "id", "sample_id", "SampleID")))
    other_cols <- setdiff(seq_len(ncol(meta)), sample_cols)
    
    if (length(other_cols) >= 1) {
      cond_col_name <- colnames(meta)[other_cols[1]]
      meta$condition <- as.character(meta[[other_cols[1]]])
      cat(sprintf("⚠️  Using first available column as condition: '%s'\n", cond_col_name))
    } else {
      stop("Unable to detect a 'condition' column from metadata. Please ensure metadata has a column indicating sample groups (e.g., 'condition', 'group', 'treatment', 'disease').")
    }
  }
}

# Ensure condition column exists and is character
if (!("condition" %in% colnames(meta))) {
  stop("Failed to create 'condition' column from metadata.")
}

# Get condition column index for exclusion in sample detection
cond_col_idx <- which(colnames(meta) == "condition")[1]

# ---- Find/construct SAMPLE column robustly ----
sample_aliases <- c("sample","Sample","id","ID","sample_id","SampleID","Sample_ID","Run","run","GSM","gsm")
cand <- which(tolower(colnames(meta)) %in% tolower(sample_aliases))

if (length(cand) >= 1) {
  meta$sample <- as.character(meta[[cand[1]]])
} else {
  rn <- rownames(meta)
  if (!is.null(rn) && length(intersect(rn, colnames(counts_df))) > 0) {
    meta$sample <- rn
  } else {
    overlaps <- sapply(seq_len(ncol(meta)), function(i) {
      # Exclude condition column from sample detection
      if (!is.na(cond_col_idx) && i == cond_col_idx) return(0)
      length(intersect(colnames(counts_df), as.character(meta[[i]])))
    })
    best <- which.max(overlaps)
    if (length(best) == 1 && overlaps[best] > 0) {
      meta$sample <- as.character(meta[[best]])
    } else {
      # Exclude condition column when selecting fallback
      others <- setdiff(seq_len(ncol(meta)), if (!is.na(cond_col_idx)) cond_col_idx else integer(0))
      if (length(others) >= 1) {
        meta$sample <- as.character(meta[[others[1]]])
      } else {
        stop("Unable to infer a 'sample' column from metadata.")
      }
    }
  }
}

# ---- Separate gene column & numeric matrix; collapse duplicate genes ----
if (ncol(counts_df) < 2) stop("Counts file must have a gene column + at least one sample column.")
gene_col <- counts_df[[1]]
expr_df  <- counts_df[, -1, drop=FALSE]

# Drop HTSeq summary rows
keep_rows <- !(startsWith(as.character(gene_col), "__"))
gene_col  <- as.character(gene_col[keep_rows])
expr_df   <- expr_df[keep_rows, , drop=FALSE]

# Coerce to numeric matrix
expr_mat <- as.matrix(expr_df)
mode(expr_mat) <- "numeric"
expr_mat[is.na(expr_mat)] <- 0

# Collapse duplicate genes by sum
expr <- rowsum(expr_mat, group=gene_col, reorder=FALSE)
colnames(expr) <- make.unique(colnames(expr), sep=".")

# ---- Align columns to metadata order ----
samples <- as.character(meta$sample)
common  <- intersect(colnames(expr), samples)
if (length(common) == 0) stop("No overlap between counts columns and metadata samples.")
ord <- match(samples, colnames(expr))
if (any(is.na(ord))) {
  missing <- samples[is.na(ord)]
  stop(paste("These metadata samples were not found in counts:", paste(missing, collapse=", ")))
}
expr <- expr[, ord, drop=FALSE]

# ---- Build design/contrast ----
group <- factor(make.names(meta$condition))
cat("Group levels in R after sanitizing:", paste(levels(group), collapse=", "), "\n")
if (any(is.na(group))) stop("NAs in group variable!")
if (length(levels(group)) != 2) stop("Unified limma expects exactly two groups!")

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)
cat("Design colnames:", paste(colnames(design), collapse=", "), "\n")

contrast_str <- paste0("`", levels(group)[2], "` - `", levels(group)[1], "`")
cat("Contrast string: ", contrast_str, "\n")
cont.matrix <- makeContrasts(contrasts=contrast_str, levels=design)

# ---- Auto-detect: voom (raw counts) vs trend (normalized/log-like) ----
m <- expr
nonneg    <- all(m >= 0, na.rm=TRUE)
frac_dec  <- mean(abs(m - round(m)) > 1e-6, na.rm=TRUE)
lib_sizes <- colSums(m, na.rm=TRUE)
cv_lib    <- ifelse(all(lib_sizes > 0), sd(lib_sizes)/mean(lib_sizes), 0)
vmin      <- suppressWarnings(min(m, na.rm=TRUE))
q99       <- suppressWarnings(as.numeric(quantile(m, 0.99, na.rm=TRUE)))
file_hint <- grepl("combat|cpm|logcpm|log2", tolower(basename("<<COUNTS_FILE>>")))

method <- if (nonneg && frac_dec < 1e-6) {
  "voom"
} else if ((is.finite(vmin) && vmin < 0) || (is.finite(q99) && q99 <= 25) || file_hint || (cv_lib < 0.05)) {
  "trend"
} else {
  "voom"
}

cat(sprintf("Auto-detect → nonneg=%s, frac_dec=%.3f, cv_lib=%.3f, vmin=%.3f, q99=%.3f, hint=%s ⇒ method=%s\n",
            nonneg, frac_dec, cv_lib, vmin, q99, file_hint, method))

# ---- Fit model ----
if (method == "voom") {
  cat("Building DGEList and running voom...\n")
  dge <- DGEList(counts=expr)
  dge <- calcNormFactors(dge)
  v   <- voom(dge, design, plot=FALSE)
  fit <- lmFit(v, design)
  fit <- contrasts.fit(fit, cont.matrix)
  fit <- eBayes(fit)
  tt  <- topTable(fit, n=Inf, sort.by="none")
  tool_name <- "limma-voom"
} else {
  cat("Running limma-trend on normalized/log-scale expression...\n")
  expr2 <- expr
  if (is.finite(q99) && q99 > 50) {
    cat("Detected large-scale values; applying log2(x+0.5)\n")
    expr2 <- log2(expr2 + 0.5)
  }
  fit <- lmFit(expr2, design)
  fit <- contrasts.fit(fit, cont.matrix)
  fit <- eBayes(fit, trend=TRUE)
  tt  <- topTable(fit, n=Inf, sort.by="none")
  tool_name <- "limma-trend"
}

comparison_name <- paste0(levels(group)[2], "_vs_", levels(group)[1])

out <- data.frame(
  Gene       = rownames(tt),
  logFC      = tt$logFC,
  t          = tt$t,
  P.Value    = tt$P.Value,
  adj.P.Val  = tt$adj.P.Val,
  AveExpr    = tt$AveExpr,
  Comparison = comparison_name,
  tool       = tool_name
)

cat("Writing output CSV...\n")
write.csv(out, "<<RESULT_CSV>>", row.names=FALSE)
cat(sprintf("Unified limma finished successfully using %s.\n", tool_name))
"""
            # Convert paths to strings and normalize for Windows (R prefers forward slashes)
            counts_file_str = str(counts_file.resolve()).replace('\\', '/')
            metadata_file_str = str(metadata_file.resolve()).replace('\\', '/')
            result_csv_str = str(Path(result_csv).resolve()).replace('\\', '/')
            
            r_code = (
                r_code
                .replace("<<COUNTS_FILE>>", counts_file_str)
                .replace("<<META_FILE>>", metadata_file_str)
                .replace("<<RESULT_CSV>>", result_csv_str)
            )

            with open(rscript_path, "w", encoding='utf-8') as f:
                f.write(r_code)

            # Find Rscript executable
            rscript_exe = self._find_rscript()
            if not rscript_exe:
                raise AnalysisError(
                    "Rscript not found. Please install R and ensure Rscript is in your PATH, "
                    "or install R to a standard Windows location (C:\\Program Files\\R\\)."
                )
            
            # Run R script
            cmd = [rscript_exe, rscript_path]
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, encoding='utf-8')
                self.logger.info(output)
            except subprocess.CalledProcessError as e:
                error_msg = e.output.decode('utf-8') if isinstance(e.output, bytes) else str(e.output) if hasattr(e, 'output') else str(e)
                raise AnalysisError(f"limma (auto) failed: {error_msg}")

            try:
                df = pd.read_csv(result_csv)
                if 'tool' not in df.columns:
                    df['tool'] = 'limma-voom'
                df.to_csv(output_file, index=False)
                n_sig = int((df['adj.P.Val'] < self.config.padj_threshold).sum())
                
                # Generate summary
                summary = {
                    "n_genes": len(df),
                    "n_significant": n_sig,
                    "comparisons": [df["Comparison"].iloc[0]] if "Comparison" in df.columns else [],
                    "padj_threshold": self.config.padj_threshold,
                    "summary": f"Found {n_sig} significant genes out of {len(df)} total",
                    "tool_used": df['tool'].iloc[0] if 'tool' in df.columns else "limma-voom"
                }
                
                self.logger.info(f"✅ limma analysis completed: {n_sig} significant genes")
                return summary
            except Exception as e:
                raise AnalysisError(f"limma (auto) post-processing failed: {str(e)}")
    
    def validate_input(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], 
                      output_file: Union[str, Path], **kwargs) -> None:
        """Validate input parameters."""
        if not Path(counts_file).exists():
            raise ValidationError(f"Counts file does not exist: {counts_file}")
        
        if not Path(metadata_file).exists():
            raise ValidationError(f"Metadata file does not exist: {metadata_file}")
        
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory: {e}")

