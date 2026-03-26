from agents import function_tool
import pandas as pd

# Lazy rpy2 import to avoid R initialization at import time
def _get_r():
    """Lazy import rpy2 - only initializes R when actually needed"""
    from rpy2.robjects import r
    return r

# pandas2ri.activate()  # Deprecated and not needed here
@function_tool
def run_limma_voom(counts_file: str, metadata_file: str, output_csv: str) -> str:
    """
    Run **limma-voom** and return a summary CSV file.
    """
    # Lazy initialize R only when function is called
    r = _get_r()
    # 1) read in R, forcing unique rownames
    r(f'''
      counts <- read.csv("{counts_file}", sep=",", row.names=1, check.names=FALSE, stringsAsFactors=FALSE)
      # dedupe duplicates:
      rownames(counts) <- make.unique(rownames(counts))
      metadata <- read.csv("{metadata_file}", sep=",", row.names=1)
      rownames(metadata) <- make.unique(rownames(metadata))
      
      library(edgeR)
      library(limma)
      group <- factor(metadata$condition)
      design <- model.matrix(~group)
      y <- DGEList(counts=counts, group=group)
      y <- calcNormFactors(y)
      v <- voom(y, design)
      fit <- lmFit(v, design)
      fit <- eBayes(fit)
      res <- topTable(fit, coef=2, adjust.method="BH", number=Inf)
      res$Gene <- rownames(res)           # Extract gene IDs
res <- res[, c("Gene", colnames(res)[!colnames(res) %in% "Gene"])]  # Put Gene column first
write.csv(res, file="{output_csv}", row.names=FALSE)
    ''')

    # return f"✅ limma-voom complete; results → `{output_csv}`."
    res = pd.read_csv(output_csv)
    
    return {
        "tool": "limma-voom",
        "comparisons": sorted(res["Comparison"].unique().tolist()),
        "n_significant": int((res["padj"] < 0.05).sum()),
        "result_csv": output_csv,
    }