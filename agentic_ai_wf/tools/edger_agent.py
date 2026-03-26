from agents import function_tool
import pandas as pd

# Lazy rpy2 import to avoid R initialization at import time
def _get_r():
    """Lazy import rpy2 - only initializes R when actually needed"""
    from rpy2.robjects import r
    return r

# pandas2ri.activate()  # Deprecated and not needed here


@function_tool
def run_edgeR(counts_file: str, metadata_file: str, output_csv: str) -> str:
    """
    Run **edgeR** and return a summary CSV file.
    """
    # Lazy initialize R only when function is called
    r = _get_r()
    try:
        r(f'''
        library(edgeR)

        counts <- read.csv("{counts_file}", row.names=1, check.names=FALSE)
        metadata <- read.csv("{metadata_file}", row.names=1)
        rownames(metadata) <- make.unique(rownames(metadata))

        group <- factor(metadata$condition)
        levels(group) <- make.unique(levels(group))

        y <- DGEList(counts=counts, group=group)
        y <- calcNormFactors(y)

        # Filter low-expression genes
        keep <- filterByExpr(y, group=group)
        y <- y[keep, , keep.lib.sizes=FALSE]

        design <- model.matrix(~ 0 + group)  # no intercept; group-specific design
        colnames(design) <- levels(group)

        y <- estimateDisp(y, design)
        fit <- glmFit(y, design)

        all_levels <- levels(group)
        res_list <- list()

        for (i in 2:length(all_levels)) {{
            contrast_vec <- rep(0, length(all_levels))
            contrast_vec[i] <- 1
            contrast_vec[1] <- -1
            contrast <- makeContrasts(contrasts=contrast_vec, levels=design)
            lrt <- glmLRT(fit, contrast=contrast)
            top <- topTags(lrt, n=Inf)$table
            top$Comparison <- paste0(all_levels[i], "_vs_", all_levels[1])
            res_list[[length(res_list)+1]] <- top
        }}

        combined <- do.call(rbind, res_list)
        combined$Gene <- rownames(combined)
write.csv(combined, file="{output_csv}", row.names=FALSE)

        ''')
    except Exception as e:
        return f"❌ edgeR multi-contrast failed: {e}"

    # return f"✅ edgeR with multiple contrasts complete → `{output_csv}`"
    res = pd.read_csv(output_csv)
    
    return {
        "tool": "edgeR",
        "comparisons": sorted(res["Comparison"].unique().tolist()),
        "n_significant": int((res["FDR"] < 0.05).sum()),
        "result_csv": output_csv,
    }