"""
edgeR analyzer tool for DEG Pipeline Agent.
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


class EdgeRAnalyzerTool(BaseTool):
    """Tool for running edgeR differential expression analysis."""
    
    @property
    def name(self) -> str:
        return "EdgeRAnalyzer"
    
    @property
    def description(self) -> str:
        return "Run edgeR differential expression analysis"
    
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
        Run edgeR analysis.
        
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
        
        r_script = f"""
    suppressMessages({{
      library(edgeR)
      library(limma)
    }})
    cat("Reading counts and metadata...\\n")
    counts <- read.csv("{counts_file}", row.names=1, check.names=FALSE)
    meta   <- read.csv("{metadata_file}", stringsAsFactors=FALSE)
    cat("Counts: ", dim(counts)[1], " genes x ", dim(counts)[2], " samples\\n")
    group <- as.factor(make.names(meta$condition))
    if(length(unique(group)) != 2) stop("edgeR expects exactly two groups!")
    cat("Group levels: ", levels(group), "\\n")
    cat("Filtering low-expression genes...\\n")
    myCPM <- cpm(counts)
    thresh <- myCPM > 0.5
    keep <- rowSums(thresh) >= 2
    counts <- counts[keep,]
    cat("Genes after filtering: ", nrow(counts), "\\n")
    dge <- DGEList(counts=counts, group=group)
    dge <- calcNormFactors(dge)
    dge <- estimateCommonDisp(dge)
    dge <- estimateTagwiseDisp(dge)
    cat("Running exactTest...\\n")
    et <- exactTest(dge, pair=levels(group))
    top <- topTags(et, n=nrow(counts), sort.by="none")$table
    Comparison <- paste0(levels(group)[2], "_vs_", levels(group)[1])
    out <- data.frame(
        Gene=rownames(top),
        logFC=top$logFC,
        logCPM=top$logCPM,
        PValue=top$PValue,
        FDR=top$FDR,
        Comparison=Comparison
    )
    cat("Writing output CSV...\\n")
    write.csv(out, "{output_file}", row.names=FALSE)
    cat("edgeR finished successfully.\\n")
    """

        # Find Rscript executable
        rscript_exe = self._find_rscript()
        if not rscript_exe:
            raise AnalysisError(
                "Rscript not found. Please install R and ensure Rscript is in your PATH, "
                "or install R to a standard Windows location (C:\\Program Files\\R\\)."
            )
        
        # Convert paths to strings and normalize for Windows
        counts_file_str = str(counts_file.resolve()).replace('\\', '/')
        metadata_file_str = str(metadata_file.resolve()).replace('\\', '/')
        output_file_str = str(output_file.resolve()).replace('\\', '/')
        
        # Update R script with normalized paths
        r_script = r_script.replace(str(counts_file), counts_file_str)
        r_script = r_script.replace(str(metadata_file), metadata_file_str)
        r_script = r_script.replace(str(output_file), output_file_str)
        
        # Write and run R code
        with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding='utf-8') as tf:
            tf.write(r_script)
            r_path = tf.name

        try:
            proc = subprocess.Popen(
                [rscript_exe, r_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                shell=False
            )
            stdout, stderr = proc.communicate()
            self.logger.info(stdout)
            if stderr:
                self.logger.warning(f"R WARNINGS: {stderr}")
            
            if proc.returncode != 0:
                raise AnalysisError(f"edgeR failed with return code {proc.returncode}: {stderr}")
            
            if not output_file.exists() or output_file.stat().st_size == 0:
                raise AnalysisError("edgeR output CSV missing or empty")
            
            # Load and validate results
            results_df = pd.read_csv(output_file)
            
            # Generate summary
            summary = self._generate_summary(results_df)
            
            self.logger.info(f"✅ edgeR analysis completed: {summary['n_significant']} significant genes")
            return summary
            
        except subprocess.CalledProcessError as e:
            raise AnalysisError(f"edgeR execution failed: {e}")
        except Exception as e:
            raise AnalysisError(f"edgeR analysis failed: {e}")
        finally:
            if os.path.exists(r_path):
                os.remove(r_path)
    
    def _generate_summary(self, results_df: pd.DataFrame) -> Dict:
        """Generate analysis summary."""
        if results_df.empty:
            return {
                "n_genes": 0,
                "n_significant": 0,
                "comparisons": [],
                "summary": "No results generated"
            }
        
        # Count significant results
        if "FDR" in results_df.columns:
            significant = results_df["FDR"] < self.config.padj_threshold
            n_significant = significant.sum()
        else:
            n_significant = 0
        
        # Get comparisons
        comparisons = []
        if "Comparison" in results_df.columns:
            comparisons = sorted(results_df["Comparison"].unique())
        
        return {
            "n_genes": len(results_df),
            "n_significant": int(n_significant),
            "comparisons": comparisons,
            "padj_threshold": self.config.padj_threshold,
            "summary": f"Found {n_significant} significant genes out of {len(results_df)} total"
        }
    
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

