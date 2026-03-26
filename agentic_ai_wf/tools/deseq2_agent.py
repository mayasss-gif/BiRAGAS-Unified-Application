import pandas as pd, os
from agents import function_tool
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds  import DeseqStats
import warnings
import numpy as np

@function_tool
def run_deseq2(counts_file: str, metadata_file: str, output_csv: str) -> str:
    """
    Run **DESeq2** and return a summary CSV file.
    """
    import pandas as pd, os
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds  import DeseqStats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
        np.seterr(all="ignore")  

        sep = "," if counts_file.endswith(".csv") else "\t"
        cnt = pd.read_csv(counts_file, sep=sep, index_col=0)
        meta= pd.read_csv(metadata_file, sep=sep, index_col=0)

        if not pd.api.types.is_integer_dtype(cnt.values.flatten()):
            return "❌ Counts not integer → cannot run DESeq2."

        if set(cnt.columns) != set(meta.index):
            return "❌ Sample ID mismatch."

        conds = list(dict.fromkeys(meta["condition"]))
        ref = "control" if "control" in conds else conds[0]

        dds = DeseqDataSet(counts=cnt.T, metadata=meta,
                        design_factors=["condition"],
                        ref_level={"condition": ref})
        dds.obs_names_make_unique()
        dds.var_names_make_unique()
        dds.deseq2()

        allres = []
        for lvl in conds:
            if lvl == ref:
                continue
            stat = DeseqStats(dds, contrast=["condition", lvl, ref])
            stat.summary()

            df = stat.results_df.dropna(subset=["padj", "log2FoldChange"])
            df["Comparison"] = f"{lvl}_vs_{ref}"
            allres.append(df)

        res = pd.concat(allres) if allres else pd.DataFrame()
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        res = res.reset_index().rename(columns={"index": "Gene"})
        res.to_csv(output_csv, index=False)

        # return f"✅ DESeq2 done. Results → {output_csv}"

        return {
        "tool": "DESeq2",
        "comparisons": sorted(res["Comparison"].unique().tolist()),
        "n_significant": int((res["padj"] < 0.05).sum()),
        "result_csv": output_csv,
    }