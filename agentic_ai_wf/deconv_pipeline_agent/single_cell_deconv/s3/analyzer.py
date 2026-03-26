

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# seaborn is optional; we guard usage
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except Exception:
    _HAVE_SEABORN = False

# ---------------------------------------------------------------------
# Optional: use your package's loader if available; otherwise fallback.
# ---------------------------------------------------------------------
try:
    # If your project already exposes a central IO helper:
    # from .io_utils import load_table_auto
    from . import io_utils as _io_utils  # type: ignore
    load_table_auto = _io_utils.load_table_auto  # type: ignore[attr-defined]
except Exception:
    def load_table_auto(path: str, index_col: Optional[int] = None) -> pd.DataFrame:
        """Fallback loader: detects delimiter by extension (csv/tsv/txt) or sniffs."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".csv":
            sep = ","
        elif ext in {".tsv", ".txt"}:
            sep = "\t"
        else:
            import csv
            with open(path, "r", encoding="utf-8", newline="") as f:
                sample = f.read(8192)
            try:
                sep = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
            except Exception:
                sep = ","
        return pd.read_csv(path, sep=sep, index_col=index_col, comment="#")

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =====================================================================
# DeconvolutionAnalyzer (enhanced, but package-safe)
# =====================================================================
@dataclass
class DeconvolutionAnalyzer:
    """Analyze Bisque/NNLS deconvolution outputs and produce plots + reports.

    - Backwards-compatible with your earlier minimal class.
    - Uses package plots if available, otherwise internal plotting fallback.
    """
    output_dir: str = "deconvolution_analysis"
    plots_enabled: bool = True
    results: Dict[str, Any] = field(default_factory=dict)

    # NEW: configurable thresholds (same defaults as your rich S3)
    major_delta: float = 0.05
    high_impact_delta: float = 0.10

    # Internal state
    _plots: Any = field(default=None, init=False, repr=False)
    _meta: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.out = Path(self.output_dir)
        (self.out / "plots").mkdir(parents=True, exist_ok=True)
        (self.out / "data").mkdir(exist_ok=True)
        (self.out / "reports").mkdir(exist_ok=True)

        # Optional external plots module
        self._plots = None
        if self.plots_enabled:
            try:
                from . import plots as _plots  # type: ignore
                self._plots = _plots
            except Exception:
                self._plots = None  # Optional

        # Style
        if _HAVE_SEABORN:
            try:
                # Use stable seaborn style if present
                plt.style.use("seaborn-v0_8")
                sns.set_palette("husl")
            except Exception:
                pass

        self._meta = {
            "analysis_date": datetime.now().isoformat(),
            "version": "2.2",
            "analyzer": "Enhanced BISQUE Deconvolution Analyzer (pkg-fallback)",
        }

    # ---------------- IO & validation ----------------
    def load_and_validate_data(self, proportions_file: str) -> pd.DataFrame:
        """Load proportions table (cell_type as first col OR index)."""
        logger.info(f"Loading proportions from: {proportions_file}")
        df = load_table_auto(proportions_file, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("Proportions file seems empty or malformed.")

        if df.isna().all(axis=None):
            raise ValueError("All values are NaN after coercion—check the input file.")

        if (df < 0).any().any():
            logger.warning("Negative proportions detected; downstream stats may be unreliable.")

        sums = df.sum(axis=0, skipna=True)
        if not np.allclose(sums.fillna(0).values, 1.0, atol=0.1):
            logger.warning("Column sums not ~1.0; proceeding but please verify input.")

        self._meta["input_proportions_path"] = str(proportions_file)
        self._meta["n_cell_types"] = int(df.shape[0])
        self._meta["n_samples"] = int(df.shape[1])
        return df

    def load_metadata(
        self,
        metadata_file: str,
        props: pd.DataFrame,
        sample_col: Optional[str] = None,
        condition_col: Optional[str] = None,
        control_label: Optional[str] = None,
        patient_label: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Load metadata with intelligent column detection.
        
        Auto-detects sample/condition columns and control/patient labels if not provided.
        Compatible with all metadata files validated by deg_file_validator_tools.py.
        
        Args:
            metadata_file: Path to metadata file
            props: Proportions DataFrame (for sample matching)
            sample_col: Optional sample column name (auto-detected if None)
            condition_col: Optional condition column name (auto-detected if None)
            control_label: Optional control group label (auto-detected if None)
            patient_label: Optional patient/disease group label (auto-detected if None)
            
        Returns:
            Tuple of (patient_samples, control_samples) lists
        """
        logger.info(f"Loading metadata from: {metadata_file}")
        meta = load_table_auto(metadata_file, index_col=None)
        
        # Get available sample names from proportions
        prop_samples = list(map(str, props.columns))
        
        # STEP 1: Intelligently detect sample column
        if sample_col is None:
            # Try to find column by matching with proportions samples
            prop_samples_set = set(s.lower().strip() for s in prop_samples)
            best_match_col = None
            best_overlap = 0
            
            for col in meta.columns:
                col_values_set = set(str(v).lower().strip() for v in meta[col].astype(str))
                overlap = len(prop_samples_set.intersection(col_values_set))
                if overlap > best_overlap and overlap > len(prop_samples) * 0.5:  # >50% match
                    best_match_col = col
                    best_overlap = overlap
            
            if best_match_col:
                sample_col = best_match_col
                logger.info(f"✅ Auto-detected sample column: '{sample_col}' ({best_overlap} matches)")
            else:
                # Fallback: keyword matching
                for col in meta.columns:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in ['sample', 'id', 'sampleid', 'sample_id', 'sample_name']):
                        sample_col = col
                        logger.info(f"✅ Auto-detected sample column by keyword: '{sample_col}'")
                        break
            
            if sample_col is None:
                # Final fallback: first column
                sample_col = meta.columns[0]
                logger.warning(f"⚠️ Using first column as sample column: '{sample_col}'")
        
        # Verify sample column exists
        if sample_col not in meta.columns:
            raise ValueError(f"Sample column '{sample_col}' not found in metadata. Available: {list(meta.columns)}")
        
        # STEP 2: Intelligently detect condition column
        if condition_col is None:
            # Try to find categorical column with 2-10 unique values
            best_condition_col = None
            best_score = 0
            
            for col in meta.columns:
                if col == sample_col:
                    continue
                unique_vals = meta[col].nunique()
                if 2 <= unique_vals <= 10:
                    # Check if column name suggests condition/group
                    col_lower = col.lower()
                    keyword_match = any(kw in col_lower for kw in ['condition', 'group', 'treatment', 'disease', 'status', 'phenotype', 'class'])
                    score = unique_vals + (10 if keyword_match else 0)
                    if score > best_score:
                        best_condition_col = col
                        best_score = score
            
            if best_condition_col:
                condition_col = best_condition_col
                logger.info(f"✅ Auto-detected condition column: '{condition_col}' ({meta[condition_col].nunique()} unique values)")
            else:
                # Fallback: second column
                condition_col = meta.columns[1] if len(meta.columns) > 1 else meta.columns[0]
                logger.warning(f"⚠️ Using fallback condition column: '{condition_col}'")
        
        # Verify condition column exists
        if condition_col not in meta.columns:
            raise ValueError(f"Condition column '{condition_col}' not found in metadata. Available: {list(meta.columns)}")
        
        # Standardize column values
        meta[sample_col] = meta[sample_col].astype(str).str.strip()
        meta[condition_col] = meta[condition_col].astype(str).str.strip()
        
        # Filter to samples in proportions
        prop_cols = set(map(str, props.columns))
        meta = meta[meta[sample_col].isin(prop_cols)]
        if meta.empty:
            raise ValueError("No metadata rows match proportions samples.")
        
        # STEP 3: Intelligently detect control and patient labels
        unique_conditions = meta[condition_col].unique()
        logger.info(f"Found {len(unique_conditions)} unique conditions: {list(unique_conditions)}")
        
        if control_label is None or patient_label is None:
            # Auto-detect control vs patient labels
            detected_control = None
            detected_patient = None
            
            control_keywords = ['control', 'ctrl', 'normal', 'wt', 'wildtype', 'healthy', 'baseline']
            disease_keywords = ['disease', 'patient', 'treatment', 'treated', 'case', 'tumor', 'cancer']
            
            for condition in unique_conditions:
                condition_lower = str(condition).lower()
                
                # Check for control keywords
                if any(kw in condition_lower for kw in control_keywords):
                    detected_control = condition
                # Check for disease keywords
                elif any(kw in condition_lower for kw in disease_keywords):
                    detected_patient = condition
            
            # If only 2 conditions and one is control, the other is patient
            if len(unique_conditions) == 2:
                if detected_control and not detected_patient:
                    detected_patient = [c for c in unique_conditions if c != detected_control][0]
                elif detected_patient and not detected_control:
                    detected_control = [c for c in unique_conditions if c != detected_patient][0]
            
            # Apply detected labels
            if control_label is None:
                if detected_control:
                    control_label = detected_control
                    logger.info(f"✅ Auto-detected control label: '{control_label}'")
                else:
                    # Fallback: use first condition as control
                    control_label = unique_conditions[0]
                    logger.warning(f"⚠️ Using first condition as control: '{control_label}'")
            
            if patient_label is None:
                if detected_patient:
                    patient_label = detected_patient
                    logger.info(f"✅ Auto-detected patient/disease label: '{patient_label}'")
                else:
                    # Fallback: use second condition as patient (or first if only one)
                    patient_label = unique_conditions[1] if len(unique_conditions) > 1 else unique_conditions[0]
                    logger.warning(f"⚠️ Using fallback patient label: '{patient_label}'")
        
        # Extract sample lists
        patients = meta.loc[meta[condition_col] == patient_label, sample_col].tolist()
        controls = meta.loc[meta[condition_col] == control_label, sample_col].tolist()
        
        # Filter to samples in proportions
        patients = [s for s in patients if s in props.columns]
        controls = [s for s in controls if s in props.columns]
        
        if not controls:
            logger.warning(f"⚠️ No control samples found with label '{control_label}'. Using all non-patient samples as controls.")
            # Fallback: use all other samples as controls
            controls = [s for s in meta[sample_col].tolist() if s not in patients and s in props.columns]
        
        if not controls:
            raise ValueError(f"No control samples identified. Condition column: '{condition_col}', Control label: '{control_label}'")
        
        logger.info(f"✅ Loaded metadata: {len(controls)} controls, {len(patients)} patients")
        
        self._meta["input_metadata_path"] = str(metadata_file)
        self._meta["grouping"] = {
            "mode": "metadata",
            "sample_col": sample_col,
            "condition_col": condition_col,
            "control_label": control_label,
            "patient_label": patient_label,
            "auto_detected": {
                "sample_col": sample_col not in [None, "sample_id"],
                "condition_col": condition_col not in [None, "condition"],
                "control_label": control_label not in [None, "Control"],
                "patient_label": patient_label not in [None, "Disease"],
            }
        }
        return patients, controls

    # --------------- Cohort identification fallback ---------------
    def identify_samples(self, props: pd.DataFrame) -> Tuple[List[str], List[str]]:
        all_samples = list(map(str, props.columns))
        if not all_samples:
            return [], []

        patient_patterns = ["tw-", "patient", "case", "tumor", "disease"]
        control_patterns = ["control", "healthy", "normal", "reference", "ref"]

        pats, ctrls = [], []
        for s in all_samples:
            sl = s.lower()
            is_pat = any(k in sl for k in patient_patterns)
            is_ctl = any(k in sl for k in control_patterns)
            if is_pat and not is_ctl:
                pats.append(s)
            elif is_ctl and not is_pat:
                ctrls.append(s)
            else:
                # Default to patient when ambiguous
                pats.append(s)

        if not ctrls:
            # Fallback split
            mid = max(1, len(all_samples) // 2)
            ctrls = all_samples[:mid]
            pats = all_samples[mid:]

        self._meta["grouping"] = {
            "mode": "heuristic",
            "patient_tokens": patient_patterns,
            "control_tokens": control_patterns,
        }
        return pats, ctrls

    # ---------------- Stats ----------------
    def calculate_statistics(
        self,
        props: pd.DataFrame,
        patient_samples: List[str],
        control_samples: List[str],
    ) -> pd.DataFrame:
        logger.info("Calculating statistics...")
        if not control_samples:
            raise ValueError("Need ≥1 control sample.")

        # Means and STDs with skipna
        if len(patient_samples) == 1:
            patient_mean = props[patient_samples[0]]
            patient_std = pd.Series(0.0, index=props.index)
        else:
            patient_mean = props[patient_samples].mean(axis=1, skipna=True)
            patient_std = props[patient_samples].std(axis=1, skipna=True)

        if len(control_samples) == 1:
            control_mean = props[control_samples[0]]
            control_std = pd.Series(0.0, index=props.index)
        else:
            control_mean = props[control_samples].mean(axis=1, skipna=True)
            control_std = props[control_samples].std(axis=1, skipna=True)

        comp = pd.DataFrame(
            {
                "Patient_Mean": patient_mean,
                "Patient_Std": patient_std,
                "Control_Mean": control_mean,
                "Control_Std": control_std,
            }
        )

        comp["Absolute_Difference"] = comp["Patient_Mean"] - comp["Control_Mean"]
        comp["Relative_Difference"] = np.where(
            comp["Control_Mean"] > 0,
            comp["Absolute_Difference"] / comp["Control_Mean"],
            np.inf,
        )
        comp["Fold_Change"] = np.where(
            comp["Control_Mean"] > 0,
            comp["Patient_Mean"] / comp["Control_Mean"],
            np.inf,
        )
        comp["Log2_Fold_Change"] = np.where(
            (comp["Control_Mean"] > 0) & (comp["Patient_Mean"] > 0),
            np.log2(comp["Patient_Mean"] / comp["Control_Mean"]),
            np.where(comp["Patient_Mean"] > 0, np.inf, -np.inf),
        )

        comp["P_Value"] = np.nan
        comp["Test_Type"] = ""

        # Non-parametric tests:
        # - If both groups have ≥2: Mann–Whitney U
        # - If patient n=1 and controls ≥2: one-sample Wilcoxon (control_vals - patient_value)
        try:
            from scipy import stats  # lazy import
            if len(control_samples) >= 2:
                for ct in props.index:
                    control_vals = pd.to_numeric(
                        props.loc[ct, control_samples], errors="coerce"
                    ).dropna().values
                    if len(control_vals) < 2:
                        continue

                    if len(patient_samples) >= 2:
                        patient_vals = pd.to_numeric(
                            props.loc[ct, patient_samples], errors="coerce"
                        ).dropna().values
                        if len(patient_vals) < 2:
                            continue
                        _, p = stats.mannwhitneyu(patient_vals, control_vals, alternative="two-sided")
                        comp.loc[ct, "P_Value"] = float(p)
                        comp.loc[ct, "Test_Type"] = "Mann-Whitney U (two-sample)"
                    else:
                        pv = props.loc[ct, patient_samples[0]]
                        if pd.notna(pv):
                            # one-sample Wilcoxon against patient scalar
                            try:
                                _, p = stats.wilcoxon(
                                    control_vals - float(pv),
                                    alternative="two-sided",
                                    zero_method="wilcox",
                                )
                                comp.loc[ct, "P_Value"] = float(p)
                                comp.loc[ct, "Test_Type"] = "Wilcoxon signed-rank (one-sample vs patient)"
                            except Exception:
                                # fall-through: leave NaN
                                pass
        except Exception as e:
            logger.debug(f"Stats module issue: {e}")

        comp["Significant"] = comp["P_Value"] < 0.05
        comp = comp.reindex(comp["Absolute_Difference"].abs().sort_values(ascending=False).index)
        return comp

    # ---------------- Summaries ----------------
    def generate_summary_report(
        self,
        props: pd.DataFrame,
        comparison: pd.DataFrame,
        patient_samples: List[str],
        control_samples: List[str],
    ) -> Dict[str, Any]:
        logger.info("Generating summary dictionary...")
        major_inc = comparison[comparison["Absolute_Difference"] > self.major_delta].head(5)
        major_dec = comparison[comparison["Absolute_Difference"] < -self.major_delta].head(5)

        absent_in_patient = comparison[
            (comparison["Patient_Mean"] < 0.01) & (comparison["Control_Mean"] > 0.01)
        ]
        unique_to_patient = comparison[
            (comparison["Patient_Mean"] > 0.01) & (comparison["Control_Mean"] < 0.01)
        ]

        high_impact = comparison[comparison["Absolute_Difference"].abs() > self.high_impact_delta]
        moderate_impact = comparison[
            (comparison["Absolute_Difference"].abs() > self.major_delta)
            & (comparison["Absolute_Difference"].abs() <= self.high_impact_delta)
        ]

        stat_summary: Dict[str, Any] = {"note": "Statistical testing not performed or partial"}
        if "P_Value" in comparison.columns:
            tested = comparison["P_Value"].notna().sum()
            significant = comparison[comparison["Significant"] == True]
            stat_summary = {
                "significant_changes": int(significant.shape[0]),
                "total_tested": int(tested),
                "significant_cell_types": significant.index.tolist(),
            }

        return {
            "overview": {
                "total_cell_types": int(props.shape[0]),
                "n_patient_samples": int(len(patient_samples)),
                "n_control_samples": int(len(control_samples)),
                "patient_samples": patient_samples,
                "control_samples": control_samples,
            },
            "major_increases": {
                "count": int(major_inc.shape[0]),
                "cell_types": major_inc.index.tolist(),
                "details": major_inc[["Patient_Mean", "Control_Mean", "Absolute_Difference"]].to_dict("index"),
            },
            "major_decreases": {
                "count": int(major_dec.shape[0]),
                "cell_types": major_dec.index.tolist(),
                "details": major_dec[["Patient_Mean", "Control_Mean", "Absolute_Difference"]].to_dict("index"),
            },
            "absent_in_patient": {
                "count": int(absent_in_patient.shape[0]),
                "cell_types": absent_in_patient.index.tolist(),
            },
            "unique_to_patient": {
                "count": int(unique_to_patient.shape[0]),
                "cell_types": unique_to_patient.index.tolist(),
            },
            "statistical_analysis": stat_summary,
            "impact_assessment": {
                "high_impact": int(high_impact.shape[0]),
                "moderate_impact": int(moderate_impact.shape[0]),
                "high_impact_cell_types": high_impact.index.tolist(),
            },
        }

    # ---------------- Fallback Plotters (used only if no external plots module) ----------------
    def _create_heatmap(self, props: pd.DataFrame, out_path: Path) -> str:
        # Guard empty
        if props is None or props.empty:
            return ""

        plt.figure(figsize=(12, 8))
        if _HAVE_SEABORN:
            sns.heatmap(
                props,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                mask=props.isnull(),
                cbar_kws={"label": "Cell Type Proportion"},
                annot_kws={"size": 6},  # smaller annotation text so values are visible
            )
        else:
            # Minimal heatmap fallback (no annotations if seaborn absent)
            plt.imshow(props.values, aspect="auto")
            plt.colorbar(label="Cell Type Proportion")
            plt.yticks(range(len(props.index)), props.index, fontsize=8)
            plt.xticks(range(len(props.columns)), props.columns, rotation=45, ha="right", fontsize=8)

        plt.title("Cell Type Proportions Across Samples", fontweight="bold", fontsize=12)
        plt.xlabel("Samples", fontsize=10)
        plt.ylabel("Cell Types", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(out_path)

    def _create_comparison_barplot(
        self, comparison: pd.DataFrame, patient_samples: List[str], control_samples: List[str], out_path: Path
    ) -> str:
        plt.figure(figsize=(14, 8))
        cell_types = comparison.index[:15]
        x = np.arange(len(cell_types)); width = 0.35
        patient_vals = comparison.loc[cell_types, "Patient_Mean"]
        control_vals = comparison.loc[cell_types, "Control_Mean"]
        control_errs = comparison.loc[cell_types, "Control_Std"]
        plt.bar(x - width/2, patient_vals, width, label=f"Patient(s) (n={len(patient_samples)})", alpha=0.8)
        plt.bar(x + width/2, control_vals, width, yerr=control_errs, label=f"Controls (n={len(control_samples)})",
                alpha=0.8, capsize=5)
        if "Significant" in comparison.columns:
            for i, ct in enumerate(cell_types):
                if comparison.loc[ct, "Significant"]:
                    mh = max(patient_vals.iloc[i], control_vals.iloc[i])
                    plt.text(i, mh + 0.01, "*", ha="center", va="bottom", fontsize=16, fontweight="bold", color="red")
        plt.xlabel("Cell Types"); plt.ylabel("Proportion")
        plt.title("Patient vs Control Cell Type Proportions", fontweight="bold")
        plt.xticks(x, [ct[:20] + "..." if len(ct) > 20 else ct for ct in cell_types], rotation=45, ha="right")
        plt.legend(); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
        return str(out_path)

    def _create_boxplot(
        self, props: pd.DataFrame, patient_samples: List[str], control_samples: List[str], out_path: Path
    ) -> str:
        fig, ax = plt.subplots(figsize=(16, 8))
        control_data = props[control_samples].T
        cell_types = props.index[:15]
        box_data = [control_data[ct].values for ct in cell_types]
        bp = ax.boxplot(box_data, labels=[ct[:15] + "..." if len(ct) > 15 else ct for ct in cell_types],
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_alpha(0.7)
        for i, ct in enumerate(cell_types):
            if len(patient_samples) == 1:
                pv = props.loc[ct, patient_samples[0]]
                ax.scatter(i + 1, pv, color="red", s=100, zorder=10, label="Patient" if i == 0 else "")
            else:
                pvs = props.loc[ct, patient_samples]
                ax.scatter([i + 1] * len(pvs), pvs, color="red", s=100, zorder=10, label="Patients" if i == 0 else "")
        ax.set_xlabel("Cell Types"); ax.set_ylabel("Proportion")
        ax.set_title("Control Distributions with Patient Overlay", fontweight="bold")
        ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.3)
        if len(cell_types) > 0: ax.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
        return str(out_path)

    def _create_difference_plot(self, comparison: pd.DataFrame, out_path: Path) -> str:
        plt.figure(figsize=(14, 8))
        cell_types = comparison.index[:20]
        diffs = comparison.loc[cell_types, "Absolute_Difference"]
        colors = ["#ff7f0e" if d > 0 else "#1f77b4" for d in diffs]
        plt.bar(range(len(diffs)), diffs, color=colors, alpha=0.7)
        if "Significant" in comparison.columns:
            for i, ct in enumerate(cell_types):
                if comparison.loc[ct, "Significant"]:
                    plt.text(i, diffs.iloc[i] + 0.005 * np.sign(diffs.iloc[i]), "*",
                             ha="center", va="bottom" if diffs.iloc[i] > 0 else "top",
                             fontsize=16, fontweight="bold", color="red")
        plt.axhline(y=0, color="black", linewidth=0.8)
        plt.xlabel("Cell Types"); plt.ylabel("Proportion Difference (Patient - Control)")
        plt.title("Patient-Control Differences", fontweight="bold")
        plt.xticks(range(len(cell_types)),
                   [ct[:15] + "..." if len(ct) > 15 else ct for ct in cell_types],
                   rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
        return str(out_path)

    def _create_volcano_plot(self, comparison: pd.DataFrame, out_path: Path) -> str:
        if "P_Value" not in comparison.columns or comparison["P_Value"].isna().all():
            logger.warning("No p-values available for volcano plot")
            return ""
        plt.figure(figsize=(10, 8))
        fc = comparison["Log2_Fold_Change"].replace([np.inf, -np.inf], np.nan)
        p = comparison["P_Value"].copy()
        # Clamp zeros to smallest float to avoid -log10(0)
        p = p.mask(p <= 0, np.finfo(float).tiny)
        neglogp = -np.log10(p)
        mask = ~(fc.isna() | p.isna() | np.isinf(fc) | np.isinf(neglogp))
        fc_v, p_v = fc[mask], neglogp[mask]
        thr_fc = 1.0  # |log2FC| > 1
        thr_p = -np.log10(0.05)
        colors = []
        for a, b in zip(fc_v, p_v):
            if abs(a) > thr_fc and b > thr_p: colors.append("red")
            elif abs(a) > thr_fc: colors.append("orange")
            elif b > thr_p: colors.append("blue")
            else: colors.append("gray")
        plt.scatter(fc_v, p_v, c=colors, alpha=0.6, s=50)
        plt.axhline(y=thr_p, color="black", linestyle="--", alpha=0.5, label="p = 0.05")
        plt.axvline(x=thr_fc, color="black", linestyle="--", alpha=0.5, label="2-fold change")
        plt.axvline(x=-thr_fc, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Log2 Fold Change (Patient/Control)"); plt.ylabel("-Log10 P-Value")
        plt.title("Volcano Plot: Fold Change vs Significance", fontweight="bold")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
        return str(out_path)

    def _create_combined_plot(
        self,
        props: pd.DataFrame,
        comparison: pd.DataFrame,
        patient_samples: List[str],
        control_samples: List[str],
        out_path: Path,
    ) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        top_ct = comparison.index[:10]

        # Heatmap panel
        ax1 = axes[0, 0]
        if _HAVE_SEABORN:
            sns.heatmap(
                props.loc[top_ct],
                annot=True,
                fmt=".2f",
                cmap="viridis",
                ax=ax1,
                cbar_kws={"label": "Proportion"},
                annot_kws={"size": 6},  # smaller text here too
            )
        else:
            ax1.imshow(props.loc[top_ct].values, aspect="auto")
            ax1.set_yticks(range(len(top_ct))); ax1.set_yticklabels(top_ct)
            ax1.set_xticks(range(len(props.columns))); ax1.set_xticklabels(props.columns, rotation=45, ha="right")
        ax1.set_title("Top Changed Cell Types - Proportions"); ax1.set_xlabel("Samples"); ax1.set_ylabel("Cell Types")

        # Bar panel
        ax2 = axes[0, 1]
        x = np.arange(len(top_ct)); w = 0.35
        pv = comparison.loc[top_ct, "Patient_Mean"]; cv = comparison.loc[top_ct, "Control_Mean"]
        ax2.bar(x - w/2, pv, w, label="Patient", alpha=0.8)
        ax2.bar(x + w/2, cv, w, label="Control", alpha=0.8)
        ax2.set_xlabel("Cell Types"); ax2.set_ylabel("Proportion"); ax2.set_title("Patient vs Control Comparison")
        ax2.set_xticks(x); ax2.set_xticklabels([ct[:10] + "..." if len(ct) > 10 else ct for ct in top_ct],
                                               rotation=45, ha="right")
        ax2.legend()

        # Box + patient overlay
        ax3 = axes[1, 0]
        ctrl_df = props[control_samples].T
        box_data = [ctrl_df[ct].values for ct in top_ct]
        ax3.boxplot(box_data, labels=[ct[:10] + "..." if len(ct) > 10 else ct for ct in top_ct])
        if len(patient_samples) == 1:
            for i, ct in enumerate(top_ct):
                val = props.loc[ct, patient_samples[0]]
                ax3.scatter(i + 1, val, color="red", s=80, zorder=10)
        ax3.set_xlabel("Cell Types"); ax3.set_ylabel("Proportion"); ax3.set_title("Control Distributions + Patient")
        ax3.tick_params(axis="x", rotation=45)

        # Differences
        ax4 = axes[1, 1]
        diffs = comparison.loc[top_ct, "Absolute_Difference"]
        colors = ["#ff7f0e" if d > 0 else "#1f77b4" for d in diffs]
        ax4.bar(range(len(diffs)), diffs, color=colors, alpha=0.7)
        ax4.axhline(y=0, color="black", linewidth=0.8)
        ax4.set_xlabel("Cell Types"); ax4.set_ylabel("Difference (Patient - Control)")
        ax4.set_title("Patient-Control Differences")
        ax4.set_xticks(range(len(top_ct)))
        ax4.set_xticklabels([ct[:10] + "..." if len(ct) > 10 else ct for ct in top_ct], rotation=45, ha="right")

        plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
        return str(out_path)

    # ---------------- Save artifacts ----------------
    def save_data_outputs(
        self,
        props: pd.DataFrame,
        comparison: pd.DataFrame,
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        data_dir = self.out / "data"
        reports_dir = self.out / "reports"
        files: Dict[str, str] = {}

        p_csv = data_dir / "proportions.csv"
        p_tsv = data_dir / "proportions.tsv"
        c_csv = data_dir / "statistical_comparison.csv"
        c_tsv = data_dir / "statistical_comparison.tsv"

        props.to_csv(p_csv); props.to_csv(p_tsv, sep="\t")
        comparison.to_csv(c_csv); comparison.to_csv(c_tsv, sep="\t")

        files["proportions_csv"] = str(p_csv)
        files["proportions_tsv"] = str(p_tsv)
        files["comparison_csv"] = str(c_csv)
        files["comparison_tsv"] = str(c_tsv)

        s_json = reports_dir / "analysis_summary.json"
        with open(s_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        files["summary_json"] = str(s_json)

        m_json = reports_dir / "analysis_metadata.json"
        with open(m_json, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)
        files["metadata_json"] = str(m_json)

        return files

    def generate_text_report(
        self,
        props: pd.DataFrame,
        comparison: pd.DataFrame,
        summary: Dict[str, Any],
        patient_samples: List[str],
        control_samples: List[str],
    ) -> str:
        lines: List[str] = []
        lines.extend(
            [
                "=" * 80,
                "ENHANCED BISQUE DECONVOLUTION ANALYSIS REPORT",
                "=" * 80,
                f"Analysis Date: {self._meta.get('analysis_date')}",
                f"Analyzer Version: {self._meta.get('version')}",
                "",
                "1. ANALYSIS OVERVIEW",
                "-" * 30,
                f"Total Cell Types: {summary['overview']['total_cell_types']}",
                f"Patient Samples: {summary['overview']['n_patient_samples']} ({', '.join(patient_samples)})",
                f"Control Samples: {summary['overview']['n_control_samples']} ({', '.join(control_samples)})",
                "",
                "2. MAJOR FINDINGS",
                "-" * 20,
                "",
            ]
        )

        if summary["major_increases"]["count"] > 0:
            lines.append("TOP INCREASES IN PATIENT:")
            for ct in summary["major_increases"]["cell_types"][:5]:
                det = summary["major_increases"]["details"][ct]
                lines.append(f"  • {ct}: +{det['Absolute_Difference']:.1%}")
            lines.append("")

        if summary["major_decreases"]["count"] > 0:
            lines.append("TOP DECREASES IN PATIENT:")
            for ct in summary["major_decreases"]["cell_types"][:5]:
                det = summary["major_decreases"]["details"][ct]
                lines.append(f"  • {ct}: -{abs(det['Absolute_Difference']):.1%}")
            lines.append("")

        stat = summary.get("statistical_analysis", {})
        if "significant_changes" in stat:
            lines.extend(
                [
                    "3. STATISTICAL ANALYSIS",
                    "-" * 25,
                    f"Significant Changes: {stat.get('significant_changes', 0)}/{stat.get('total_tested', 0)} tested",
                    "",
                ]
            )
            for ct in stat.get("significant_cell_types", [])[:10]:
                p_val = comparison.loc[ct, "P_Value"]
                diff = comparison.loc[ct, "Absolute_Difference"]
                lines.append(f"  • {ct}: p={p_val:.3f}, Δ={diff:+.3f}")
            lines.append("")

        lines.extend(
            [
                "4. CLINICAL IMPACT ASSESSMENT",
                "-" * 35,
                f"High Impact Changes (>={self.high_impact_delta:.0%}): {summary['impact_assessment']['high_impact']}",
                f"Moderate Impact Changes ({self.major_delta:.0%}–{self.high_impact_delta:.0%}): "
                f"{summary['impact_assessment']['moderate_impact']}",
                "",
                "5. DETAILED COMPARISON (Top 15 by |Δ|)",
                "-" * 35,
                f"{'Cell Type':<35} {'Patient':<10} {'Control':<15} {'Difference':<12} {'P-Value':<10}",
                "-" * 85,
            ]
        )
        for ct in comparison.index[:15]:
            pv = comparison.loc[ct, "Patient_Mean"]
            cv = comparison.loc[ct, "Control_Mean"]
            df = comparison.loc[ct, "Absolute_Difference"]
            p = comparison.loc[ct, "P_Value"]
            p_str = f"{p:.3f}" if not pd.isna(p) else "N/A"
            lines.append(f"{ct:<35} {pv:<10.3f} {cv:<15.3f} {df:<12.3f} {p_str:<10}")

        lines.extend(
            [
                "",
                "6. CLINICAL INTERPRETATION GUIDELINES",
                "-" * 45,
                "• Sample quality and processing conditions",
                "• Disease state and treatment history",
                "• Tissue type and anatomical location",
                "• Single-cell reference appropriateness",
                "• Batch effects and technical confounders",
                "• Multiple testing correction for p-values",
                "",
                "IMPORTANT: Consult with domain experts for clinical significance!",
                "",
                "=" * 80,
            ]
        )

        report_path = self.out / "reports" / "analysis_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return str(report_path)

    # ---------------- Orchestrator ----------------
    def analyze(
        self,
        proportions_file: str,
        metadata_file: Optional[str] = None,
        sample_col: str = "sample",
        condition_col: str = "condition",
        control_label: str = "Control",
        patient_label: str = "Disease",
    ) -> Dict[str, Any]:
        """High-level API: load → define cohorts → stats → plots → save → report."""
        logger.info(f"Output directory: {self.out}")
        props = self.load_and_validate_data(proportions_file)

        if metadata_file:
            patients, controls = self.load_metadata(
                metadata_file, props, sample_col, condition_col, control_label, patient_label
            )
        else:
            patients, controls = self.identify_samples(props)

        if not controls:
            raise ValueError("No control samples found.")

        comparison = self.calculate_statistics(props, patients, controls)
        summary = self.generate_summary_report(props, comparison, patients, controls)

        # Plots: prefer external module; fallback to internal if unavailable
        plot_files: Dict[str, str] = {}
        plots_dir = self.out / "plots"

        if self._plots is not None:
            try:
                plot_files.update(
                    self._plots.make_default_panel(
                        props=props,
                        comparison=comparison,
                        patients=patients,
                        controls=controls,
                        out_dir=str(plots_dir),
                    )
                )
            except Exception as e:
                logger.warning(f"External plots module failed ({e}); using internal fallback.")
                self._make_internal_plots(props, comparison, patients, controls, plots_dir, plot_files)
        else:
            self._make_internal_plots(props, comparison, patients, controls, plots_dir, plot_files)

        data_files = self.save_data_outputs(props, comparison, summary)
        report_file = self.generate_text_report(props, comparison, summary, patients, controls)

        result = {
            "analysis_successful": True,
            "output_directory": str(self.out),
            "summary": summary,
            "data_files": data_files,
            "plot_files": plot_files,
            "report_file": report_file,
            "metadata": self._meta,
        }
        self.results = result
        #logger.info(" ")
        #return result

    # ----- helper to build internal plots -----
    def _make_internal_plots(
        self,
        props: pd.DataFrame,
        comparison: pd.DataFrame,
        patients: List[str],
        controls: List[str],
        plots_dir: Path,
        plot_files: Dict[str, str],
    ) -> None:
        plot_files["heatmap"] = self._create_heatmap(props, plots_dir / "01_heatmap.png")
        plot_files["comparison"] = self._create_comparison_barplot(
            comparison, patients, controls, plots_dir / "02_comparison.png"
        )
        plot_files["boxplot"] = self._create_boxplot(
            props, patients, controls, plots_dir / "03_boxplot.png"
        )
        plot_files["differences"] = self._create_difference_plot(
            comparison, plots_dir / "04_differences.png"
        )
        plot_files["combined"] = self._create_combined_plot(
            props, comparison, patients, controls, plots_dir / "05_combined_analysis.png"
        )
        v = self._create_volcano_plot(comparison, plots_dir / "06_volcano.png")
        if v:
            plot_files["volcano"] = v
        logger.info("Cellular deconvolution analysis completed")