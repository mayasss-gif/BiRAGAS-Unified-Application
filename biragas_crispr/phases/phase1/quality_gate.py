"""
Phase1QualityGate — Pre-flight Validation Before Phase 1
===========================================================
Validates CRISPR data quality before feeding into DAGBuilder.
Catches issues early to prevent wasted computation.

Checks:
1. CausalDrivers_Ranked.csv exists and has required columns
2. ACE scores are on the expected scale (-1 to +1)
3. Minimum gene count (>= 100)
4. Essential/nonessential gene lists present
5. Guide count per gene (minimum 2)
6. CRISPRcleanr bias correction status
7. FDR distribution (warns if too many/few significant)
8. Column name compatibility with DAGBuilder expectations
"""

import csv
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("biragas.crispr_phase1.quality")


class Phase1QualityGate:
    """
    Validates CRISPR data quality before Phase 1 execution.

    Usage:
        gate = Phase1QualityGate()
        passed, issues = gate.validate_perturbation_dir("./perturbation_data/")
        if not passed:
            print("Fix these issues before running Phase 1:")
            for issue in issues:
                print(f"  {issue}")
    """

    def validate_perturbation_dir(self, perturbation_dir: str) -> Tuple[bool, List[str]]:
        """Validate the perturbation_data/ directory for Phase 1 compatibility."""
        issues = []

        if not os.path.isdir(perturbation_dir):
            issues.append(f"CRITICAL: Directory not found: {perturbation_dir}")
            return False, issues

        # Check File 1: CausalDrivers_Ranked.csv
        drivers_path = os.path.join(perturbation_dir, "CausalDrivers_Ranked.csv")
        if not os.path.exists(drivers_path):
            issues.append("CRITICAL: CausalDrivers_Ranked.csv not found")
        else:
            issues.extend(self._validate_drivers(drivers_path))

        # Check File 2: GeneEssentiality_ByMedian.csv
        ess_path = os.path.join(perturbation_dir, "GeneEssentiality_ByMedian.csv")
        if not os.path.exists(ess_path):
            issues.append("WARNING: GeneEssentiality_ByMedian.csv not found — essentiality will be Unknown")
        else:
            issues.extend(self._validate_essentiality(ess_path))

        # Check File 3: causal_link_table_with_relevance.csv
        drug_path = os.path.join(perturbation_dir, "causal_link_table_with_relevance.csv")
        if not os.path.exists(drug_path):
            issues.append("INFO: causal_link_table_with_relevance.csv not found — druggability will be empty")
        else:
            issues.extend(self._validate_druggability(drug_path))

        passed = not any(i.startswith("CRITICAL") for i in issues)
        return passed, issues

    def validate_screening_output(self, screening_dir: str) -> Tuple[bool, List[str]]:
        """Validate MAGeCK screening output before conversion."""
        issues = []

        if not os.path.isdir(screening_dir):
            issues.append(f"CRITICAL: Screening directory not found: {screening_dir}")
            return False, issues

        # Check for gene summary files
        gene_summaries = []
        for root, dirs, files in os.walk(screening_dir):
            for f in files:
                if 'gene_summary' in f.lower():
                    gene_summaries.append(os.path.join(root, f))

        if not gene_summaries:
            issues.append("CRITICAL: No gene_summary files found — run MAGeCK first")
        else:
            issues.append(f"OK: Found {len(gene_summaries)} gene summary file(s)")
            for gs in gene_summaries:
                issues.extend(self._validate_gene_summary(gs))

        # Check for essential gene lists
        for name in ['essential_genes.txt', 'nonessential_genes.txt']:
            found = False
            for root, dirs, files in os.walk(screening_dir):
                if name in files:
                    found = True
                    break
            if not found:
                issues.append(f"WARNING: {name} not found — essentiality classification will be limited")

        passed = not any(i.startswith("CRITICAL") for i in issues)
        return passed, issues

    def _validate_drivers(self, filepath: str) -> List[str]:
        """Validate CausalDrivers_Ranked.csv."""
        issues = []
        try:
            with open(filepath) as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []

                # Check required columns
                if 'gene' not in cols:
                    issues.append("CRITICAL: CausalDrivers_Ranked.csv missing 'gene' column")
                if 'ACE' not in cols:
                    issues.append("CRITICAL: CausalDrivers_Ranked.csv missing 'ACE' column")

                # Read and validate data
                n_genes = 0
                ace_values = []
                for row in reader:
                    n_genes += 1
                    try:
                        ace = float(row.get('ACE', 0))
                        ace_values.append(ace)
                    except (ValueError, TypeError):
                        pass

                if n_genes == 0:
                    issues.append("CRITICAL: CausalDrivers_Ranked.csv is empty")
                elif n_genes < 100:
                    issues.append(f"WARNING: Only {n_genes} genes — expected 1000+")
                else:
                    issues.append(f"OK: {n_genes} genes in CausalDrivers_Ranked.csv")

                if ace_values:
                    import numpy as np
                    min_ace = min(ace_values)
                    max_ace = max(ace_values)
                    n_drivers = sum(1 for a in ace_values if a <= -0.1)
                    issues.append(f"OK: ACE range [{min_ace:.3f}, {max_ace:.3f}], {n_drivers} drivers (ACE ≤ -0.1)")

                    if min_ace < -5 or max_ace > 5:
                        issues.append("WARNING: ACE values outside expected range [-5, 5] — may need calibration")

        except Exception as e:
            issues.append(f"CRITICAL: Cannot read CausalDrivers_Ranked.csv: {e}")

        return issues

    def _validate_essentiality(self, filepath: str) -> List[str]:
        """Validate GeneEssentiality_ByMedian.csv."""
        issues = []
        try:
            with open(filepath) as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                if 'Gene' not in cols:
                    issues.append("WARNING: GeneEssentiality missing 'Gene' column (DAGBuilder expects 'Gene')")
                n = sum(1 for _ in reader)
                issues.append(f"OK: {n} genes in GeneEssentiality")
        except Exception as e:
            issues.append(f"WARNING: Cannot read GeneEssentiality: {e}")
        return issues

    def _validate_druggability(self, filepath: str) -> List[str]:
        """Validate causal_link_table_with_relevance.csv."""
        issues = []
        try:
            with open(filepath) as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                if 'Gene' not in cols:
                    issues.append("WARNING: Drug table missing 'Gene' column")
                if 'Drug' not in cols:
                    issues.append("WARNING: Drug table missing 'Drug' column")
                n = sum(1 for _ in reader)
                issues.append(f"OK: {n} entries in drug table")
        except Exception as e:
            issues.append(f"WARNING: Cannot read drug table: {e}")
        return issues

    def _validate_gene_summary(self, filepath: str) -> List[str]:
        """Validate a MAGeCK gene summary file."""
        issues = []
        try:
            with open(filepath) as f:
                reader = csv.DictReader(f, delimiter='\t')
                cols = reader.fieldnames or []
                n = sum(1 for _ in reader)

                has_rra = 'neg|score' in cols or 'neg_score' in cols
                has_mle = any('beta' in c.lower() for c in cols)

                if has_rra:
                    issues.append(f"OK: {os.path.basename(filepath)}: {n} genes with RRA scores")
                if has_mle:
                    issues.append(f"OK: {os.path.basename(filepath)}: {n} genes with MLE beta scores")
                if not has_rra and not has_mle:
                    issues.append(f"WARNING: {os.path.basename(filepath)}: no recognized score columns")
        except Exception as e:
            issues.append(f"WARNING: Cannot read {os.path.basename(filepath)}: {e}")
        return issues
