"""
Comprehensive configuration module for the BiRAGAS Causality Framework orchestrator.

Defines all configuration dataclasses for phases, stress testing, self-correction,
learning, and the top-level orchestrator configuration.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path


@dataclass
class PhaseConfig:
    """Configuration for a single orchestrator phase."""
    enabled: bool = True
    timeout_seconds: int = 600
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    parallel_modules: bool = False
    validation_strict: bool = True


@dataclass
class StressTestScenario:
    """A single stress test scenario defining a differential diagnosis pair."""
    scenario_id: str
    disease_a: str
    disease_b: str
    rule_in_markers: List[str]
    rule_out_markers: List[str]
    causal_pathway_a: str
    causal_pathway_b: str
    expected_dag_topology: str
    expected_separation: bool = True


@dataclass
class StressTestConfig:
    """Configuration for the stress testing subsystem."""
    scenarios: List[StressTestScenario] = field(default_factory=list)
    separation_threshold: float = 0.15
    min_dag_nodes: int = 5
    min_dag_edges: int = 4
    counterfactual_threshold: float = 0.05
    run_all_phases: bool = True
    output_dir: str = "stress_test_results"
    generate_report: bool = True

    def __post_init__(self):
        """Create the 17 default stress test scenarios if none were provided."""
        if not self.scenarios:
            self.scenarios = [
                # 1. Type 2 Diabetes vs RA
                StressTestScenario(
                    scenario_id="stress_01_t2d_vs_ra",
                    disease_a="Type 2 Diabetes",
                    disease_b="Rheumatoid Arthritis",
                    rule_in_markers=["HbA1c", "Fasting_Glucose", "Insulin", "C_Peptide"],
                    rule_out_markers=["Anti_CCP", "RF", "ESR", "CRP"],
                    causal_pathway_a="Insulin_Resistance -> Beta_Cell_Dysfunction -> Hyperglycemia",
                    causal_pathway_b="Autoimmune_Activation -> Synovial_Inflammation -> Joint_Destruction",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 2. Hashimoto's vs SLE
                StressTestScenario(
                    scenario_id="stress_02_hashimotos_vs_sle",
                    disease_a="Hashimoto's Thyroiditis",
                    disease_b="Systemic Lupus Erythematosus",
                    rule_in_markers=["Anti_TPO", "Anti_Thyroglobulin", "TSH", "Free_T4"],
                    rule_out_markers=["Anti_dsDNA", "Anti_Smith", "C3", "C4"],
                    causal_pathway_a="Thyroid_Autoimmunity -> Thyroid_Destruction -> Hypothyroidism",
                    causal_pathway_b="Immune_Dysregulation -> Multi_Organ_Inflammation -> Tissue_Damage",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 3. Graves' vs RA
                StressTestScenario(
                    scenario_id="stress_03_graves_vs_ra",
                    disease_a="Graves' Disease",
                    disease_b="Rheumatoid Arthritis",
                    rule_in_markers=["TSI", "Anti_TSHR", "Free_T3", "Free_T4"],
                    rule_out_markers=["Anti_CCP", "RF", "ESR", "CRP"],
                    causal_pathway_a="TSHR_Stimulation -> Thyroid_Hyperfunction -> Thyrotoxicosis",
                    causal_pathway_b="Autoimmune_Activation -> Synovial_Inflammation -> Joint_Destruction",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 4. APS vs Factor V Leiden
                StressTestScenario(
                    scenario_id="stress_04_aps_vs_fvl",
                    disease_a="Antiphospholipid Syndrome",
                    disease_b="Factor V Leiden Thrombophilia",
                    rule_in_markers=["Anti_Cardiolipin", "Anti_Beta2GP1", "Lupus_Anticoagulant"],
                    rule_out_markers=["Factor_V_Leiden_Mutation", "Activated_Protein_C_Resistance"],
                    causal_pathway_a="Antiphospholipid_Antibodies -> Endothelial_Activation -> Thrombosis",
                    causal_pathway_b="FVL_Mutation -> APC_Resistance -> Hypercoagulability",
                    expected_dag_topology="convergent",
                    expected_separation=True,
                ),
                # 5. ACS vs Autoimmune Flare
                StressTestScenario(
                    scenario_id="stress_05_acs_vs_autoimmune_flare",
                    disease_a="Acute Coronary Syndrome",
                    disease_b="Autoimmune Flare",
                    rule_in_markers=["Troponin_I", "Troponin_T", "CK_MB", "BNP"],
                    rule_out_markers=["Anti_dsDNA", "C3", "C4", "ESR"],
                    causal_pathway_a="Plaque_Rupture -> Coronary_Thrombosis -> Myocardial_Ischemia",
                    causal_pathway_b="Immune_Activation -> Systemic_Inflammation -> Organ_Damage",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 6. Allergic Disease vs Systemic Autoimmune
                StressTestScenario(
                    scenario_id="stress_06_allergic_vs_autoimmune",
                    disease_a="Allergic Disease",
                    disease_b="Systemic Autoimmune Disease",
                    rule_in_markers=["Total_IgE", "Specific_IgE", "Eosinophils", "Tryptase"],
                    rule_out_markers=["ANA", "Anti_dsDNA", "RF", "Anti_CCP"],
                    causal_pathway_a="Allergen_Exposure -> IgE_Mediated_Response -> Mast_Cell_Degranulation",
                    causal_pathway_b="Immune_Dysregulation -> Autoantibody_Production -> Tissue_Damage",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 7. CKD vs Immune Nephritis
                StressTestScenario(
                    scenario_id="stress_07_ckd_vs_immune_nephritis",
                    disease_a="Chronic Kidney Disease",
                    disease_b="Immune-Mediated Nephritis",
                    rule_in_markers=["Creatinine", "BUN", "eGFR", "Cystatin_C"],
                    rule_out_markers=["Anti_GBM", "ANCA", "C3_Nephritic_Factor", "Anti_PLA2R"],
                    causal_pathway_a="Chronic_Injury -> Nephron_Loss -> Progressive_Fibrosis",
                    causal_pathway_b="Immune_Complex_Deposition -> Glomerular_Inflammation -> Nephritis",
                    expected_dag_topology="convergent",
                    expected_separation=True,
                ),
                # 8. Cancer vs Autoimmune
                StressTestScenario(
                    scenario_id="stress_08_cancer_vs_autoimmune",
                    disease_a="Cancer",
                    disease_b="Autoimmune Disease",
                    rule_in_markers=["CEA", "CA_19_9", "AFP", "LDH"],
                    rule_out_markers=["ANA", "Anti_dsDNA", "RF", "CRP"],
                    causal_pathway_a="Oncogenic_Mutation -> Uncontrolled_Proliferation -> Tumor_Growth",
                    causal_pathway_b="Immune_Dysregulation -> Self_Antigen_Attack -> Chronic_Inflammation",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 9. Acute Pancreatitis vs Chronic Autoimmune
                StressTestScenario(
                    scenario_id="stress_09_acute_pancreatitis_vs_chronic_autoimmune",
                    disease_a="Acute Pancreatitis",
                    disease_b="Chronic Autoimmune Pancreatitis",
                    rule_in_markers=["Lipase", "Amylase", "CRP", "WBC"],
                    rule_out_markers=["IgG4", "ANA", "Anti_Lactoferrin", "CA_II_Antibodies"],
                    causal_pathway_a="Ductal_Obstruction -> Enzyme_Autoactivation -> Pancreatic_Necrosis",
                    causal_pathway_b="IgG4_Immune_Response -> Lymphoplasmacytic_Infiltration -> Fibrosis",
                    expected_dag_topology="convergent",
                    expected_separation=True,
                ),
                # 10. RA vs Type 2 Diabetes
                StressTestScenario(
                    scenario_id="stress_10_ra_vs_t2d",
                    disease_a="Rheumatoid Arthritis",
                    disease_b="Type 2 Diabetes",
                    rule_in_markers=["Anti_CCP", "RF", "ESR", "CRP"],
                    rule_out_markers=["HbA1c", "Fasting_Glucose", "Insulin", "C_Peptide"],
                    causal_pathway_a="Autoimmune_Activation -> Synovial_Inflammation -> Joint_Destruction",
                    causal_pathway_b="Insulin_Resistance -> Beta_Cell_Dysfunction -> Hyperglycemia",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 11. SLE vs Hashimoto's
                StressTestScenario(
                    scenario_id="stress_11_sle_vs_hashimotos",
                    disease_a="Systemic Lupus Erythematosus",
                    disease_b="Hashimoto's Thyroiditis",
                    rule_in_markers=["Anti_dsDNA", "Anti_Smith", "C3", "C4"],
                    rule_out_markers=["Anti_TPO", "Anti_Thyroglobulin", "TSH", "Free_T4"],
                    causal_pathway_a="Immune_Dysregulation -> Multi_Organ_Inflammation -> Tissue_Damage",
                    causal_pathway_b="Thyroid_Autoimmunity -> Thyroid_Destruction -> Hypothyroidism",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 12. APS vs RA
                StressTestScenario(
                    scenario_id="stress_12_aps_vs_ra",
                    disease_a="Antiphospholipid Syndrome",
                    disease_b="Rheumatoid Arthritis",
                    rule_in_markers=["Anti_Cardiolipin", "Anti_Beta2GP1", "Lupus_Anticoagulant"],
                    rule_out_markers=["Anti_CCP", "RF", "ESR", "CRP"],
                    causal_pathway_a="Antiphospholipid_Antibodies -> Endothelial_Activation -> Thrombosis",
                    causal_pathway_b="Autoimmune_Activation -> Synovial_Inflammation -> Joint_Destruction",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 13. Acute Pancreatitis vs RA
                StressTestScenario(
                    scenario_id="stress_13_acute_pancreatitis_vs_ra",
                    disease_a="Acute Pancreatitis",
                    disease_b="Rheumatoid Arthritis",
                    rule_in_markers=["Lipase", "Amylase", "CRP", "WBC"],
                    rule_out_markers=["Anti_CCP", "RF", "ESR"],
                    causal_pathway_a="Ductal_Obstruction -> Enzyme_Autoactivation -> Pancreatic_Necrosis",
                    causal_pathway_b="Autoimmune_Activation -> Synovial_Inflammation -> Joint_Destruction",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 14. MI vs SLE
                StressTestScenario(
                    scenario_id="stress_14_mi_vs_sle",
                    disease_a="Myocardial Infarction",
                    disease_b="Systemic Lupus Erythematosus",
                    rule_in_markers=["Troponin_I", "Troponin_T", "CK_MB", "BNP"],
                    rule_out_markers=["Anti_dsDNA", "Anti_Smith", "C3", "C4"],
                    causal_pathway_a="Coronary_Occlusion -> Myocardial_Ischemia -> Cardiomyocyte_Death",
                    causal_pathway_b="Immune_Dysregulation -> Multi_Organ_Inflammation -> Tissue_Damage",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 15. PBC vs SLE
                StressTestScenario(
                    scenario_id="stress_15_pbc_vs_sle",
                    disease_a="Primary Biliary Cholangitis",
                    disease_b="Systemic Lupus Erythematosus",
                    rule_in_markers=["Anti_Mitochondrial", "ALP", "GGT", "IgM"],
                    rule_out_markers=["Anti_dsDNA", "Anti_Smith", "C3", "C4"],
                    causal_pathway_a="Biliary_Autoimmunity -> Bile_Duct_Destruction -> Cholestasis",
                    causal_pathway_b="Immune_Dysregulation -> Multi_Organ_Inflammation -> Tissue_Damage",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 16. Allergic Asthma vs RA
                StressTestScenario(
                    scenario_id="stress_16_allergic_asthma_vs_ra",
                    disease_a="Allergic Asthma",
                    disease_b="Rheumatoid Arthritis",
                    rule_in_markers=["Total_IgE", "Specific_IgE", "Eosinophils", "FeNO"],
                    rule_out_markers=["Anti_CCP", "RF", "ESR", "CRP"],
                    causal_pathway_a="Allergen_Sensitization -> Airway_Inflammation -> Bronchoconstriction",
                    causal_pathway_b="Autoimmune_Activation -> Synovial_Inflammation -> Joint_Destruction",
                    expected_dag_topology="divergent",
                    expected_separation=True,
                ),
                # 17. APS Thrombosis vs Cancer Thrombosis
                StressTestScenario(
                    scenario_id="stress_17_aps_thrombosis_vs_cancer_thrombosis",
                    disease_a="APS-Associated Thrombosis",
                    disease_b="Cancer-Associated Thrombosis",
                    rule_in_markers=["Anti_Cardiolipin", "Anti_Beta2GP1", "Lupus_Anticoagulant"],
                    rule_out_markers=["D_Dimer", "Fibrinogen", "CEA", "CA_125"],
                    causal_pathway_a="Antiphospholipid_Antibodies -> Endothelial_Activation -> Venous_Thrombosis",
                    causal_pathway_b="Tumor_Procoagulant -> Hypercoagulable_State -> Venous_Thrombosis",
                    expected_dag_topology="convergent",
                    expected_separation=True,
                ),
            ]


@dataclass
class SelfCorrectionConfig:
    """Configuration for the self-correction subsystem."""
    max_correction_attempts: int = 3
    confidence_floor: float = 0.40
    edge_removal_threshold: float = 0.30
    dag_validity_check: bool = True
    auto_fix_cycles: bool = True
    auto_fix_orphans: bool = True
    auto_fix_low_confidence: bool = True
    log_corrections: bool = True


@dataclass
class LearningConfig:
    """Configuration for the learning engine subsystem."""
    track_performance: bool = True
    store_history: bool = True
    history_file: str = "learning_history.json"
    adaptation_rate: float = 0.1
    min_samples_for_adaptation: int = 5
    weight_decay: float = 0.01


@dataclass
class OrchestratorConfig:
    """Top-level configuration for the BiRAGAS orchestrator."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    disease_name: str = "Disease"
    disease_node: str = "Disease_Activity"
    data_dir: str = ""
    output_dir: str = "biragas_output"
    phase_configs: Dict[str, PhaseConfig] = field(default_factory=dict)
    stress_test: StressTestConfig = field(default_factory=StressTestConfig)
    self_correction: SelfCorrectionConfig = field(default_factory=SelfCorrectionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    verbose: bool = True
    log_level: str = "INFO"
    save_intermediate: bool = True
    parallel_phases: bool = False
    llm_call: Optional[Callable] = None
    llm_model: str = "claude-sonnet-4-20250514"
    llm_temperature: float = 0.1

    def __post_init__(self):
        """Initialize default phase configs if none provided."""
        default_phases = [
            "phase1_knowledge",
            "phase2_generation",
            "phase3_causal",
            "phase4_counterfactual",
            "phase5_validation",
            "phase6_stress_test",
            "phase7_reporting",
        ]
        for phase in default_phases:
            if phase not in self.phase_configs:
                self.phase_configs[phase] = PhaseConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a dictionary.

        Excludes non-serializable fields like llm_call.
        """
        result = {}
        for fld in self.__dataclass_fields__:
            value = getattr(self, fld)
            if fld == "llm_call":
                # Callable is not JSON-serializable; store its presence as a flag
                result[fld] = value is not None
            elif fld == "phase_configs":
                result[fld] = {k: asdict(v) for k, v in value.items()}
            elif fld == "stress_test":
                result[fld] = asdict(value)
            elif fld == "self_correction":
                result[fld] = asdict(value)
            elif fld == "learning":
                result[fld] = asdict(value)
            else:
                result[fld] = value
        return result

    @classmethod
    def from_json(cls, json_path: str) -> "OrchestratorConfig":
        """Load an OrchestratorConfig from a JSON file.

        Args:
            json_path: Path to the JSON configuration file.

        Returns:
            A fully hydrated OrchestratorConfig instance.
        """
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        phase_configs = {}
        for name, pc_dict in data.get("phase_configs", {}).items():
            phase_configs[name] = PhaseConfig(**pc_dict)
        data["phase_configs"] = phase_configs

        # Reconstruct stress test config with scenarios
        st_data = data.get("stress_test", {})
        scenarios = []
        for sc_dict in st_data.get("scenarios", []):
            scenarios.append(StressTestScenario(**sc_dict))
        st_data["scenarios"] = scenarios
        data["stress_test"] = StressTestConfig(**st_data)

        # Reconstruct self-correction config
        sc_data = data.get("self_correction", {})
        data["self_correction"] = SelfCorrectionConfig(**sc_data)

        # Reconstruct learning config
        lc_data = data.get("learning", {})
        data["learning"] = LearningConfig(**lc_data)

        # llm_call cannot be deserialized from JSON
        data.pop("llm_call", None)

        return cls(**data)

    def save_json(self, json_path: str) -> None:
        """Save the configuration to a JSON file.

        Args:
            json_path: Destination file path.
        """
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
