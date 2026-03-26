"""
BiRAGAS Scenario Engine
========================
Generates UNLIMITED differential diagnosis stress test scenarios
for any disease using the Disease Knowledge Agent's taxonomy.

Scenario Generation Strategies:
1. Same organ, different mechanism
2. Same immune axis, different organ
3. Opposite immune polarity (Th1/Th17 vs Th2)
4. Cancer vs autoimmune (immune evasion vs activation)
5. Metabolic vs immune inflammation
6. Acute vs chronic temporal patterns
7. Genetic vs acquired etiology
8. Organ-specific vs systemic autoimmunity
9. Infectious vs autoimmune mimicry
10. Drug-induced vs idiopathic

Each scenario includes:
- Disease A (confirmed) and Disease B (excluded)
- Expected causal pathway for each
- Rule-in and rule-out biomarkers
- Expected DAG topology distinction
- BiRAGAS phase-level validation criteria
"""

import logging
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("biragas.scenario_engine")


@dataclass
class UniversalScenario:
    """A single differential diagnosis scenario."""
    scenario_id: str = ""
    disease_a: str = ""
    disease_b: str = ""
    category_a: str = ""
    category_b: str = ""
    relationship_type: str = ""
    causal_distinction: str = ""
    expected_dag_overlap: str = "low"

    # Biomarker expectations
    rule_in_a: List[str] = field(default_factory=list)
    rule_out_b: List[str] = field(default_factory=list)

    # Phase-level validation criteria
    expect_topology_distinct: bool = True
    expect_genetic_distinct: bool = True
    expect_propagation_distinct: bool = True
    expect_targets_distinct: bool = True

    # Difficulty level
    difficulty: str = "standard"  # easy, standard, hard, expert


# Disease-specific biomarker knowledge base
DISEASE_BIOMARKERS = {
    "sle": ["dsDNA", "SM", "C3_low", "C4_low", "ANA"],
    "systemic lupus erythematosus": ["dsDNA", "SM", "C3_low", "C4_low"],
    "rheumatoid arthritis": ["CCP3.1", "RF", "ESR", "CRP"],
    "ra": ["CCP3.1", "RF"],
    "type 2 diabetes": ["HbA1c", "fasting_glucose", "insulin", "C-peptide"],
    "type 1 diabetes": ["GAD65_Ab", "IA2_Ab", "ZnT8_Ab", "C-peptide_low"],
    "hashimoto's thyroiditis": ["TPO-Ab", "Tg-Ab", "TSH_elevated"],
    "graves' disease": ["TRAb", "TSH_suppressed", "FT3_elevated", "FT4_elevated"],
    "antiphospholipid syndrome": ["B2GP1", "aCL", "LAC"],
    "factor v leiden": ["APCR", "Factor_V_Leiden_mutation"],
    "myocardial infarction": ["Troponin_I", "CK-MB", "BNP"],
    "coronary artery disease": ["Troponin", "BNP", "LDL_elevated"],
    "heart failure": ["BNP_elevated", "NT-proBNP", "EF_reduced"],
    "stroke": ["CT_findings", "MRI_DWI", "NIHSS"],
    "pancreatic cancer": ["CA19-9", "CEA", "CT_mass"],
    "melanoma": ["S100B", "LDH", "BRAF_V600E"],
    "breast cancer": ["CA15-3", "HER2", "ER", "PR"],
    "lung cancer": ["CEA", "CYFRA21-1", "NSE"],
    "colorectal cancer": ["CEA", "CA19-9", "MSI"],
    "prostate cancer": ["PSA", "free_PSA_ratio"],
    "ovarian cancer": ["CA125", "HE4", "ROMA"],
    "chronic kidney disease": ["eGFR_low", "creatinine", "BUN", "albumin_low"],
    "iga nephropathy": ["IgA_elevated", "hematuria", "proteinuria"],
    "lupus nephritis": ["dsDNA", "C3_low", "proteinuria", "active_sediment"],
    "fsgs": ["proteinuria_nephrotic", "INF2_mutation", "APOL1"],
    "asthma": ["IgE_elevated", "eosinophils", "FEV1_reduced", "FeNO"],
    "allergic rhinitis": ["IgE_elevated", "specific_IgE", "skin_prick"],
    "atopic dermatitis": ["IgE_elevated", "eosinophils", "SCORAD"],
    "copd": ["FEV1_FVC_reduced", "DLCO_reduced", "alpha1AT"],
    "idiopathic pulmonary fibrosis": ["HRCT_UIP", "FVC_decline", "DLCO_reduced"],
    "multiple sclerosis": ["OCB_positive", "MRI_lesions", "IgG_index"],
    "alzheimer's disease": ["Abeta42_low", "tau_elevated", "pTau", "PET_amyloid"],
    "parkinson's disease": ["DaTscan", "alpha_synuclein", "clinical_motor"],
    "crohn's disease": ["ASCA", "CRP", "fecal_calprotectin", "ileoscopy"],
    "ulcerative colitis": ["pANCA", "CRP", "fecal_calprotectin", "colonoscopy"],
    "psoriasis": ["PASI_score", "skin_biopsy", "HLA-Cw6"],
    "ankylosing spondylitis": ["HLA-B27", "sacroiliitis_MRI", "CRP"],
    "sjogren's syndrome": ["SSA_Ro", "SSB_La", "Schirmer_test"],
    "myasthenia gravis": ["AChR_Ab", "MuSK_Ab", "decrement_EMG"],
    "primary biliary cholangitis": ["M2_AMA", "ALP_elevated", "GGT_elevated"],
    "celiac disease": ["tTG_IgA", "EMA", "DGP", "villous_atrophy"],
    "obesity": ["BMI_elevated", "leptin", "insulin_resistance"],
    "nafld": ["ALT_elevated", "FIB-4", "liver_US_steatosis"],
    "gout": ["uric_acid_elevated", "MSU_crystals", "joint_aspiration"],
    "covid-19": ["SARS-CoV-2_PCR", "antigen_test", "IgG_IgM"],
    "hepatitis b": ["HBsAg", "HBeAg", "HBV_DNA"],
    "hepatitis c": ["anti-HCV", "HCV_RNA", "genotype"],
    "hiv": ["HIV_Ab", "p24_Ag", "viral_load", "CD4_count"],
    "sickle cell disease": ["HbS", "Hb_electrophoresis", "reticulocytes"],
    "cystic fibrosis": ["sweat_chloride", "CFTR_mutation", "FEV1"],
    "epilepsy": ["EEG", "MRI_brain", "seizure_semiology"],
    "hypertension": ["BP_elevated", "renin", "aldosterone"],
    "atrial fibrillation": ["ECG_AF", "CHA2DS2-VASc", "LAA_thrombus"],
    "metabolic syndrome": ["waist_circumference", "triglycerides", "HDL_low", "glucose", "BP"],
    "polycystic kidney disease": ["renal_US_cysts", "PKD1_mutation", "eGFR_decline"],
}


class ScenarioEngine:
    """
    Generates unlimited differential diagnosis scenarios for any disease.

    Uses the Disease Knowledge Agent's taxonomy to create scientifically
    valid disease pairs with expected causal distinctions.
    """

    def __init__(self, disease_knowledge_agent=None):
        self.knowledge_agent = disease_knowledge_agent
        self._scenario_counter = 0

    def generate_scenarios_for_disease(
        self,
        disease_name: str,
        max_scenarios: int = 20,
        include_difficulty: str = "all",
    ) -> List[UniversalScenario]:
        """
        Generate differential diagnosis scenarios for a specific disease.

        Args:
            disease_name: The target disease
            max_scenarios: Maximum number of scenarios to generate
            include_difficulty: "easy", "standard", "hard", "expert", or "all"

        Returns:
            List of UniversalScenario objects
        """
        scenarios = []

        if self.knowledge_agent:
            # Use the knowledge agent's negative control generator
            negatives = self.knowledge_agent.negative_gen.generate_negatives(
                disease_name, max_pairs=max_scenarios * 2
            )

            for neg in negatives:
                scenario = self._build_scenario(
                    disease_name, neg['disease_b'],
                    neg['relationship_type'],
                    neg['causal_distinction'],
                    neg.get('expected_dag_overlap', 'low'),
                )
                if scenario:
                    scenarios.append(scenario)

        # Add taxonomy-based scenarios
        taxonomy_scenarios = self._generate_from_taxonomy(disease_name)
        for ts in taxonomy_scenarios:
            if not any(s.disease_b.lower() == ts.disease_b.lower() for s in scenarios):
                scenarios.append(ts)

        # Filter by difficulty
        if include_difficulty != "all":
            scenarios = [s for s in scenarios if s.difficulty == include_difficulty]

        return scenarios[:max_scenarios]

    def generate_comprehensive_battery(
        self,
        diseases: Optional[List[str]] = None,
        max_per_disease: int = 5,
    ) -> List[UniversalScenario]:
        """
        Generate a comprehensive stress test battery across multiple diseases.

        If no diseases specified, generates scenarios for ALL known diseases.
        """
        if diseases is None:
            if self.knowledge_agent:
                diseases = self.knowledge_agent.get_supported_diseases()
            else:
                diseases = list(DISEASE_BIOMARKERS.keys())

        all_scenarios = []
        seen_pairs = set()

        for disease in diseases:
            disease_scenarios = self.generate_scenarios_for_disease(
                disease, max_scenarios=max_per_disease
            )
            for s in disease_scenarios:
                pair_key = tuple(sorted([s.disease_a.lower(), s.disease_b.lower()]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    all_scenarios.append(s)

        logger.info(f"Generated {len(all_scenarios)} unique scenarios from {len(diseases)} diseases")
        return all_scenarios

    def generate_by_category(self, category: str, max_scenarios: int = 20) -> List[UniversalScenario]:
        """Generate scenarios within a disease category (e.g., 'autoimmune', 'cancer')."""
        if not self.knowledge_agent:
            return []

        diseases = self.knowledge_agent.resolver.get_diseases_by_parent(category)
        scenarios = []

        # Pairwise within category (hardest — same parent, must distinguish)
        for d1, d2 in itertools.combinations(diseases[:10], 2):
            scenario = self._build_scenario(
                d1.replace('_', ' ').title(),
                d2.replace('_', ' ').title(),
                "intra_category",
                f"Both {category} but different molecular mechanisms",
                "moderate",
            )
            if scenario:
                scenario.difficulty = "hard"
                scenarios.append(scenario)

        return scenarios[:max_scenarios]

    def _build_scenario(
        self,
        disease_a: str,
        disease_b: str,
        relationship_type: str,
        causal_distinction: str,
        expected_overlap: str,
    ) -> Optional[UniversalScenario]:
        """Build a complete scenario with biomarkers and validation criteria."""
        self._scenario_counter += 1

        key_a = disease_a.lower().strip()
        key_b = disease_b.lower().strip()

        # Get biomarkers
        rule_in = DISEASE_BIOMARKERS.get(key_a, [])
        rule_out = DISEASE_BIOMARKERS.get(key_b, [])

        # Determine difficulty
        if expected_overlap in ("none", "low"):
            difficulty = "easy"
        elif expected_overlap in ("moderate", "moderate_upstream"):
            difficulty = "standard"
        elif expected_overlap in ("convergent_downstream", "high"):
            difficulty = "hard"
        else:
            difficulty = "standard"

        # Determine validation expectations based on relationship
        expect_genetic = relationship_type not in ("same_immune_axis_different_organ",)
        expect_propagation = True
        expect_targets = True
        expect_topology = True

        if relationship_type == "same_immune_axis_different_organ":
            expect_genetic = False  # May share some GWAS loci
            difficulty = "hard"

        # Get category info
        cat_a = ""
        cat_b = ""
        if self.knowledge_agent:
            info_a = self.knowledge_agent.resolver.resolve(disease_a)
            info_b = self.knowledge_agent.resolver.resolve(disease_b)
            cat_a = info_a.get('category', '')
            cat_b = info_b.get('category', '')

        return UniversalScenario(
            scenario_id=f"universal_{self._scenario_counter:04d}",
            disease_a=disease_a,
            disease_b=disease_b,
            category_a=cat_a,
            category_b=cat_b,
            relationship_type=relationship_type,
            causal_distinction=causal_distinction,
            expected_dag_overlap=expected_overlap,
            rule_in_a=rule_in[:5],
            rule_out_b=rule_out[:5],
            expect_topology_distinct=expect_topology,
            expect_genetic_distinct=expect_genetic,
            expect_propagation_distinct=expect_propagation,
            expect_targets_distinct=expect_targets,
            difficulty=difficulty,
        )

    def _generate_from_taxonomy(self, disease_name: str) -> List[UniversalScenario]:
        """Generate scenarios using taxonomy relationships."""
        scenarios = []
        key = disease_name.lower().strip()

        # Common cross-category pairings
        cross_category_pairs = {
            "autoimmune": [
                ("metabolic", "Autoimmune inflammation vs Metabolic inflammation"),
                ("cancer", "Immune hyperactivation vs Immune evasion"),
                ("infectious", "Autoimmune vs Post-infectious mimicry"),
                ("allergic", "Th1/Th17 autoimmunity vs Th2 allergic"),
            ],
            "cancer": [
                ("autoimmune", "Immune evasion vs Immune hyperactivation"),
                ("infectious", "Cancer vs Chronic infection"),
            ],
            "metabolic": [
                ("autoimmune", "Metabolic vs Immune-mediated"),
                ("cardiovascular", "Metabolic vs Vascular injury"),
            ],
            "cardiovascular": [
                ("autoimmune", "Acute vascular injury vs Chronic autoimmune"),
                ("metabolic", "Vascular vs Metabolic"),
            ],
            "neurological": [
                ("autoimmune", "Neurodegeneration vs Neuroimmune"),
            ],
            "renal": [
                ("autoimmune", "Metabolic renal vs Immune nephritis"),
            ],
        }

        # Determine parent category
        if self.knowledge_agent:
            info = self.knowledge_agent.resolver.resolve(disease_name)
            cat = info.get('category', '')
            cat_info = self.knowledge_agent.resolver.CATEGORY_HIERARCHY.get(cat, {})
            parent = cat_info.get('parent', '')
        else:
            parent = ""

        if parent in cross_category_pairs:
            for target_parent, distinction in cross_category_pairs[parent]:
                if self.knowledge_agent:
                    target_diseases = self.knowledge_agent.resolver.get_diseases_by_parent(target_parent)
                    for td in target_diseases[:2]:
                        scenario = self._build_scenario(
                            disease_name,
                            td.replace('_', ' ').title(),
                            f"{parent}_vs_{target_parent}",
                            distinction,
                            "low",
                        )
                        if scenario:
                            scenarios.append(scenario)

        return scenarios

    def export_scenarios_json(self, scenarios: List[UniversalScenario], filepath: str):
        """Export scenarios to JSON."""
        data = []
        for s in scenarios:
            data.append({
                'scenario_id': s.scenario_id,
                'disease_a': s.disease_a,
                'disease_b': s.disease_b,
                'category_a': s.category_a,
                'category_b': s.category_b,
                'relationship_type': s.relationship_type,
                'causal_distinction': s.causal_distinction,
                'expected_dag_overlap': s.expected_dag_overlap,
                'rule_in_a': s.rule_in_a,
                'rule_out_b': s.rule_out_b,
                'difficulty': s.difficulty,
                'validation': {
                    'topology': s.expect_topology_distinct,
                    'genetic': s.expect_genetic_distinct,
                    'propagation': s.expect_propagation_distinct,
                    'targets': s.expect_targets_distinct,
                },
            })

        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} scenarios to {filepath}")
