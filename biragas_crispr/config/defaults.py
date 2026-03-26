"""
Default Configuration for BiRAGAS CRISPR Platform
===================================================
"""

DEFAULT_CONFIG = {
    # Editing Engine
    'editing': {
        'default_pam': 'NGG',
        'default_n_guides': 4,
        'min_gc': 0.30,
        'max_gc': 0.75,
        'min_on_target': 0.3,
    },

    # Screening Engine
    'screening': {
        'rra_threshold': 0.05,
        'mle_beta_threshold': 0.5,
        'bagel2_bf_threshold': 5.0,
    },

    # ACE Scoring
    'ace': {
        'streams': 15,
        'confidence_method': 'bayesian',
    },

    # Knockout Engine
    'knockout': {
        'mc_samples': 200  # Optimized: was 1000,
        'propagation_alpha': 0.15,
        'methods': ['topological', 'bayesian', 'monte_carlo',
                     'pathway', 'feedback', 'ode', 'mutual_info'],
    },

    # Mega Scale Engine
    'mega': {
        'alpha': 0.15,
        'chunk_size': 512,
        'synergy_threshold': 0.05,
        'max_combination_pairs': 10000,
    },

    # Combination Engine
    'combination': {
        'synergy_threshold': 0.05,
        'models': ['bliss', 'hsa', 'loewe', 'zip', 'epistasis', 'compensation'],
    },

    # Self-Corrector
    'corrector': {
        'max_cycle_removal': 50,
        'min_edge_confidence': 0.1,
    },

    # Pipeline Debugger
    'debugger': {
        'max_retries': 3,
    },

    # Server
    'server': {
        'host': '0.0.0.0',
        'port': 8000,
    },

    # Scale targets
    'scale': {
        'genes_brunello': 19169,
        'guides_per_gene': 4,
        'configs_per_gene': 11,
        'total_configs': 210859,
        'total_combinations': 22_229_938_881,
        'total_guides_brunello': 77441,
    },
}
