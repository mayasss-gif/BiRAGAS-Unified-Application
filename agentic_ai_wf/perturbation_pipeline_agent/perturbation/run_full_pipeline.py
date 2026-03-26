from __future__ import annotations
from .run_depmap import run_depmap_pipeline
from .run_l1000 import run_l1000_pipeline
from .run_integration import run_integration_pipeline
from .l1000.build_user_input import DEFAULT_MAX_SIGS

from pathlib import Path

from .logging_utils import setup_logger



dep_map_addons = {
    "mode_model" : None,
    "genes_selection" : "all",
    "top_up" : None,
    "top_down" : None,
}


l1000_addons = {
    "tissue" : None,
    "drug" : None,
    "time_points" : None,
    "cell_lines" : None,
    "max_sigs" : DEFAULT_MAX_SIGS,
}

def run_full_pipeline(
    raw_deg_path: Path,
    pathway_path: Path,
    output_dir: Path,
    disease: str,
    dep_map_addons: dict = dep_map_addons,
    l1000_addons: dict = l1000_addons,
    max_sigs: int | None = DEFAULT_MAX_SIGS,
):

    """
    Runs the full pipeline.

    Args:
        raw_deg_path: Path,
        pathway_path: Path,
        output_dir: Path,
        disease: str,
        dep_map_addons: dict = dep_map_addons,
            # Example:
            # dep_map_addons = {
            #     "mode_model": "by_disease", # "by_lineage", "by_ids", "by_names", "keyword"
            #     "genes_selection": "top", # "all"
            #     "top_up": 10, # None
            #     "top_down": 15, # None
            # }
        l1000_addons: dict = l1000_addons,
            # Example:
            # l1000_addons = {
            #     "tissue": "liver", # None
            #     "drug": "doxorubicin", # None
            #     "time_points": [6, 24], # None
            #     "cell_lines": ["HepG2", "Huh7"], # None
            #     "max_sigs": 250000 # None
            # }
        max_sigs: int | None = DEFAULT_MAX_SIGS,
            # Forces max_sigs; set to None to allow LLM choice.


    """

    output_dir.mkdir(parents=True, exist_ok=True)

    depmap_output_dir = output_dir / "depmap"
    depmap_output_dir.mkdir(parents=True, exist_ok=True)
    l1000_output_dir = output_dir / "l1000"
    l1000_output_dir.mkdir(parents=True, exist_ok=True)
    integration_output_dir = output_dir / "integration"
    integration_output_dir.mkdir(parents=True, exist_ok=True)


    logger = setup_logger(log_dir=output_dir, name="Full Pipeline")

    logger.info(f"Running Full Pipeline with raw DEG file: {raw_deg_path}")
    logger.info(f"Running Full Pipeline with pathway file: {pathway_path}")



    run_depmap_pipeline(
        raw_deg_path=raw_deg_path,
        output_dir=depmap_output_dir,
        disease=disease,
        mode_model=dep_map_addons["mode_model"],
        genes_selection=dep_map_addons["genes_selection"],
        top_up=dep_map_addons["top_up"],
        top_down=dep_map_addons["top_down"],
    )

    logger.info(f"DEPMAP pipeline completed successfully")
    logger.info(f"Running L1000 pipeline")

    run_l1000_pipeline(
        deg_path=raw_deg_path,
        pathway_path=pathway_path,
        output_dir=l1000_output_dir,
        disease=disease,
        tissue=l1000_addons["tissue"],
        drug=l1000_addons["drug"],
        time_points=l1000_addons["time_points"],
        cell_lines=l1000_addons["cell_lines"],
        max_sigs=max_sigs if max_sigs is not None else l1000_addons.get("max_sigs"),
    )

    logger.info(f"L1000 pipeline completed successfully")
    logger.info(f"Running Integration pipeline")

    run_integration_pipeline(
        deg_path=raw_deg_path,
        output_dir=integration_output_dir,
        l1000_path=l1000_output_dir,
        depmap_path=depmap_output_dir,
    )
    logger.info(f"Integration pipeline completed successfully")

    logger.info(f"Full Pipeline completed successfully")
