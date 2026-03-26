# mdp_baseline.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Iterable, Dict, List
import importlib.util, sys
import pandas as pd
from .mdp_config import CONFIG
from .mdp_logging import info, warn, trace
from .mdp_io import load_table_auto
from .mdp_enrichr import fetch_gene_sets_dict_any

def _load_module_at(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _import_baseline_stack() -> tuple[Optional[object], Optional[object]]:
    candidates = []
    for d in CONFIG.get("MDP2_DIR_CANDIDATES", []):
        candidates.append(Path(d))
    candidates += [
        Path(__file__).parent / "MDP_2",
        Path(__file__).parent,
        Path("/mnt/data"),
    ]
    for base in candidates:
        utils_p = base / "utils_io.py"
        base_p = base / "baseline_expectations.py"
        if utils_p.exists() and base_p.exists():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            try:
                utils_mod = _load_module_at("utils_io", utils_p)
                sys.modules["utils_io"] = utils_mod
                info(f"[baseline] loaded utils_io from {utils_p}")
                base_mod = _load_module_at("baseline_expectations", base_p)
                info(f"[baseline] loaded baseline_expectations from {base_p}")
                return base_mod, utils_mod
            except Exception as e:
                warn(f"[baseline] import failed at {base_p}: {trace(e)}")
    warn("[baseline] consensus module or utils_io missing; skipping baseline.")
    return None, None

def symbol_to_ensembl_map(std_frames: List[pd.DataFrame]) -> Dict[str, str]:
    try:
        cat = pd.concat(std_frames, ignore_index=True)
        if not {"ensembl_id", "gene_symbol"}.issubset(cat.columns):
            return {}
        cat = cat.dropna(subset=["ensembl_id", "gene_symbol"])
        cat["gene_symbol"] = cat["gene_symbol"].astype(str).str.upper().str.strip()
        cat["ensembl_id"] = cat["ensembl_id"].astype(str).str.strip()
        m = (
            cat.groupby(["gene_symbol", "ensembl_id"])
            .size()
            .reset_index(name="n")
            .sort_values(["gene_symbol", "n"], ascending=[True, False])
            .drop_duplicates("gene_symbol")
        )
        return dict(zip(m["gene_symbol"].values, m["ensembl_id"].values))
    except Exception as e:
        warn(f"symbol_to_ensembl_map failed: {trace(e)}")
        return {}

def gene_sets_SYMBOL_to_ENSEMBL(
    gs_sym: Mapping[str, Iterable[str]],
    sym2ens: Mapping[str, str],
) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    try:
        for pid, genes in gs_sym.items():
            ens = [sym2ens.get(str(g).upper(), None) for g in genes]
            ens = [e for e in ens if isinstance(e, str) and len(e) > 0]
            if ens:
                out[pid] = ens
        return out
    except Exception as e:
        warn(f"gene_sets_SYMBOL_to_ENSEMBL failed: {trace(e)}")
        return {}

def build_consensus_expectations(
    out_root: Path,
    data_dir: Path,
    hpa_file: str,
    gtex_file: str,
    fantom_file: str,
    pathway_libs: list[str],
    tissues: list[str],
) -> Dict[str, Path]:
    base_mod, utils_io = _import_baseline_stack()
    if base_mod is None or utils_io is None:
        return {}
    try:
        hpa_raw = load_table_auto(data_dir / hpa_file)
        gtex_raw = load_table_auto(data_dir / gtex_file)
        fantom_raw = load_table_auto(data_dir / fantom_file)
        if any(x.empty for x in [hpa_raw, gtex_raw, fantom_raw]):
            warn("[baseline] One or more baseline inputs empty; skipping.")
            return {}
        std_hpa = utils_io.standardize_long_any(hpa_raw)
        std_gtex = utils_io.standardize_long_any(gtex_raw)
        std_fantom = utils_io.standardize_long_any(fantom_raw)

        _, Z_HPA = base_mod.build_z_from_long(std_hpa)
        _, Z_GTEX = base_mod.build_z_from_long(std_gtex)
        _, Z_FANTOM = base_mod.build_z_from_long(std_fantom)

        Z_cons, _ = base_mod.build_consensus_Z(
            {"HPA": Z_HPA, "GTEx": Z_GTEX, "FANTOM": Z_FANTOM}
        )
        gs_sym = fetch_gene_sets_dict_any(pathway_libs)
        sym2ens = symbol_to_ensembl_map([std_hpa, std_gtex, std_fantom])
        gs_ens = gene_sets_SYMBOL_to_ENSEMBL(gs_sym, sym2ens)
        if not gs_ens:
            warn("[baseline] No gene sets after SYMBOL→ENSEMBL conversion; skipping.")
            return {}
        all_expect = base_mod.compute_pathway_expectations_for_all_tissues(
            Z_cons=Z_cons,
            gene_sets_ENSEMBL=gs_ens,
            min_genes=5,
            source_used="CONSENSUS",
        )
        out_dir_root = out_root / "baseline_consensus"
        out_dir_root.mkdir(parents=True, exist_ok=True)
        out_paths: Dict[str, Path] = {}
        for t in tissues:
            try:
                tkey = getattr(utils_io, "canon_tissue", lambda x: x)(t)
                if tkey not in all_expect:
                    keys = [k for k in all_expect.keys() if k.lower() == t.lower()]
                    if not keys:
                        warn(f"[baseline] tissue '{t}' not in consensus; skipping.")
                        continue
                    tkey = keys[0]
                tdir = out_dir_root / tkey
                tdir.mkdir(parents=True, exist_ok=True)
                exp = all_expect[tkey]
                exp_path = tdir / "baseline.pathway_expectations.consensus.tsv"
                exp.to_csv(exp_path, sep="\t", index=False)
                out_paths[t] = exp_path
                info(f"[baseline] {t} → {exp_path}")
            except Exception as e:
                warn(f"[baseline] write failed for {t}: {trace(e)}")
        return out_paths
    except Exception as e:
        warn(f"build_consensus_expectations failed: {trace(e)}")
        return {}
