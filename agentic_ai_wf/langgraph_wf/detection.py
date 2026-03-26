"""Query detection for routing (e.g., single-cell vs bulk)."""


async def detect_single_cell_query(query: str, disease_name: str) -> bool:
    """Detect if query is for single-cell RNA-seq data."""
    query_lower = query.lower()
    sc_keywords = [
        "single cell", "single-cell", "scRNA", "scRNA-seq", "scRNAseq",
        "single cell rna", "10x", "cellxgene", "scrna", "sc rna",
    ]
    return any(kw in query_lower for kw in sc_keywords)
