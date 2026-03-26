# 🧬 GeneCards to Neo4j Import Guide

## Overview

This guide walks you through importing your **10GB GeneCards dataset** into Neo4j, creating a powerful biomedical knowledge graph for research and drug discovery.

## 📄 JSON File Format

The importer expects GeneCards JSON files with the following characteristics:

- **Gene Symbol**: Extracted from the filename (e.g., `BRCA1.json` → gene symbol is `BRCA1`)
- **JSON Structure**: Direct JSON object without top-level key
- **Expected Fields**: `Gene`, `Aliases`, `ExternalIdentifiers`, `Genomics`, `Summaries`, `MalaCardsDisorders`, `UnifiedDrugs`, `Pathways`, `Interactions`, `Proteins`, `Publications`, etc.

Example structure:

```json
{
  "Gene": [{"Name": "BRCA1", "Category": "protein-coding gene", "Gifts": 65, "GeneCardsId": "GC17P043044295", "Source": "HGNC", "IsApproved": true}],
  "Aliases": [{"Value": "BRCC1", "Sources": ["HGNC"]}, {"Value": "PSCP", "Sources": ["HGNC"]}],
  "ExternalIdentifiers": [{"Source": "NCBI", "Value": "672"}, {"Source": "Ensembl", "Value": "ENSG00000012048"}],
  "Genomics": [{"Hg38": {"Chromosome": "17", "Locations": [{"Start": 43044295, "End": 43125483}]}}],
  "Summaries": [{"Summary": "BRCA1 DNA repair associated", "Source": "GeneCards"}],
  "MalaCardsDisorders": [...],
  "UnifiedDrugs": [...],
  "Pathways": [...],
  "Interactions": [...],
  "Proteins": [...],
  "Publications": [...]
}
```

## 📁 Expected Data Structure

Your data should be organized like this:

```
/path/to/genecards/data/
├── AB/
│   ├── A1BG.json
│   ├── A2M.json
│   └── ...
├── XY/
│   ├── XAGE1A.json
│   ├── XRCC1.json
│   └── ...
└── ZZ/
    ├── ZZZ3.json
    └── ...
```

## 🚀 Quick Start

### 1. Validate Your Data

First, validate your dataset structure:

```bash
cd agentic_ai_wf
python neo4j_integration/test_genecards_import.py /path/to/your/genecards/data
```

### 2. Test with Small Subset

Create a test import with a few files:

```bash
# Create test directory
mkdir -p /tmp/genecards_test/AB

# Copy a few files for testing
cp /path/to/your/data/AB/*.json /tmp/genecards_test/AB/ | head -5

# Run test import
python -m neo4j_integration.genecards_importer /tmp/genecards_test --workers 2
```

### 3. Full Dataset Import

Run the full import (this will take several hours):

```bash
python -m neo4j_integration.genecards_importer /path/to/your/genecards/data \
    --workers 8 \
    --progress-interval 100
```

## 🏗️ Database Schema

### Node Types

The importer creates the following node types:

#### Gene Nodes

```cypher
(:Gene {
    symbol: "BRCA1",
    name: "BRCA1 DNA repair associated",
    category: "protein-coding gene",
    gifts_score: 58.5,
    genecards_id: "GC13P032315408",
    chromosome: "13",
    start_position: 32315408,
    end_position: 32400266,
    gene_size: 84858,
    strand: "-",
    ncbi_id: "672",
    ensembl_id: "ENSG00000012048",
    uniprot_id: "P38398"
})
```

#### Disease Nodes

```cypher
(:Disease {
    name: "Breast Cancer",
    malacards_id: "MC0000004",
    aliases: ["Mammary Carcinoma", "Breast Malignancy"],
    is_elite: true,
    is_cancer_census: true
})
```

#### Drug Nodes

```cypher
(:Drug {
    name: "Tamoxifen",
    drugbank_id: "DB00675",
    chembl_id: "CHEMBL83",
    drug_type: "small molecule",
    mechanism: "Selective estrogen receptor modulator",
    approval_status: "approved"
})
```

#### Pathway Nodes

```cypher
(:Pathway {
    name: "DNA repair",
    pathway_id: "hsa03440",
    source: "KEGG",
    category: "Genetic Information Processing"
})
```

#### Publication Nodes

```cypher
(:Publication {
    pubmed_id: "12345678",
    title: "BRCA1 mutations in breast cancer",
    year: "2023",
    journal: "Nature Genetics"
})
```

#### Protein Nodes

```cypher
(:Protein {
    uniprot_id: "P38398",
    name: "Breast cancer type 1 susceptibility protein",
    length: 1863,
    molecular_weight: 207721,
    function: "DNA repair"
})
```

### Relationship Types

The importer creates rich relationships between entities:

```cypher
// Gene-Disease associations
(gene:Gene)-[:ASSOCIATED_WITH {gene_score: 15.8, disorder_score: 12.3}]->(disease:Disease)
(gene:Gene)-[:INFERRED_ASSOCIATION {confidence: 0.8}]->(disease:Disease)

// Gene-Drug interactions
(gene:Gene)-[:INTERACTS_WITH_DRUG {interaction_type: "target"}]->(drug:Drug)

// Gene-Pathway membership
(gene:Gene)-[:BELONGS_TO_PATHWAY {evidence: "experimental"}]->(pathway:Pathway)

// Gene-Publication mentions
(gene:Gene)-[:MENTIONED_IN]->(publication:Publication)

// Gene-Protein encoding
(gene:Gene)-[:ENCODES]->(protein:Protein)

// Gene-Gene interactions
(gene1:Gene)-[:INTERACTS_WITH {interaction_type: "physical"}]->(gene2:Gene)
```

## 📊 Performance Optimization

### Neo4j Configuration

For optimal performance with large datasets, update your `neo4j.conf`:

```properties
# Memory settings
dbms.memory.heap.initial_size=4G
dbms.memory.heap.max_size=8G
dbms.memory.pagecache.size=4G

# Performance tuning
dbms.transaction.timeout=60s
dbms.query.cache_size=1000
```

### Import Settings

```bash
# For systems with more RAM/CPU
python -m neo4j_integration.genecards_importer /path/to/data \
    --workers 12 \
    --batch-size 100 \
    --progress-interval 50

# For systems with limited resources
python -m neo4j_integration.genecards_importer /path/to/data \
    --workers 2 \
    --batch-size 25 \
    --progress-interval 200
```

## 🔍 Analysis Examples

Once your data is imported, you can perform powerful analyses:

### 1. Disease Gene Discovery

Find genes most strongly associated with a disease:

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(d:Disease {name: "Breast Cancer"})
RETURN g.symbol, g.name, r.gene_score
ORDER BY r.gene_score DESC
LIMIT 20
```

### 2. Drug Target Identification

Find potential drug targets for a disease:

```cypher
MATCH (d:Disease {name: "Alzheimer Disease"})<-[:ASSOCIATED_WITH]-(g:Gene)-[:INTERACTS_WITH_DRUG]->(drug:Drug)
RETURN g.symbol, drug.name, drug.mechanism
ORDER BY g.symbol
```

### 3. Pathway Enrichment Analysis

Find pathways enriched in disease-associated genes:

```cypher
MATCH (d:Disease {name: "Diabetes"})<-[:ASSOCIATED_WITH]-(g:Gene)-[:BELONGS_TO_PATHWAY]->(p:Pathway)
WITH p, COUNT(g) as gene_count
WHERE gene_count >= 3
RETURN p.name, p.source, gene_count
ORDER BY gene_count DESC
```

### 4. Comorbidity Analysis

Find diseases that share many genes:

```cypher
MATCH (d1:Disease)<-[:ASSOCIATED_WITH]-(g:Gene)-[:ASSOCIATED_WITH]->(d2:Disease)
WHERE d1 <> d2
WITH d1, d2, COUNT(g) as shared_genes
WHERE shared_genes >= 5
RETURN d1.name, d2.name, shared_genes
ORDER BY shared_genes DESC
LIMIT 20
```

### 5. Drug Repurposing Opportunities

Find drugs that target genes associated with a different disease:

```cypher
MATCH (drug:Drug)<-[:INTERACTS_WITH_DRUG]-(g:Gene)-[:ASSOCIATED_WITH]->(d:Disease)
WHERE NOT EXISTS((drug)-[:TREATS]->(d))
WITH drug, d, COUNT(g) as target_genes
WHERE target_genes >= 2
RETURN drug.name, d.name, target_genes, drug.approval_status
ORDER BY target_genes DESC
```

### 6. Protein-Protein Interaction Networks

Find hub genes (highly connected):

```cypher
MATCH (g:Gene)-[r:INTERACTS_WITH]-()
WITH g, COUNT(r) as connections
WHERE connections >= 10
RETURN g.symbol, g.name, connections
ORDER BY connections DESC
LIMIT 50
```

### 7. Publication Impact Analysis

Find most studied genes:

```cypher
MATCH (g:Gene)-[:MENTIONED_IN]->(p:Publication)
WITH g, COUNT(p) as publication_count
WHERE publication_count >= 100
RETURN g.symbol, g.name, publication_count
ORDER BY publication_count DESC
LIMIT 30
```

### 8. Genomic Location Queries

Find genes in a specific chromosomal region:

```cypher
MATCH (g:Gene)
WHERE g.chromosome = "17"
  AND g.start_position >= 41000000
  AND g.end_position <= 42000000
RETURN g.symbol, g.name, g.start_position, g.end_position
ORDER BY g.start_position
```

### 9. Complex Multi-hop Queries

Find indirect drug-disease connections through shared pathways:

```cypher
MATCH path = (drug:Drug)<-[:INTERACTS_WITH_DRUG]-(g1:Gene)-[:BELONGS_TO_PATHWAY]->(p:Pathway)<-[:BELONGS_TO_PATHWAY]-(g2:Gene)-[:ASSOCIATED_WITH]->(d:Disease)
WHERE drug.name = "Aspirin" AND d.name = "Cardiovascular Disease"
WITH drug, d, COUNT(DISTINCT p) as shared_pathways, COLLECT(DISTINCT p.name) as pathways
WHERE shared_pathways >= 2
RETURN drug.name, d.name, shared_pathways, pathways[0..5]
```

### 10. Time-based Analysis

Analyze publication trends over time:

```cypher
MATCH (g:Gene {symbol: "TP53"})-[:MENTIONED_IN]->(p:Publication)
WHERE p.year IS NOT NULL
WITH p.year as year, COUNT(p) as publications
ORDER BY year
RETURN year, publications
```

## 🛠️ Advanced Analytics

### Graph Algorithms

Run graph algorithms to find important nodes and communities:

```cypher
// Find central genes using PageRank
CALL gds.pageRank.stream('gene-interaction-graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).symbol AS gene, score
ORDER BY score DESC
LIMIT 20

// Find gene modules/communities
CALL gds.louvain.stream('gene-interaction-graph')
YIELD nodeId, communityId
WITH communityId, COUNT(*) as size, COLLECT(gds.util.asNode(nodeId).symbol) as genes
WHERE size >= 5
RETURN communityId, size, genes[0..10]
ORDER BY size DESC
```

### Machine Learning Features

Extract features for ML models:

```cypher
// Gene feature vector
MATCH (g:Gene)
OPTIONAL MATCH (g)-[:ASSOCIATED_WITH]->(d:Disease)
OPTIONAL MATCH (g)-[:BELONGS_TO_PATHWAY]->(p:Pathway)
OPTIONAL MATCH (g)-[:INTERACTS_WITH_DRUG]->(drug:Drug)
OPTIONAL MATCH (g)-[:INTERACTS_WITH]-(other:Gene)
RETURN g.symbol,
       COUNT(DISTINCT d) as disease_count,
       COUNT(DISTINCT p) as pathway_count,
       COUNT(DISTINCT drug) as drug_count,
       COUNT(DISTINCT other) as interaction_count,
       g.gifts_score
```

## 📈 Monitoring and Maintenance

### Check Import Progress

```cypher
// Database statistics
CALL db.stats.retrieve('GRAPH COUNTS')

// Node counts by type
MATCH (n)
RETURN labels(n)[0] as node_type, COUNT(n) as count
ORDER BY count DESC

// Relationship counts by type
MATCH ()-[r]->()
RETURN type(r) as relationship_type, COUNT(r) as count
ORDER BY count DESC
```

### Data Quality Checks

```cypher
// Find genes without external IDs
MATCH (g:Gene)
WHERE g.ncbi_id IS NULL AND g.ensembl_id IS NULL
RETURN COUNT(g) as genes_without_ids

// Find orphaned nodes (no relationships)
MATCH (n:Gene)
WHERE NOT EXISTS((n)-[]-())
RETURN COUNT(n) as orphaned_genes
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Errors**

   - Increase Neo4j heap size
   - Reduce number of workers
   - Process data in smaller batches

2. **Import Slowdown**

   - Check disk space
   - Monitor CPU/memory usage
   - Restart Neo4j if needed

3. **Connection Timeouts**
   - Increase transaction timeout
   - Check network connectivity
   - Verify Neo4j service status

### Resume Interrupted Import

The importer is designed to handle interruptions gracefully:

```bash
# The importer uses MERGE operations, so re-running is safe
python -m neo4j_integration.genecards_importer /path/to/data
```

## 🎯 Expected Results

After importing 10GB of GeneCards data, you should have:

- **~20,000 genes** with comprehensive annotations
- **~10,000 diseases** with gene associations
- **~15,000 drugs** with target information
- **~5,000 pathways** with gene memberships
- **~100,000 publications** linked to genes
- **~50,000 proteins** with functional data
- **~1,000,000 relationships** connecting all entities

## 🔗 Integration with Existing Workflows

### Connect with DEG Analysis

```cypher
// Load your DEG results and connect to GeneCards
LOAD CSV WITH HEADERS FROM 'file:///deg_results.csv' AS row
MATCH (g:Gene {symbol: row.gene_symbol})
SET g.log2_fold_change = toFloat(row.log2FC),
    g.p_value = toFloat(row.p_value),
    g.is_differentially_expressed = true
```

### Pathway Enrichment Integration

```cypher
// Find enriched pathways from your analysis
MATCH (g:Gene {is_differentially_expressed: true})-[:BELONGS_TO_PATHWAY]->(p:Pathway)
WITH p, COUNT(g) as deg_count
MATCH (p)<-[:BELONGS_TO_PATHWAY]-(all_genes:Gene)
WITH p, deg_count, COUNT(all_genes) as total_genes
RETURN p.name, deg_count, total_genes,
       (deg_count * 1.0 / total_genes) as enrichment_score
ORDER BY enrichment_score DESC
```

## 🎉 Congratulations!

You now have a powerful biomedical knowledge graph ready for:

- **Drug discovery** and repurposing
- **Disease mechanism** understanding
- **Biomarker identification**
- **Pathway analysis**
- **Literature mining**
- **Precision medicine** applications

Start exploring your data with the Neo4j Browser at `http://localhost:7474`!

---

**Next Steps:**

1. Try the analysis examples above
2. Integrate with your existing research data
3. Build custom dashboards and visualizations
4. Develop machine learning models using graph features
