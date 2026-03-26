# Neo4j Integration Module

A comprehensive Neo4j integration module for biomedical data analysis within the Agentic AI Workflow.

## Overview

This module provides a complete solution for storing, querying, and analyzing biomedical data using Neo4j graph database. It's specifically designed for genomics, pathway analysis, drug discovery, and disease research workflows.

## Features

- 🔗 **Easy Connection Management**: Simple connection setup with environment variable support
- 📊 **Biomedical Data Models**: Pre-built models for genes, pathways, drugs, and diseases
- 🚀 **High-Performance Loading**: Batch processing for large datasets
- 🔍 **Rich Query Builder**: Pre-built queries for common biomedical analysis
- 📈 **Graph Analytics**: Network analysis and pathway exploration
- 🛡️ **Error Handling**: Robust error handling and retry mechanisms
- 📝 **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Installation

### 1. Install Neo4j

First, install Neo4j database:

#### Option A: Docker (Recommended)

```bash
# Pull and run Neo4j
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

#### Option B: Local Installation

- Download from [Neo4j Download Center](https://neo4j.com/download/)
- Follow the installation guide for your operating system

### 2. Install Python Dependencies

```bash
# Install the Neo4j Python driver
pip install neo4j==5.26.0

# Or install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Verify Installation

1. Open Neo4j Browser: http://localhost:7474
2. Login with username: `neo4j`, password: `password` (or your custom password)
3. Run a test query: `RETURN "Hello Neo4j!" as message`

## Quick Start

### 1. Basic Setup

```python
from agentic_ai_wf.neo4j_integration import Neo4jConnection, DataLoader

# Connect to Neo4j
db = Neo4jConnection(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Test connection
if db.test_connection():
    print("✅ Connected to Neo4j!")
else:
    print("❌ Connection failed")
```

### 2. Create Sample Data

```python
from agentic_ai_wf.neo4j_integration import GeneNode, PathwayNode, DrugNode

# Create a gene node
gene = GeneNode(
    symbol="BRCA1",
    ensembl_id="ENSG00000012048",
    description="BRCA1 DNA repair associated",
    log2_fold_change=-2.1,
    p_value=0.001
)

# Save to database
query, params = gene.to_cypher_create()
db.execute_write(query, params)

# Create a pathway
pathway = PathwayNode(
    pathway_id="hsa03440",
    name="Homologous recombination",
    source="KEGG",
    p_value=0.005,
    gene_count=12
)

query, params = pathway.to_cypher_create()
db.execute_write(query, params)
```

### 3. Query Data

```python
from agentic_ai_wf.neo4j_integration import CypherQueryBuilder

# Find genes by disease
query, params = CypherQueryBuilder.find_genes_by_disease("Breast Cancer")
results = db.execute_query(query, params)

for result in results:
    print(f"Gene: {result['gene_symbol']}, FC: {result['log2_fold_change']}")

# Find drugs targeting a specific gene
query, params = CypherQueryBuilder.find_drugs_by_gene("BRCA1")
results = db.execute_query(query, params)

for result in results:
    print(f"Drug: {result['name']}, FDA Approved: {result['fda_approved']}")
```

### 4. Load Data from CSV

```python
from agentic_ai_wf.neo4j_integration import DataLoader

# Initialize data loader
loader = DataLoader(db)

# Load genes from CSV file
# CSV should have columns: symbol, ensembl_id, log2_fold_change, p_value, etc.
gene_stats = loader.load_genes_from_csv("path/to/genes.csv")
print(f"Loaded {gene_stats['created']} genes")

# Load pathways from CSV
pathway_stats = loader.load_pathways_from_csv("path/to/pathways.csv")
print(f"Loaded {pathway_stats['created']} pathways")
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Connection settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Performance settings
NEO4J_BATCH_SIZE=1000
NEO4J_TIMEOUT=30
NEO4J_CREATE_INDEXES=true
```

### Using Configuration

```python
from agentic_ai_wf.neo4j_integration.config import ENV_CONFIG, Neo4jConfig

# Use environment-based configuration
db = Neo4jConnection(
    uri=ENV_CONFIG.uri,
    username=ENV_CONFIG.username,
    password=ENV_CONFIG.password
)

# Or create custom configuration
config = Neo4jConfig(
    uri="bolt://remote-server:7687",
    username="myuser",
    password="mypassword",
    batch_size=2000
)
```

## Data Models

### Gene Node

```python
gene = GeneNode(
    symbol="TP53",                    # Required
    ensembl_id="ENSG00000141510",     # Optional
    entrez_id="7157",                 # Optional
    description="tumor protein p53",  # Optional
    chromosome="17",                  # Optional
    log2_fold_change=2.5,             # Optional (expression data)
    p_value=0.001,                    # Optional (expression data)
    adjusted_p_value=0.01             # Optional (expression data)
)
```

### Pathway Node

```python
pathway = PathwayNode(
    pathway_id="hsa05200",            # Required
    name="Pathways in cancer",        # Required
    source="KEGG",                    # Required
    description="Cancer pathways",    # Optional
    category="Disease",               # Optional
    p_value=0.001,                    # Optional (enrichment data)
    enrichment_score=3.5,             # Optional
    gene_count=25                     # Optional
)
```

### Drug Node

```python
drug = DrugNode(
    name="Doxorubicin",               # Required
    drugbank_id="DB00997",            # Optional
    mechanism_of_action="DNA intercalation", # Optional
    drug_class="Anthracycline",       # Optional
    fda_approved=True,                # Optional
    clinical_trial_phase="Approved",  # Optional
    target_genes=["TOP2A", "TOP2B"]   # Optional
)
```

### Disease Node

```python
disease = DiseaseNode(
    name="Breast Cancer",             # Required
    disease_id="DOID:1612",           # Optional
    description="Malignant breast neoplasm", # Optional
    icd10_code="C50",                 # Optional
    mesh_id="D001943"                 # Optional
)
```

## Query Examples

### Basic Queries

```python
from agentic_ai_wf.neo4j_integration import CypherQueryBuilder

# 1. Find genes associated with a disease
query, params = CypherQueryBuilder.find_genes_by_disease("Alzheimer's Disease")
results = db.execute_query(query, params)

# 2. Find pathways containing a gene
query, params = CypherQueryBuilder.find_pathways_by_gene("APOE")
results = db.execute_query(query, params)

# 3. Find drugs targeting a gene
query, params = CypherQueryBuilder.find_drugs_by_gene("EGFR")
results = db.execute_query(query, params)

# 4. Find enriched pathways
query, params = CypherQueryBuilder.find_enriched_pathways(min_p_value=0.05)
results = db.execute_query(query, params)
```

### Advanced Queries

```python
# 1. Find common pathways for multiple genes
genes = ["BRCA1", "BRCA2", "TP53"]
query, params = CypherQueryBuilder.find_common_pathways_for_genes(genes, min_genes=2)
results = db.execute_query(query, params)

# 2. Find potential drug targets for a disease
query, params = CypherQueryBuilder.find_potential_drug_targets(
    disease_name="Cancer",
    min_fold_change=2.0,
    max_p_value=0.01
)
results = db.execute_query(query, params)

# 3. Find hub genes (highly connected)
query, params = CypherQueryBuilder.find_hub_genes(min_degree=10)
results = db.execute_query(query, params)

# 4. Analyze pathway overlap
pathway_ids = ["hsa05200", "hsa04110", "hsa04151"]
query, params = CypherQueryBuilder.analyze_pathway_overlap(pathway_ids)
results = db.execute_query(query, params)
```

### Custom Queries

```python
# Write custom Cypher queries
custom_query = """
MATCH (g:Gene)-[:BELONGS_TO]->(p:Pathway)<-[:BELONGS_TO]-(g2:Gene)
WHERE g.symbol = $gene_symbol AND g <> g2
RETURN g2.symbol as related_gene, p.name as shared_pathway,
       g2.log2_fold_change as fc
ORDER BY abs(g2.log2_fold_change) DESC
LIMIT 20
"""

query, params = CypherQueryBuilder.custom_query(
    custom_query,
    gene_symbol="BRCA1"
)
results = db.execute_query(query, params)
```

## Integration with Agentic AI Workflow

### Loading Workflow Results

```python
from agentic_ai_wf.neo4j_integration import DataLoader

# Load results from your agentic AI pipeline
loader = DataLoader(db)

results = loader.load_from_workflow_results(
    genes_csv="path/to/deg_results.csv",
    pathways_csv="path/to/pathway_enrichment.csv",
    drugs_csv="path/to/drug_discovery.csv",
    disease_name="Alzheimer's Disease"
)

print(f"Loaded: {results}")
```

### Expected CSV Formats

#### Genes CSV (`deg_results.csv`)

```csv
symbol,ensembl_id,description,log2_fold_change,p_value,adjusted_p_value
APOE,ENSG00000130203,Apolipoprotein E,2.1,0.001,0.01
APP,ENSG00000142192,Amyloid precursor protein,1.8,0.002,0.015
PSEN1,ENSG00000080815,Presenilin 1,1.5,0.005,0.03
```

#### Pathways CSV (`pathway_enrichment.csv`)

```csv
pathway_id,name,source,p_value,enrichment_score,gene_count
hsa05010,Alzheimer disease,KEGG,0.001,3.5,15
GO:0042775,mitochondrial ATP synthesis,GO,0.003,2.8,12
R-HSA-3000157,Laminin interactions,Reactome,0.008,2.1,8
```

#### Drugs CSV (`drug_discovery.csv`)

```csv
name,drugbank_id,mechanism_of_action,drug_class,fda_approved,target_genes
Donepezil,DB00843,Acetylcholinesterase inhibitor,Cholinesterase Inhibitor,true,"ACHE,BCHE"
Memantine,DB01043,NMDA receptor antagonist,NMDA Antagonist,true,"GRIN1,GRIN2A"
Aducanumab,ADU001,Anti-amyloid beta antibody,Monoclonal Antibody,true,APP
```

## Testing

Run the comprehensive test suite:

```bash
# Navigate to the neo4j_integration directory
cd agentic_ai_wf/neo4j_integration

# Run the test script
python test_neo4j.py
```

The test script will:

1. ✅ Test database connection
2. 🗃️ Create sample biomedical data
3. 🔍 Run basic and advanced queries
4. 📄 Test CSV data loading
5. ⚙️ Test custom query functionality

## Performance Tips

### 1. Indexing

```python
# Indexes are automatically created, but you can add custom ones
custom_indexes = [
    "CREATE INDEX protein_uniprot IF NOT EXISTS FOR (p:Protein) ON (p.uniprot_id)",
    "CREATE INDEX compound_smiles IF NOT EXISTS FOR (c:Compound) ON (c.smiles)"
]

for index_query in custom_indexes:
    db.execute_write(index_query)
```

### 2. Batch Loading

```python
# Use batch processing for large datasets
loader = DataLoader(db)
results = loader.load_genes_from_csv("large_dataset.csv", batch_size=5000)
```

### 3. Transaction Management

```python
# Use transactions for bulk operations
with db.driver.session() as session:
    with session.begin_transaction() as tx:
        for gene in gene_list:
            query, params = gene.to_cypher_create()
            tx.run(query, params)
```

## Troubleshooting

### Common Issues

1. **Connection Failed**

   ```
   ❌ Neo4j service unavailable
   ```

   - Check if Neo4j is running: `docker ps` or check service status
   - Verify connection details (URI, username, password)

2. **Authentication Error**

   ```
   ❌ Neo4j authentication failed
   ```

   - Check username/password
   - Reset password if needed: `ALTER CURRENT USER SET PASSWORD FROM 'old' TO 'new'`

3. **Import Errors**

   ```
   ❌ Import "neo4j" could not be resolved
   ```

   - Install Neo4j driver: `pip install neo4j==5.26.0`

4. **Memory Issues**
   ```
   ❌ Java heap space error
   ```
   - Increase Neo4j memory settings in `neo4j.conf`
   - Use smaller batch sizes for data loading

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show all Cypher queries and parameters
db = Neo4jConnection(uri="bolt://localhost:7687", username="neo4j", password="password")
```

## Advanced Usage

### Custom Relationship Types

```python
from agentic_ai_wf.neo4j_integration import create_relationship

# Create custom relationships
query, params = create_relationship(
    from_node_label="Gene",
    from_property="symbol",
    from_value="BRCA1",
    to_node_label="Gene",
    to_property="symbol",
    to_value="BRCA2",
    relationship_type="INTERACTS_WITH",
    properties={"interaction_type": "protein-protein", "confidence": 0.95}
)

db.execute_write(query, params)
```

### Graph Analytics

```python
# Find shortest paths
query = """
MATCH path = shortestPath(
    (g1:Gene {symbol: $gene1})-[*1..4]-(g2:Gene {symbol: $gene2})
)
RETURN path, length(path) as path_length
"""

results = db.execute_query(query, {"gene1": "BRCA1", "gene2": "TP53"})
```

### Export Data

```python
# Export results to CSV
import pandas as pd

query = "MATCH (g:Gene) RETURN g.symbol, g.log2_fold_change, g.p_value"
results = db.execute_query(query)

df = pd.DataFrame(results)
df.to_csv("exported_genes.csv", index=False)
```

## API Reference

### Classes

- `Neo4jConnection`: Database connection management
- `DataLoader`: Bulk data loading utilities
- `CypherQueryBuilder`: Pre-built query templates
- `GeneNode`, `PathwayNode`, `DrugNode`, `DiseaseNode`: Data models

### Functions

- `create_relationship()`: Create relationships between nodes
- `verify_fda_approval()`: FDA drug verification (if available)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Add tests for new functionality
4. Submit a pull request

## License

This module is part of the Agentic AI Workflow project.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the test script for examples
3. Create an issue in the project repository
