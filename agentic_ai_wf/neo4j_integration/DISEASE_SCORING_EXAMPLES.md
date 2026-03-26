# 🧬 Disease Scoring Query Examples

## Overview

The updated GeneCards importer now properly extracts and stores **GeneScore** and **DisorderScore** from both `MalaCardsDisorders` and `MalaCardsInferredDisorders`. You can search by disease name or aliases to get these scoring metrics.

## Database Structure

### Disease Relationships

- **ASSOCIATED_WITH**: Direct disease associations from `MalaCardsDisorders`
- **INFERRED_ASSOCIATION**: Inferred disease associations from `MalaCardsInferredDisorders`

### Relationship Properties

- `gene_score`: The GeneScore from MalaCards
- `disorder_score`: The DisorderScore from MalaCards
- `is_elite`: Whether the disease is marked as elite
- `disorder_type`: "direct" or "inferred"

### Disease Node Properties

- `name`: Disease name
- `aliases`: JSON array of disease aliases
- `malacards_id`: MalaCards accession ID
- `is_elite`: Elite status
- `is_cancer_census`: Cancer census status

## Query Examples

### 1. Find All Genes for a Specific Disease

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease {name: "Long Qt Syndrome"})
RETURN g.symbol, g.name, r.gene_score, r.disorder_score, r.is_elite, r.disorder_type
ORDER BY r.gene_score DESC
```

### 2. Search by Disease Name Pattern

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
WHERE d.name CONTAINS "Seizures"
RETURN g.symbol, d.name, r.gene_score, r.disorder_score, r.disorder_type
ORDER BY r.gene_score DESC
```

### 3. Search by Disease Aliases

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
WHERE d.aliases CONTAINS "Lupus" OR d.name CONTAINS "Lupus"
RETURN g.symbol, d.name, r.gene_score, r.disorder_score, r.disorder_type
ORDER BY r.gene_score DESC
```

### 4. Find High-Scoring Disease Associations

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
WHERE r.gene_score > 10
RETURN g.symbol, d.name, r.gene_score, r.disorder_score, r.is_elite
ORDER BY r.gene_score DESC
```

### 5. Search for Elite Disease Associations

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(d:Disease)
WHERE r.is_elite = true
RETURN g.symbol, d.name, r.gene_score, r.disorder_score
ORDER BY r.gene_score DESC
```

### 6. Find Diseases by Multiple Aliases

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
WHERE d.aliases CONTAINS "BFIS1"
   OR d.aliases CONTAINS "Benign Familial Infantile Seizures"
   OR d.name CONTAINS "Infantile"
RETURN g.symbol, d.name, r.gene_score, r.disorder_score
ORDER BY r.gene_score DESC
```

### 7. Compare Direct vs Inferred Disease Associations

```cypher
MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
WHERE d.name CONTAINS "Syndrome"
RETURN g.symbol, d.name, r.gene_score, r.disorder_score, r.disorder_type,
       CASE r.disorder_type
         WHEN "direct" THEN "MalaCardsDisorders"
         WHEN "inferred" THEN "MalaCardsInferredDisorders"
       END as source_section
ORDER BY r.gene_score DESC
```

### 8. Get Disease Statistics by Gene

```cypher
MATCH (g:Gene {symbol: "ABCF1"})-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
RETURN g.symbol,
       count(d) as total_diseases,
       count(CASE WHEN r.disorder_type = "direct" THEN 1 END) as direct_diseases,
       count(CASE WHEN r.disorder_type = "inferred" THEN 1 END) as inferred_diseases,
       avg(r.gene_score) as avg_gene_score,
       max(r.gene_score) as max_gene_score
```

## Python Integration Examples

### Using the Neo4jConnection Class

```python
from neo4j_integration.connection import Neo4jConnection

db = Neo4jConnection()

# Search for disease by name
def find_genes_for_disease(disease_name):
    query = """
    MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
    WHERE d.name CONTAINS $disease_name OR d.aliases CONTAINS $disease_name
    RETURN g.symbol, g.name, d.name, r.gene_score, r.disorder_score, r.is_elite
    ORDER BY r.gene_score DESC
    """
    return db.execute_query(query, {"disease_name": disease_name})

# Get top scoring disease associations
def get_top_disease_associations(limit=10):
    query = """
    MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
    RETURN g.symbol, d.name, r.gene_score, r.disorder_score, r.disorder_type
    ORDER BY r.gene_score DESC
    LIMIT $limit
    """
    return db.execute_query(query, {"limit": limit})

# Search by disease aliases
def search_disease_aliases(alias_pattern):
    query = """
    MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
    WHERE d.aliases CONTAINS $pattern
    RETURN g.symbol, d.name, r.gene_score, r.disorder_score
    ORDER BY r.gene_score DESC
    """
    return db.execute_query(query, {"pattern": alias_pattern})
```

## Data Source Mapping

| JSON Field                           | Database Storage              | Description                  |
| ------------------------------------ | ----------------------------- | ---------------------------- |
| `MalaCardsDisorders[].GeneScore`     | `relationship.gene_score`     | Gene association score       |
| `MalaCardsDisorders[].DisorderScore` | `relationship.disorder_score` | Disease score                |
| `MalaCardsDisorders[].IsElite`       | `relationship.is_elite`       | Elite status                 |
| `MalaCardsDisorders[].Name`          | `disease.name`                | Disease name                 |
| `MalaCardsDisorders[].Aliases`       | `disease.aliases`             | Disease aliases (JSON array) |
| `MalaCardsInferredDisorders[].*`     | Same fields                   | Inferred associations        |

## Performance Tips

1. **Use indexes**: Disease name and aliases are indexed for fast searching
2. **Filter early**: Use WHERE clauses to limit results before processing
3. **Order by scores**: Most queries should order by gene_score or disorder_score
4. **Limit results**: Use LIMIT for large datasets
5. **Use CONTAINS**: For partial text matching in names and aliases

Your GeneCards data is now fully searchable by disease names and aliases with proper scoring! 🎯
