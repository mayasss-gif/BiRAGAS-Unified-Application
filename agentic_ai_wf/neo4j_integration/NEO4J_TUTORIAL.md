# 🚀 Neo4j Zero to Hero Tutorial

## Table of Contents

1. [What is Neo4j?](#what-is-neo4j)
2. [Graph Database Fundamentals](#graph-database-fundamentals)
3. [Browser Interface](#browser-interface)
4. [Cypher Query Language Basics](#cypher-query-language-basics)
5. [Hands-on Examples](#hands-on-examples)
6. [Advanced Concepts](#advanced-concepts)
7. [Real-world Use Cases](#real-world-use-cases)
8. [Best Practices](#best-practices)

---

## 🎯 What is Neo4j?

Neo4j is a **graph database** - a database designed to treat relationships between data as equally important as the data itself.

### Traditional vs Graph Database

**Traditional (Relational):**

```
Users Table: [id, name, email]
Posts Table: [id, user_id, content, likes]
Follows Table: [follower_id, following_id]
```

**Graph Database:**

```
(User)-[:POSTED]->(Post)
(User)-[:LIKES]->(Post)
(User)-[:FOLLOWS]->(User)
```

### Why Graph Databases?

- **Natural relationships:** Model real-world connections
- **Performance:** Fast relationship traversals
- **Flexibility:** Easy schema evolution
- **Intuitive queries:** SQL vs Cypher comparison

---

## 🔗 Graph Database Fundamentals

### Core Components

1. **Nodes** (Vertices)

   - Entities in your data
   - Can have labels: `(p:Person)`, `(m:Movie)`
   - Contain properties: `{name: "John", age: 30}`

2. **Relationships** (Edges)

   - Connections between nodes
   - Have direction: `(a)-[:KNOWS]->(b)`
   - Can have properties: `{since: 2020, strength: 0.8}`

3. **Properties**

   - Key-value pairs on nodes/relationships
   - Support various data types

4. **Labels**
   - Categories for nodes
   - Used for indexing and queries

### Graph Model Example

```
(Alice:Person {name: "Alice", age: 30})
(Bob:Person {name: "Bob", age: 25})
(Company:Organization {name: "TechCorp"})

(Alice)-[:WORKS_FOR {since: 2020}]->(Company)
(Alice)-[:KNOWS {since: 2018}]->(Bob)
```

---

## 🌐 Browser Interface

### Access Your Neo4j Browser

1. **URL:** http://localhost:7474
2. **Credentials:** neo4j / password
3. **Interface Components:**
   - Query editor (top)
   - Results visualization (bottom)
   - Database info (sidebar)

### Browser Features

- **Visual query results:** Nodes and relationships display
- **Query history:** Previous queries saved
- **Database statistics:** Node/relationship counts
- **Schema visualization:** Data model overview
- **Query guides:** Built-in tutorials

---

## 📝 Cypher Query Language Basics

Cypher is Neo4j's query language - designed to be intuitive and readable.

### 1. Basic Syntax

#### CREATE Nodes

```cypher
// Create a person
CREATE (p:Person {name: "Alice", age: 30})

// Create multiple nodes
CREATE (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
```

#### CREATE Relationships

```cypher
// Create nodes with relationship
CREATE (a:Person {name: "Alice"})-[:KNOWS]->(b:Person {name: "Bob"})

// Create relationship between existing nodes
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:FRIENDS_WITH {since: 2020}]->(b)
```

#### MATCH (Read/Query)

```cypher
// Find all persons
MATCH (p:Person) RETURN p

// Find specific person
MATCH (p:Person {name: "Alice"}) RETURN p

// Find relationships
MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b
```

#### WHERE Clauses

```cypher
// Filter by property
MATCH (p:Person) WHERE p.age > 25 RETURN p

// Multiple conditions
MATCH (p:Person)
WHERE p.age > 20 AND p.name STARTS WITH "A"
RETURN p
```

#### UPDATE Properties

```cypher
// Set property
MATCH (p:Person {name: "Alice"})
SET p.email = "alice@example.com"

// Update multiple properties
MATCH (p:Person {name: "Alice"})
SET p += {age: 31, city: "New York"}
```

#### DELETE

```cypher
// Delete relationships
MATCH (a:Person)-[r:KNOWS]->(b:Person) DELETE r

// Delete nodes (must delete relationships first)
MATCH (p:Person {name: "Alice"}) DELETE p

// Delete everything (careful!)
MATCH (n) DETACH DELETE n
```

### 2. Essential RETURN Patterns

```cypher
// Return nodes
MATCH (p:Person) RETURN p

// Return properties
MATCH (p:Person) RETURN p.name, p.age

// Return with aliases
MATCH (p:Person) RETURN p.name AS name, p.age AS age

// Count
MATCH (p:Person) RETURN COUNT(p)

// Distinct values
MATCH (p:Person) RETURN DISTINCT p.city
```

### 3. Relationship Patterns

```cypher
// Any direction
MATCH (a:Person)-[:KNOWS]-(b:Person) RETURN a, b

// Specific direction
MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b

// Multiple relationships
MATCH (a:Person)-[:KNOWS]->(b:Person)-[:WORKS_FOR]->(c:Company)
RETURN a, b, c

// Variable length paths
MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN a, b

// Any relationship type
MATCH (a:Person)-[r]->(b) RETURN a, r, b
```

---

## 🛠️ Hands-on Examples

Let's build a social network step by step!

### Example 1: Simple Social Network

#### Step 1: Create People

```cypher
CREATE (alice:Person {name: "Alice", age: 30, city: "New York"})
CREATE (bob:Person {name: "Bob", age: 25, city: "San Francisco"})
CREATE (charlie:Person {name: "Charlie", age: 35, city: "London"})
CREATE (diana:Person {name: "Diana", age: 28, city: "New York"})
```

#### Step 2: Create Friendships

```cypher
MATCH (alice:Person {name: "Alice"}), (bob:Person {name: "Bob"})
CREATE (alice)-[:FRIENDS_WITH {since: 2018}]->(bob)

MATCH (alice:Person {name: "Alice"}), (diana:Person {name: "Diana"})
CREATE (alice)-[:FRIENDS_WITH {since: 2020}]->(diana)

MATCH (bob:Person {name: "Bob"}), (charlie:Person {name: "Charlie"})
CREATE (bob)-[:FRIENDS_WITH {since: 2019}]->(charlie)
```

#### Step 3: Query the Network

```cypher
// Find all of Alice's friends
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)
RETURN alice.name, friend.name

// Find mutual friends
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(mutual)<-[:FRIENDS_WITH]-(other:Person)
WHERE other.name <> "Alice"
RETURN other.name, mutual.name

// Find friends of friends
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH*2]->(fof:Person)
WHERE fof <> alice
RETURN DISTINCT fof.name
```

### Example 2: Movie Database

#### Create Movies and Actors

```cypher
// Create movies
CREATE (matrix:Movie {title: "The Matrix", year: 1999, rating: 8.7})
CREATE (john_wick:Movie {title: "John Wick", year: 2014, rating: 7.4})

// Create actors
CREATE (keanu:Actor {name: "Keanu Reeves", born: 1964})
CREATE (laurence:Actor {name: "Laurence Fishburne", born: 1961})

// Create relationships
MATCH (keanu:Actor {name: "Keanu Reeves"}), (matrix:Movie {title: "The Matrix"})
CREATE (keanu)-[:ACTED_IN {role: "Neo"}]->(matrix)

MATCH (laurence:Actor {name: "Laurence Fishburne"}), (matrix:Movie {title: "The Matrix"})
CREATE (laurence)-[:ACTED_IN {role: "Morpheus"}]->(matrix)

MATCH (keanu:Actor {name: "Keanu Reeves"}), (john_wick:Movie {title: "John Wick"})
CREATE (keanu)-[:ACTED_IN {role: "John Wick"}]->(john_wick)
```

#### Query Movies

```cypher
// Find all movies by Keanu Reeves
MATCH (keanu:Actor {name: "Keanu Reeves"})-[:ACTED_IN]->(movie:Movie)
RETURN movie.title, movie.year

// Find co-actors
MATCH (keanu:Actor {name: "Keanu Reeves"})-[:ACTED_IN]->(movie)<-[:ACTED_IN]-(coactor:Actor)
WHERE coactor <> keanu
RETURN movie.title, coactor.name

// Find actors who worked together in multiple movies
MATCH (a1:Actor)-[:ACTED_IN]->(movie)<-[:ACTED_IN]-(a2:Actor)
WHERE a1 <> a2
WITH a1, a2, COUNT(movie) as collaborations
WHERE collaborations > 1
RETURN a1.name, a2.name, collaborations
```

### Example 3: Biomedical Network (Building on our integration!)

```cypher
// Create genes
CREATE (brca1:Gene {symbol: "BRCA1", chromosome: "17", function: "DNA repair"})
CREATE (tp53:Gene {symbol: "TP53", chromosome: "17", function: "Tumor suppressor"})

// Create diseases
CREATE (cancer:Disease {name: "Breast Cancer", type: "Cancer"})

// Create drugs
CREATE (tamoxifen:Drug {name: "Tamoxifen", type: "Hormone therapy"})

// Create relationships
MATCH (brca1:Gene {symbol: "BRCA1"}), (cancer:Disease {name: "Breast Cancer"})
CREATE (brca1)-[:ASSOCIATED_WITH {strength: 0.9}]->(cancer)

MATCH (tamoxifen:Drug {name: "Tamoxifen"}), (cancer:Disease {name: "Breast Cancer"})
CREATE (tamoxifen)-[:TREATS {efficacy: 0.7}]->(cancer)
```

---

## 🔥 Advanced Concepts

### 1. Aggregation Functions

```cypher
// Count, sum, average
MATCH (p:Person)
RETURN COUNT(p), AVG(p.age), MIN(p.age), MAX(p.age)

// Group by
MATCH (p:Person)
RETURN p.city, COUNT(p) as population
ORDER BY population DESC

// Collect into lists
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)
RETURN alice.name, COLLECT(friend.name) as friends
```

### 2. Conditional Logic

```cypher
// CASE statements
MATCH (p:Person)
RETURN p.name,
  CASE
    WHEN p.age < 25 THEN "Young"
    WHEN p.age < 35 THEN "Adult"
    ELSE "Senior"
  END as age_group

// EXISTS for optional patterns
MATCH (p:Person)
WHERE EXISTS((p)-[:FRIENDS_WITH]->())
RETURN p.name
```

### 3. Path Finding

```cypher
// Shortest path
MATCH path = shortestPath((alice:Person {name: "Alice"})-[:FRIENDS_WITH*]-(charlie:Person {name: "Charlie"}))
RETURN path

// All shortest paths
MATCH paths = allShortestPaths((alice:Person {name: "Alice"})-[:FRIENDS_WITH*]-(charlie:Person {name: "Charlie"}))
RETURN paths
```

### 4. Working with Lists

```cypher
// List comprehension
MATCH (p:Person)
RETURN [x IN COLLECT(p.age) WHERE x > 25] as adult_ages

// UNWIND (flatten lists)
WITH [1, 2, 3] as numbers
UNWIND numbers as number
RETURN number * 2
```

### 5. Indexes and Constraints

```cypher
// Create index
CREATE INDEX person_name FOR (p:Person) ON (p.name)

// Create unique constraint
CREATE CONSTRAINT person_email FOR (p:Person) REQUIRE p.email IS UNIQUE

// Show indexes
SHOW INDEXES

// Show constraints
SHOW CONSTRAINTS
```

---

## 🌟 Real-world Use Cases

### 1. Recommendation Engine

```cypher
// Find movies liked by people with similar tastes
MATCH (user:User {name: "Alice"})-[:RATED]->(movie:Movie)<-[:RATED]-(similar:User)
MATCH (similar)-[:RATED]->(recommendation:Movie)
WHERE NOT EXISTS((user)-[:RATED]->(recommendation))
RETURN recommendation.title, COUNT(*) as score
ORDER BY score DESC
LIMIT 5
```

### 2. Fraud Detection

```cypher
// Find suspicious patterns: users sharing multiple attributes
MATCH (u1:User)-[:HAS_PHONE]->(phone:Phone)<-[:HAS_PHONE]-(u2:User)
MATCH (u1)-[:HAS_ADDRESS]->(addr:Address)<-[:HAS_ADDRESS]-(u2)
WHERE u1 <> u2
RETURN u1.name, u2.name, "Suspicious: shared phone and address" as reason
```

### 3. Supply Chain Analysis

```cypher
// Find all suppliers for a product (multi-level)
MATCH path = (product:Product {name: "iPhone"})<-[:SUPPLIES*]-(supplier:Supplier)
RETURN supplier.name, LENGTH(path) as supply_level
ORDER BY supply_level
```

### 4. Social Network Analysis

```cypher
// Find influencers (most connected people)
MATCH (p:Person)-[r:FRIENDS_WITH]-()
RETURN p.name, COUNT(r) as connections
ORDER BY connections DESC
LIMIT 10

// Find communities (people who are mutually connected)
MATCH (a:Person)-[:FRIENDS_WITH]-(b:Person)-[:FRIENDS_WITH]-(c:Person)-[:FRIENDS_WITH]-(a)
WHERE a <> b AND b <> c AND c <> a
RETURN a.name, b.name, c.name
```

---

## 💡 Best Practices

### 1. Data Modeling

**DO:**

- Use meaningful labels and relationship types
- Keep node properties simple
- Use relationships to model verbs/actions
- Index frequently queried properties

**DON'T:**

- Store lists as comma-separated strings
- Create deeply nested JSON in properties
- Use relationships as flags (use properties instead)

### 2. Query Optimization

**Performance Tips:**

```cypher
// Use PROFILE to analyze queries
PROFILE MATCH (p:Person {name: "Alice"}) RETURN p

// Use indexes
MATCH (p:Person) WHERE p.name = "Alice" RETURN p  // Better than scanning

// Limit early
MATCH (p:Person) WHERE p.age > 25
RETURN p LIMIT 10

// Use WITH for complex queries
MATCH (p:Person)
WITH p WHERE p.age > 25
MATCH (p)-[:FRIENDS_WITH]->(friend)
RETURN p.name, COUNT(friend)
```

### 3. Schema Design

```cypher
// Good: Specific relationships
(:Person)-[:WORKS_FOR]->(:Company)
(:Person)-[:LIVES_IN]->(:City)

// Bad: Generic relationships
(:Person)-[:RELATED_TO {type: "employment"}]->(:Company)
```

### 4. Data Import Strategies

```cypher
// Use MERGE for upserts
MERGE (p:Person {email: "alice@example.com"})
ON CREATE SET p.name = "Alice", p.created = timestamp()
ON MATCH SET p.lastSeen = timestamp()

// Batch processing with UNWIND
UNWIND [
  {name: "Alice", age: 30},
  {name: "Bob", age: 25}
] as person
MERGE (p:Person {name: person.name})
SET p.age = person.age
```

---

## 🚀 Next Steps

### Practice Exercises

1. **Build a University Database:**

   - Students, Courses, Professors
   - Enrollment, Teaching relationships
   - Query: Find all students taking courses from the same professor

2. **Create a Transportation Network:**

   - Cities, Routes, Airlines
   - Model flight connections
   - Find shortest routes between cities

3. **Design a Knowledge Graph:**
   - Concepts, People, Publications
   - Model who researches what
   - Find research collaborations

### Advanced Topics to Explore

1. **Graph Algorithms:**

   - PageRank, Centrality measures
   - Community detection
   - Shortest path algorithms

2. **APOC Procedures:**

   - Extended functionality
   - Data import/export
   - Advanced graph algorithms

3. **Multi-database Operations:**

   - Fabric (multiple databases)
   - Federation
   - Sharding strategies

4. **Graph Data Science:**
   - Machine learning on graphs
   - Node embeddings
   - Link prediction

### Resources

- **Neo4j Documentation:** https://neo4j.com/docs/
- **Cypher Manual:** https://neo4j.com/docs/cypher-manual/
- **Graph Algorithms:** https://neo4j.com/docs/graph-data-science/
- **APOC Documentation:** https://neo4j.com/docs/apoc/

---

## 🎉 Congratulations!

You've learned Neo4j from zero to hero! You now understand:

- Graph database concepts
- Cypher query language
- Real-world applications
- Best practices
- Performance optimization

**Start practicing with the Neo4j browser at:** http://localhost:7474

Happy graphing! 🚀
