#!/usr/bin/env python3
"""
Neo4j Hands-on Practice Script

This script provides interactive examples to practice Neo4j concepts
as you go through the tutorial.

Usage: python hands_on_practice.py
"""

import sys
import time
from agentic_ai_wf.neo4j_integration.connection import Neo4jConnection


class Neo4jPractice:
    """Interactive Neo4j practice session."""

    def __init__(self):
        self.db = None
        self.connect_to_neo4j()

    def connect_to_neo4j(self):
        """Connect to Neo4j database."""
        try:
            self.db = Neo4jConnection()
            print("✅ Connected to Neo4j successfully!")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            print("Make sure Neo4j is running at http://localhost:7474")
            sys.exit(1)

    def wait_for_user(self, message="Press Enter to continue..."):
        """Wait for user input."""
        input(f"\n💡 {message}")

    def execute_and_show(self, query, description, show_results=True):
        """Execute a query and display results."""
        print(f"\n🔍 {description}")
        print(f"Query: {query}")
        print("-" * 60)

        try:
            assert self.db is not None, "Database connection not established"
            results = self.db.execute_query(query)
            if show_results and results:
                # Show max 5 results
                for i, result in enumerate(results[:5], 1):
                    print(f"Result {i}: {result}")
                if len(results) > 5:
                    print(f"... and {len(results) - 5} more results")
            else:
                print(
                    f"✅ Query executed successfully! ({len(results)} results)")
        except Exception as e:
            print(f"❌ Error: {e}")

    def lesson_1_basic_nodes(self):
        """Lesson 1: Creating and querying basic nodes."""
        print("\n" + "="*60)
        print("📚 LESSON 1: Basic Nodes and Properties")
        print("="*60)

        # Clear database
        self.wait_for_user("Let's start fresh. Clear the database?")
        self.execute_and_show("MATCH (n) DETACH DELETE n", "Clearing database")

        # Create simple nodes
        self.wait_for_user("Create some people nodes")
        queries = [
            ('CREATE (alice:Person {name: "Alice", age: 30, city: "New York"})',
             "Creating Alice"),
            ('CREATE (bob:Person {name: "Bob", age: 25, city: "San Francisco"})',
             "Creating Bob"),
            ('CREATE (charlie:Person {name: "Charlie", age: 35, city: "London"})',
             "Creating Charlie")
        ]

        for query, desc in queries:
            self.execute_and_show(query, desc, False)

        # Query nodes
        self.wait_for_user("Query all people")
        self.execute_and_show("MATCH (p:Person) RETURN p.name, p.age, p.city",
                              "Finding all people")

        # Filter queries
        self.wait_for_user("Filter by age")
        self.execute_and_show("MATCH (p:Person) WHERE p.age > 25 RETURN p.name, p.age",
                              "Finding people older than 25")

        print("\n🎉 Great! You've learned to create and query nodes!")

    def lesson_2_relationships(self):
        """Lesson 2: Creating and querying relationships."""
        print("\n" + "="*60)
        print("📚 LESSON 2: Relationships")
        print("="*60)

        # Create relationships
        self.wait_for_user("Create friendship relationships")
        queries = [
            ('MATCH (alice:Person {name: "Alice"}), (bob:Person {name: "Bob"}) '
             'CREATE (alice)-[:FRIENDS_WITH {since: 2018}]->(bob)',
             "Alice is friends with Bob"),
            ('MATCH (alice:Person {name: "Alice"}), (charlie:Person {name: "Charlie"}) '
             'CREATE (alice)-[:FRIENDS_WITH {since: 2020}]->(charlie)',
             "Alice is friends with Charlie"),
            ('MATCH (bob:Person {name: "Bob"}), (charlie:Person {name: "Charlie"}) '
             'CREATE (bob)-[:FRIENDS_WITH {since: 2019}]->(charlie)',
             "Bob is friends with Charlie")
        ]

        for query, desc in queries:
            self.execute_and_show(query, desc, False)

        # Query relationships
        self.wait_for_user("Find Alice's friends")
        self.execute_and_show(
            'MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend) '
            'RETURN alice.name, friend.name',
            "Finding Alice's friends"
        )

        # Bidirectional queries
        self.wait_for_user("Find all friendships (any direction)")
        self.execute_and_show(
            'MATCH (a:Person)-[r:FRIENDS_WITH]-(b:Person) '
            'RETURN a.name, b.name, r.since',
            "Finding all friendships"
        )

        print("\n🎉 Excellent! You understand relationships now!")

    def lesson_3_movie_database(self):
        """Lesson 3: Movie database example."""
        print("\n" + "="*60)
        print("📚 LESSON 3: Movie Database")
        print("="*60)

        # Create movies and actors
        self.wait_for_user("Create a movie database")
        queries = [
            ('CREATE (matrix:Movie {title: "The Matrix", year: 1999, rating: 8.7})',
             "Creating The Matrix"),
            ('CREATE (johnwick:Movie {title: "John Wick", year: 2014, rating: 7.4})',
             "Creating John Wick"),
            ('CREATE (keanu:Actor {name: "Keanu Reeves", born: 1964})',
             "Creating Keanu Reeves"),
            ('CREATE (laurence:Actor {name: "Laurence Fishburne", born: 1961})',
             "Creating Laurence Fishburne")
        ]

        for query, desc in queries:
            self.execute_and_show(query, desc, False)

        # Create acting relationships
        self.wait_for_user("Connect actors to movies")
        queries = [
            ('MATCH (keanu:Actor {name: "Keanu Reeves"}), (matrix:Movie {title: "The Matrix"}) '
             'CREATE (keanu)-[:ACTED_IN {role: "Neo"}]->(matrix)',
             "Keanu acted in The Matrix"),
            ('MATCH (laurence:Actor {name: "Laurence Fishburne"}), (matrix:Movie {title: "The Matrix"}) '
             'CREATE (laurence)-[:ACTED_IN {role: "Morpheus"}]->(matrix)',
             "Laurence acted in The Matrix"),
            ('MATCH (keanu:Actor {name: "Keanu Reeves"}), (johnwick:Movie {title: "John Wick"}) '
             'CREATE (keanu)-[:ACTED_IN {role: "John Wick"}]->(johnwick)',
             "Keanu acted in John Wick")
        ]

        for query, desc in queries:
            self.execute_and_show(query, desc, False)

        # Query the movie database
        self.wait_for_user("Find movies by Keanu Reeves")
        self.execute_and_show(
            'MATCH (keanu:Actor {name: "Keanu Reeves"})-[:ACTED_IN]->(movie:Movie) '
            'RETURN movie.title, movie.year',
            "Movies starring Keanu Reeves"
        )

        self.wait_for_user("Find co-actors")
        self.execute_and_show(
            'MATCH (keanu:Actor {name: "Keanu Reeves"})-[:ACTED_IN]->(movie)<-[:ACTED_IN]-(coactor:Actor) '
            'WHERE coactor <> keanu '
            'RETURN movie.title, coactor.name',
            "Keanu's co-actors"
        )

        print("\n🎉 Amazing! You've built a movie database!")

    def lesson_4_advanced_queries(self):
        """Lesson 4: Advanced query patterns."""
        print("\n" + "="*60)
        print("📚 LESSON 4: Advanced Queries")
        print("="*60)

        # Aggregation
        self.wait_for_user("Count nodes by type")
        self.execute_and_show(
            'MATCH (n) RETURN labels(n)[0] as type, COUNT(n) as count ORDER BY count DESC',
            "Counting nodes by type"
        )

        # Path finding
        self.wait_for_user("Find paths between Alice and Charlie")
        self.execute_and_show(
            'MATCH path = (alice:Person {name: "Alice"})-[:FRIENDS_WITH*1..3]-(charlie:Person {name: "Charlie"}) '
            'RETURN length(path) as path_length, nodes(path) as path',
            "Paths between Alice and Charlie"
        )

        # Complex patterns
        self.wait_for_user("Find actors and their total movies")
        self.execute_and_show(
            'MATCH (actor:Actor)-[:ACTED_IN]->(movie:Movie) '
            'RETURN actor.name, COUNT(movie) as movie_count, COLLECT(movie.title) as movies '
            'ORDER BY movie_count DESC',
            "Actors and their movie counts"
        )

        print("\n🎉 Outstanding! You're mastering advanced queries!")

    def lesson_5_data_analysis(self):
        """Lesson 5: Data analysis patterns."""
        print("\n" + "="*60)
        print("📚 LESSON 5: Data Analysis")
        print("="*60)

        # Network analysis
        self.wait_for_user("Find most connected people")
        self.execute_and_show(
            'MATCH (p:Person)-[r:FRIENDS_WITH]-() '
            'RETURN p.name, COUNT(r) as connections '
            'ORDER BY connections DESC',
            "Most connected people"
        )

        # Conditional logic
        self.wait_for_user("Categorize people by age")
        self.execute_and_show(
            'MATCH (p:Person) '
            'RETURN p.name, p.age, '
            'CASE '
            '  WHEN p.age < 25 THEN "Young" '
            '  WHEN p.age < 35 THEN "Adult" '
            '  ELSE "Senior" '
            'END as age_group',
            "Age group categorization"
        )

        # Geography analysis
        self.wait_for_user("Group people by city")
        self.execute_and_show(
            'MATCH (p:Person) '
            'RETURN p.city, COUNT(p) as population, COLLECT(p.name) as residents '
            'ORDER BY population DESC',
            "People grouped by city"
        )

        print("\n🎉 Fantastic! You're doing real data analysis!")

    def lesson_6_biomedical_example(self):
        """Lesson 6: Biomedical graph example."""
        print("\n" + "="*60)
        print("📚 LESSON 6: Biomedical Knowledge Graph")
        print("="*60)

        # Create biomedical entities
        self.wait_for_user("Create genes, diseases, and drugs")
        queries = [
            ('CREATE (brca1:Gene {symbol: "BRCA1", chromosome: "17", function: "DNA repair"})',
             "Creating BRCA1 gene"),
            ('CREATE (tp53:Gene {symbol: "TP53", chromosome: "17", function: "Tumor suppressor"})',
             "Creating TP53 gene"),
            ('CREATE (cancer:Disease {name: "Breast Cancer", type: "Cancer"})',
             "Creating Breast Cancer disease"),
            ('CREATE (tamoxifen:Drug {name: "Tamoxifen", type: "Hormone therapy"})',
             "Creating Tamoxifen drug")
        ]

        for query, desc in queries:
            self.execute_and_show(query, desc, False)

        # Create biomedical relationships
        self.wait_for_user("Connect biomedical entities")
        queries = [
            ('MATCH (brca1:Gene {symbol: "BRCA1"}), (cancer:Disease {name: "Breast Cancer"}) '
             'CREATE (brca1)-[:ASSOCIATED_WITH {strength: 0.9}]->(cancer)',
             "BRCA1 associated with Breast Cancer"),
            ('MATCH (tamoxifen:Drug {name: "Tamoxifen"}), (cancer:Disease {name: "Breast Cancer"}) '
             'CREATE (tamoxifen)-[:TREATS {efficacy: 0.7}]->(cancer)',
             "Tamoxifen treats Breast Cancer"),
            ('MATCH (tp53:Gene {symbol: "TP53"}), (cancer:Disease {name: "Breast Cancer"}) '
             'CREATE (tp53)-[:ASSOCIATED_WITH {strength: 0.8}]->(cancer)',
             "TP53 associated with Breast Cancer")
        ]

        for query, desc in queries:
            self.execute_and_show(query, desc, False)

        # Query biomedical network
        self.wait_for_user("Find genes associated with Breast Cancer")
        self.execute_and_show(
            'MATCH (gene:Gene)-[r:ASSOCIATED_WITH]->(disease:Disease {name: "Breast Cancer"}) '
            'RETURN gene.symbol, gene.function, r.strength '
            'ORDER BY r.strength DESC',
            "Genes associated with Breast Cancer"
        )

        self.wait_for_user("Find treatment options")
        self.execute_and_show(
            'MATCH (drug:Drug)-[r:TREATS]->(disease:Disease {name: "Breast Cancer"}) '
            'RETURN drug.name, drug.type, r.efficacy',
            "Drugs treating Breast Cancer"
        )

        print("\n🎉 Incredible! You've built a biomedical knowledge graph!")

    def lesson_7_final_challenges(self):
        """Lesson 7: Final challenges."""
        print("\n" + "="*60)
        print("📚 LESSON 7: Final Challenges")
        print("="*60)

        # Challenge 1: Complex query
        self.wait_for_user("Challenge 1: Find indirect connections")
        self.execute_and_show(
            'MATCH path = (person:Person)-[:FRIENDS_WITH*2..3]-(other:Person) '
            'WHERE person.name = "Alice" AND other.name <> "Alice" '
            'RETURN other.name, length(path) as degrees_of_separation '
            'ORDER BY degrees_of_separation',
            "People connected to Alice through 2-3 degrees"
        )

        # Challenge 2: Cross-domain query
        self.wait_for_user("Challenge 2: Cross-domain analysis")
        self.execute_and_show(
            'MATCH (n) '
            'RETURN labels(n)[0] as node_type, COUNT(n) as count, '
            'COLLECT(DISTINCT CASE '
            '  WHEN "Person" IN labels(n) THEN n.name '
            '  WHEN "Movie" IN labels(n) THEN n.title '
            '  WHEN "Actor" IN labels(n) THEN n.name '
            '  WHEN "Gene" IN labels(n) THEN n.symbol '
            '  WHEN "Disease" IN labels(n) THEN n.name '
            '  WHEN "Drug" IN labels(n) THEN n.name '
            'END)[0..3] as examples '
            'ORDER BY count DESC',
            "Database summary with examples"
        )

        print("\n🎉🎉🎉 CONGRATULATIONS! You've completed the Neo4j hands-on tutorial!")
        print("\n🚀 Next steps:")
        print("1. Open Neo4j Browser: http://localhost:7474")
        print("2. Try the queries from the tutorial")
        print("3. Experiment with your own data")
        print("4. Read the full tutorial: NEO4J_TUTORIAL.md")

    def run_full_tutorial(self):
        """Run the complete hands-on tutorial."""
        print("🚀 Welcome to Neo4j Hands-on Practice!")
        print("This interactive session will teach you Neo4j step by step.")
        print("\nYou can also access Neo4j Browser at: http://localhost:7474")
        print("Login with: neo4j / password")

        self.wait_for_user("Ready to start? Press Enter!")

        try:
            self.lesson_1_basic_nodes()
            self.lesson_2_relationships()
            self.lesson_3_movie_database()
            self.lesson_4_advanced_queries()
            self.lesson_5_data_analysis()
            self.lesson_6_biomedical_example()
            self.lesson_7_final_challenges()

        except KeyboardInterrupt:
            print("\n\n👋 Tutorial interrupted. You can restart anytime!")
        except Exception as e:
            print(f"\n❌ Error during tutorial: {e}")
        finally:
            if self.db:
                self.db.close()

    def quick_demo(self):
        """Run a quick 5-minute demo."""
        print("⚡ Quick Neo4j Demo (5 minutes)")

        # Clear and create simple data
        self.execute_and_show("MATCH (n) DETACH DELETE n",
                              "Clearing database", False)
        self.execute_and_show(
            'CREATE (alice:Person {name: "Alice"})-[:KNOWS]->(bob:Person {name: "Bob"})-[:KNOWS]->(charlie:Person {name: "Charlie"})',
            "Creating a simple network", False
        )

        # Show what we created
        self.execute_and_show("MATCH (n) RETURN n", "What's in our database?")

        # Show relationships
        self.execute_and_show(
            "MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name",
            "Show relationships"
        )

        print("\n🎉 Quick demo complete! Try the full tutorial for more!")


def main():
    """Main function."""
    print("Neo4j Learning Options:")
    print("1. Full hands-on tutorial (30 minutes)")
    print("2. Quick demo (5 minutes)")
    print("3. Just connect and explore")

    choice = input("\nChoice (1-3): ").strip()

    practice = Neo4jPractice()

    if choice == "1":
        practice.run_full_tutorial()
    elif choice == "2":
        practice.quick_demo()
    else:
        print("✅ Connected! You can now:")
        print("- Open Neo4j Browser: http://localhost:7474")
        print("- Read the tutorial: NEO4J_TUTORIAL.md")
        print("- Run this script again for guided practice")


if __name__ == "__main__":
    main()
