"""
Neo4j Database Connection Manager

Handles Neo4j database connections, sessions, and basic operations.
"""

import logging
from typing import Dict, List, Any, Optional, cast, LiteralString
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from decouple import config

logger = logging.getLogger(__name__)

NEO4J_URI = str(config("NEO4J_URI", default="bolt://localhost:7687"))
NEO4J_USERNAME = str(config("NEO4J_USERNAME", default="neo4j"))
NEO4J_PASSWORD = str(config("NEO4J_PASSWORD", default="password"))
NEO4J_DATABASE = str(config("NEO4J_DATABASE", default="neo4j"))


class Neo4jConnection:
    """
    Neo4j database connection manager with automatic retry and error handling.
    """

    def __init__(self,
                 uri: str = NEO4J_URI,
                 username: str = NEO4J_USERNAME,
                 password: str = NEO4J_PASSWORD,
                 database: str = NEO4J_DATABASE,
                 lazy: bool = True):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password
            database: Database name
            lazy: If True, connect only on first query (default). If False, connect immediately.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self._connected = False
        self._connection_id = id(self)  # Unique ID for debugging

        if not lazy:
            self._connect()

    def _connect(self):
        """Establish connection to Neo4j database (lazy - only when needed)."""
        # Check if already connected and driver is valid
        if self._connected and self.driver:
            # Skip expensive verify_connectivity() - let query execution test the connection
            return
        
        # Create new connection
        try:
            self.driver = GraphDatabase.driver(
                str(self.uri),
                auth=(str(self.username), str(self.password))
            )
            # Test the connection (only on initial connect)
            self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Successfully connected to Neo4j at {self.uri} (connection_id={self._connection_id})")
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        # Lazy connection - connect on first query if not already connected
        if not self._connected or not self.driver:
            self._connect()
        
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")

        parameters = parameters or {}

        try:
            with self.driver.session(database=str(self.database)) as session:
                result = session.run(cast(LiteralString, query), parameters)
                return [record.data() for record in result]
        except Exception as e:
            # If connection is dead, reset and try once more
            if "Connection" in str(e) or "closed" in str(e).lower() or "disconnected" in str(e).lower():
                logger.warning(f"Neo4j connection lost, reconnecting: {e}")
                self._connected = False
                self.driver = None
                self._connect()
                # Retry once
                with self.driver.session(database=str(self.database)) as session:
                    result = session.run(cast(LiteralString, query), parameters)
                    return [record.data() for record in result]
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise

    def execute_write(self, query: str, parameters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a write transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        # Lazy connection - connect on first query if not already connected
        if not self._connected:
            self._connect()
        
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")

        parameters = parameters or {}

        def write_tx(tx):
            result = tx.run(cast(LiteralString, query), parameters)
            return [record.data() for record in result]

        try:
            with self.driver.session(database=str(self.database)) as session:
                return session.execute_write(write_tx)
        except Exception as e:
            logger.error(f"Write transaction failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise

    def create_indexes(self):
        """Create recommended indexes for better performance."""
        indexes = [
            "CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)",
            "CREATE INDEX gene_ensembl IF NOT EXISTS FOR (g:Gene) ON (g.ensembl_id)",
            "CREATE INDEX pathway_id IF NOT EXISTS FOR (p:Pathway) ON (p.pathway_id)",
            "CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX disease_name IF NOT EXISTS FOR (dis:Disease) ON (dis.name)",
        ]

        for index_query in indexes:
            try:
                self.execute_write(index_query)
                logger.info(f"Created index: {index_query}")
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        query = "MATCH (n) DETACH DELETE n"
        try:
            self.execute_write(query)
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise

    def get_database_stats(self) -> Dict[str, int]:
        """Get basic database statistics."""
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "gene_count": "MATCH (g:Gene) RETURN count(g) as count",
            "pathway_count": "MATCH (p:Pathway) RETURN count(p) as count",
            "drug_count": "MATCH (d:Drug) RETURN count(d) as count",
            "disease_count": "MATCH (dis:Disease) RETURN count(dis) as count"
        }

        stats = {}
        for stat_name, query in queries.items():
            try:
                result = self.execute_query(query)
                stats[stat_name] = result[0]["count"] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get {stat_name}: {e}")
                stats[stat_name] = 0

        return stats

    def test_connection(self) -> bool:
        """Test if the connection is working."""
        try:
            result = self.execute_query("RETURN 1 as test")
            return len(result) > 0 and result[0]["test"] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j connection closed")
            except Exception:
                pass  # Ignore errors when closing
            finally:
                self.driver = None
                self._connected = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
