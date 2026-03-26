# Neo4j Database Transfer to EC2 Server

## Overview

This guide covers transferring your local Neo4j database with GeneCards data to a remote EC2 server for production use.

## Method 1: Database Dump/Restore (Recommended)

### Step 1: Create Database Dump (Local)

```bash
# Stop Neo4j service
sudo systemctl stop neo4j

# Create dump file
sudo neo4j-admin dump --database=neo4j --to=/tmp/neo4j-dump.dump

# Restart Neo4j
sudo systemctl start neo4j
```

### Step 2: Transfer to EC2

```bash
# Compress for faster transfer
gzip /tmp/neo4j-dump.dump

# Transfer to EC2 (replace with your EC2 details)
scp -i your-key.pem /tmp/neo4j-dump.dump.gz ec2-user@your-ec2-ip:/tmp/

# Or use rsync for better reliability
rsync -avz -e "ssh -i your-key.pem" /tmp/neo4j-dump.dump.gz ec2-user@your-ec2-ip:/tmp/
```

### Step 3: Restore on EC2

```bash
# On EC2 server
sudo systemctl stop neo4j

# Decompress
gunzip /tmp/neo4j-dump.dump.gz

# Load database
sudo neo4j-admin load --database=neo4j --from=/tmp/neo4j-dump.dump --force

# Set proper ownership
sudo chown -R neo4j:neo4j /var/lib/neo4j/

# Start Neo4j
sudo systemctl start neo4j
```

## Method 2: APOC Export/Import

### Step 1: Export Data (Local)

```cypher
// Export all nodes and relationships
CALL apoc.export.cypher.all("/tmp/neo4j-export.cypher", {
    format: "cypher-shell",
    useOptimizations: {type: "UNWIND_BATCH", unwindBatchSize: 20}
})

// Or export specific data
CALL apoc.export.cypher.query(
    "MATCH (n:Gene)-[r]->(m) RETURN n,r,m",
    "/tmp/genes-export.cypher",
    {}
)
```

### Step 2: Transfer and Import

```bash
# Transfer file
scp -i your-key.pem /tmp/neo4j-export.cypher ec2-user@your-ec2-ip:/tmp/

# On EC2: Import via cypher-shell
cypher-shell -u neo4j -p password -f /tmp/neo4j-export.cypher
```

## Method 3: Incremental Data Migration

### Step 1: Create Migration Script

```python
# migration_script.py
import json
from neo4j_integration.connection import Neo4jConnection
from neo4j_integration.genecards_importer import GeneCardsImporter

def migrate_data():
    # Local connection
    local_db = Neo4jConnection(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )

    # Remote connection
    remote_db = Neo4jConnection(
        uri="bolt://your-ec2-ip:7687",
        username="neo4j",
        password="your-remote-password"
    )

    # Export data from local
    export_query = """
    MATCH (g:Gene)
    RETURN g.symbol as symbol, g.ensembl_id as ensembl_id,
           g.description as description, g.aliases as aliases
    """

    genes = local_db.execute_query(export_query)

    # Import to remote
    importer = GeneCardsImporter(remote_db)
    for gene in genes:
        # Process and import each gene
        pass

    local_db.close()
    remote_db.close()
```

## EC2 Setup for Neo4j

### Step 1: Install Neo4j on EC2

```bash
# Update system
sudo yum update -y  # Amazon Linux
# sudo apt update    # Ubuntu

# Install Java
sudo yum install -y java-11-amazon-corretto-headless

# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install neo4j=1:5.26.0
```

### Step 2: Configure Neo4j for Remote Access

```bash
# Edit Neo4j configuration
sudo nano /etc/neo4j/neo4j.conf

# Uncomment and modify these lines:
server.default_listen_address=0.0.0.0
server.bolt.listen_address=0.0.0.0:7687
server.http.listen_address=0.0.0.0:7474

# Set initial password
sudo neo4j-admin set-initial-password your-secure-password
```

### Step 3: Configure Security Groups

```bash
# AWS CLI example
aws ec2 authorize-security-group-ingress \
    --group-id sg-your-security-group \
    --protocol tcp \
    --port 7474 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-your-security-group \
    --protocol tcp \
    --port 7687 \
    --cidr 0.0.0.0/0
```

## Production Considerations

### Performance Optimization

```bash
# /etc/neo4j/neo4j.conf
server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g

# Enable query logging
server.logs.query.enabled=true
server.logs.query.threshold=1000ms
```

### Security Hardening

```bash
# Enable authentication
server.security.auth_enabled=true

# Configure SSL (recommended for production)
server.bolt.tls_level=REQUIRED
server.https.enabled=true
server.bolt.tls_level=REQUIRED
```

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/neo4j"
mkdir -p $BACKUP_DIR

# Create backup
sudo neo4j-admin dump --database=neo4j --to=$BACKUP_DIR/neo4j-backup-$DATE.dump

# Upload to S3
aws s3 cp $BACKUP_DIR/neo4j-backup-$DATE.dump s3://your-backup-bucket/neo4j/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "*.dump" -mtime +7 -delete
```

## Monitoring and Maintenance

### Health Check Script

```python
# health_check.py
from neo4j_integration.connection import Neo4jConnection

def check_neo4j_health():
    try:
        db = Neo4jConnection(
            uri="bolt://your-ec2-ip:7687",
            username="neo4j",
            password="your-password"
        )

        # Test connection
        if db.test_connection():
            print("✓ Neo4j connection successful")

            # Check data integrity
            stats = db.get_database_stats()
            print(f"✓ Database stats: {stats}")

            # Check indexes
            indexes = db.execute_query("SHOW INDEXES")
            print(f"✓ Active indexes: {len(indexes)}")

        db.close()

    except Exception as e:
        print(f"✗ Neo4j health check failed: {e}")
```

### Docker Deployment (Alternative)

```dockerfile
# Dockerfile
FROM neo4j:5.26.0

# Copy data
COPY --chown=neo4j:neo4j neo4j-dump.dump /var/lib/neo4j/

# Custom configuration
COPY neo4j.conf /var/lib/neo4j/conf/

# Set environment variables
ENV NEO4J_AUTH=neo4j/your-password
ENV NEO4J_server_memory_heap_initial__size=2g
ENV NEO4J_server_memory_heap_max__size=4g

EXPOSE 7474 7687
```

```bash
# Build and run
docker build -t neo4j-genecards .
docker run -d -p 7474:7474 -p 7687:7687 neo4j-genecards
```

## Data Validation After Transfer

### Verification Script

```python
# verify_transfer.py
def verify_data_integrity():
    db = Neo4jConnection(uri="bolt://your-ec2-ip:7687")

    # Check node counts
    stats = db.get_database_stats()
    print(f"Genes: {stats['gene_count']}")
    print(f"Diseases: {stats['disease_count']}")
    print(f"Drugs: {stats['drug_count']}")

    # Sample data checks
    sample_genes = db.execute_query("""
        MATCH (g:Gene)
        WHERE g.symbol IS NOT NULL
        RETURN g.symbol, g.ensembl_id
        LIMIT 10
    """)

    print("Sample genes transferred:")
    for gene in sample_genes:
        print(f"  {gene['g.symbol']} ({gene['g.ensembl_id']})")

    db.close()
```

## Best Practices Summary

1. **Use Method 1 (dump/restore)** for complete database transfer
2. **Test the transfer** with a small dataset first
3. **Set up proper security** (authentication, SSL, firewall)
4. **Configure monitoring** and health checks
5. **Implement backup strategy** with S3 or EBS snapshots
6. **Optimize performance** for your instance type
7. **Use Docker** for consistent deployments
8. **Document your configuration** for team members

## Estimated Transfer Times

For your 10GB GeneCards dataset:

- **Dump creation**: 10-20 minutes
- **Transfer to EC2**: 5-15 minutes (depends on bandwidth)
- **Restore on EC2**: 15-30 minutes
- **Total**: ~30-60 minutes

The dump/restore method is the most reliable and efficient for large datasets like yours.
