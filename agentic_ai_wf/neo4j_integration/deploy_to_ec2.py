#!/usr/bin/env python3
"""
Automated Neo4j Database Deployment to EC2

This script automates the process of transferring your local Neo4j database
to a remote EC2 server.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from agentic_ai_wf.neo4j_integration.connection import Neo4jConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Neo4jDeployment:
    """
    Handles automated deployment of Neo4j database to EC2.
    """

    def __init__(self,
                 ec2_host: str,
                 ec2_user: str = "ec2-user",
                 key_path: str = "~/.ssh/id_rsa",
                 local_db_name: str = "neo4j",
                 remote_db_name: str = "neo4j"):
        """
        Initialize deployment configuration.

        Args:
            ec2_host: EC2 instance IP or hostname
            ec2_user: SSH username for EC2
            key_path: Path to SSH private key
            local_db_name: Local database name
            remote_db_name: Remote database name
        """
        self.ec2_host = ec2_host
        self.ec2_user = ec2_user
        self.key_path = os.path.expanduser(key_path)
        self.local_db_name = local_db_name
        self.remote_db_name = remote_db_name

        # Deployment paths
        self.local_dump_path = "/tmp/neo4j-deployment-dump.dump"
        self.remote_dump_path = "/tmp/neo4j-deployment-dump.dump"

        # Validate SSH key
        if not os.path.exists(self.key_path):
            raise FileNotFoundError(f"SSH key not found: {self.key_path}")

    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute shell command with logging."""
        logger.info(f"Executing: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stderr:
                logger.error(f"Error: {e.stderr}")
            raise

    def ssh_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute command on remote server via SSH."""
        ssh_cmd = f'ssh -i {self.key_path} {self.ec2_user}@{self.ec2_host} "{command}"'
        return self.run_command(ssh_cmd, check)

    def scp_upload(self, local_path: str, remote_path: str):
        """Upload file to remote server via SCP."""
        scp_cmd = f'scp -i {self.key_path} {local_path} {self.ec2_user}@{self.ec2_host}:{remote_path}'
        self.run_command(scp_cmd)

    def create_local_dump(self) -> bool:
        """Create database dump from local Neo4j."""
        logger.info("Creating local database dump...")

        try:
            # Stop Neo4j
            logger.info("Stopping local Neo4j service...")
            self.run_command("sudo systemctl stop neo4j")

            # Create dump
            logger.info(f"Creating dump of database '{self.local_db_name}'...")
            self.run_command(
                f"sudo neo4j-admin dump --database={self.local_db_name} --to={self.local_dump_path}"
            )

            # Start Neo4j
            logger.info("Starting local Neo4j service...")
            self.run_command("sudo systemctl start neo4j")

            # Verify dump was created
            if not os.path.exists(self.local_dump_path):
                raise FileNotFoundError(
                    f"Dump file not created: {self.local_dump_path}")

            # Get dump size
            dump_size = os.path.getsize(self.local_dump_path)
            logger.info(f"Dump created successfully: {dump_size:,} bytes")

            return True

        except Exception as e:
            logger.error(f"Failed to create local dump: {e}")
            # Try to restart Neo4j if something went wrong
            try:
                self.run_command("sudo systemctl start neo4j", check=False)
            except:
                pass
            return False

    def transfer_dump(self) -> bool:
        """Transfer dump file to EC2 server."""
        logger.info("Transferring dump to EC2 server...")

        try:
            # Compress dump for faster transfer
            compressed_path = f"{self.local_dump_path}.gz"
            logger.info("Compressing dump file...")
            self.run_command(
                f"gzip -c {self.local_dump_path} > {compressed_path}")

            # Transfer compressed file
            logger.info("Uploading to EC2...")
            self.scp_upload(compressed_path, f"{self.remote_dump_path}.gz")

            # Decompress on remote server
            logger.info("Decompressing on EC2...")
            self.ssh_command(f"gunzip {self.remote_dump_path}.gz")

            # Clean up local compressed file
            os.remove(compressed_path)

            logger.info("Transfer completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to transfer dump: {e}")
            return False

    def restore_remote_dump(self) -> bool:
        """Restore database dump on remote server."""
        logger.info("Restoring database on EC2 server...")

        try:
            # Stop remote Neo4j
            logger.info("Stopping remote Neo4j service...")
            self.ssh_command("sudo systemctl stop neo4j")

            # Load database
            logger.info(f"Loading database '{self.remote_db_name}'...")
            self.ssh_command(
                f"sudo neo4j-admin load --database={self.remote_db_name} --from={self.remote_dump_path} --force"
            )

            # Set proper ownership
            logger.info("Setting proper file ownership...")
            self.ssh_command("sudo chown -R neo4j:neo4j /var/lib/neo4j/")

            # Start remote Neo4j
            logger.info("Starting remote Neo4j service...")
            self.ssh_command("sudo systemctl start neo4j")

            # Wait for service to start
            logger.info("Waiting for Neo4j to start...")
            time.sleep(10)

            # Verify service is running
            result = self.ssh_command(
                "sudo systemctl is-active neo4j", check=False)
            if result.stdout.strip() != "active":
                raise RuntimeError("Neo4j service failed to start")

            logger.info("Database restored successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to restore remote dump: {e}")
            # Try to start Neo4j if something went wrong
            try:
                self.ssh_command("sudo systemctl start neo4j", check=False)
            except:
                pass
            return False

    def verify_deployment(self, remote_password: str) -> bool:
        """Verify the deployment was successful."""
        logger.info("Verifying deployment...")

        try:
            # Connect to remote database
            remote_db = Neo4jConnection(
                uri=f"bolt://{self.ec2_host}:7687",
                username="neo4j",
                password=remote_password
            )

            # Test connection
            if not remote_db.test_connection():
                raise RuntimeError("Failed to connect to remote database")

            # Get database statistics
            stats = remote_db.get_database_stats()
            logger.info("Remote database statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value:,}")

            # Test a simple query
            result = remote_db.execute_query(
                "MATCH (n) RETURN count(n) as total_nodes")
            total_nodes = result[0]["total_nodes"]

            if total_nodes == 0:
                logger.warning("No nodes found in remote database")
                return False

            logger.info(
                f"✓ Deployment verified: {total_nodes:,} nodes in remote database")
            remote_db.close()
            return True

        except Exception as e:
            logger.error(f"Deployment verification failed: {e}")
            return False

    def cleanup(self):
        """Clean up temporary files."""
        logger.info("Cleaning up temporary files...")

        try:
            # Remove local dump
            if os.path.exists(self.local_dump_path):
                os.remove(self.local_dump_path)
                logger.info("Local dump file removed")

            # Remove remote dump
            self.ssh_command(f"rm -f {self.remote_dump_path}", check=False)
            logger.info("Remote dump file removed")

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def deploy(self, remote_password: str, skip_verification: bool = False) -> bool:
        """
        Execute complete deployment process.

        Args:
            remote_password: Password for remote Neo4j instance
            skip_verification: Skip deployment verification

        Returns:
            True if deployment successful, False otherwise
        """
        start_time = datetime.now()
        logger.info(f"Starting Neo4j deployment to {self.ec2_host}")

        success = False
        try:
            # Step 1: Create local dump
            if not self.create_local_dump():
                return False

            # Step 2: Transfer to EC2
            if not self.transfer_dump():
                return False

            # Step 3: Restore on EC2
            if not self.restore_remote_dump():
                return False

            # Step 4: Verify deployment
            if not skip_verification:
                if not self.verify_deployment(remote_password):
                    return False

            success = True

        except KeyboardInterrupt:
            logger.error("Deployment interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
        finally:
            # Always cleanup
            self.cleanup()

            # Log completion
            duration = datetime.now() - start_time
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Deployment {status} in {duration}")

        return success


def main():
    """Main deployment script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy Neo4j database to EC2")
    parser.add_argument("ec2_host", help="EC2 instance IP or hostname")
    parser.add_argument("--user", default="ec2-user", help="SSH username")
    parser.add_argument("--key", default="~/.ssh/id_rsa",
                        help="SSH private key path")
    parser.add_argument("--password", required=True,
                        help="Remote Neo4j password")
    parser.add_argument("--local-db", default="neo4j",
                        help="Local database name")
    parser.add_argument("--remote-db", default="neo4j",
                        help="Remote database name")
    parser.add_argument("--skip-verification", action="store_true",
                        help="Skip deployment verification")

    args = parser.parse_args()

    # Create deployment instance
    deployment = Neo4jDeployment(
        ec2_host=args.ec2_host,
        ec2_user=args.user,
        key_path=args.key,
        local_db_name=args.local_db,
        remote_db_name=args.remote_db
    )

    # Execute deployment
    success = deployment.deploy(args.password, args.skip_verification)

    if success:
        print(f"\n✓ Deployment completed successfully!")
        print(f"Remote Neo4j Browser: http://{args.ec2_host}:7474")
        print(f"Connection URI: bolt://{args.ec2_host}:7687")
        sys.exit(0)
    else:
        print(f"\n✗ Deployment failed. Check deployment.log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
