#!/usr/bin/env python3
"""
Neo4j Setup and Diagnosis Script

This script helps diagnose and fix common Neo4j connection issues.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_neo4j_service():
    """Check if Neo4j service is running."""
    try:
        result = subprocess.run(['systemctl', 'is-active', 'neo4j'],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == 'active':
            print("✅ Neo4j service is running")
            return True
        else:
            print("❌ Neo4j service is not running")
            return False
    except FileNotFoundError:
        print("⚠️  systemctl not found (not a systemd system)")
        return None


def check_neo4j_process():
    """Check if Neo4j process is running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'neo4j'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Neo4j process is running")
            return True
        else:
            print("❌ Neo4j process is not running")
            return False
    except FileNotFoundError:
        print("⚠️  pgrep not found")
        return None


def check_neo4j_port():
    """Check if Neo4j port is open."""
    try:
        result = subprocess.run(['netstat', '-ln'],
                                capture_output=True, text=True)
        if ':7687' in result.stdout or ':7474' in result.stdout:
            print("✅ Neo4j ports are open")
            return True
        else:
            print("❌ Neo4j ports (7687, 7474) are not open")
            return False
    except FileNotFoundError:
        print("⚠️  netstat not found")
        return None


def try_start_neo4j():
    """Try to start Neo4j service."""
    try:
        print("🔄 Attempting to start Neo4j service...")
        result = subprocess.run(['sudo', 'systemctl', 'start', 'neo4j'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Neo4j service started successfully")
            return True
        else:
            print(f"❌ Failed to start Neo4j service: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error starting Neo4j: {e}")
        return False


def reset_neo4j_password():
    """Reset Neo4j password."""
    try:
        print("🔄 Resetting Neo4j password...")
        result = subprocess.run(['sudo', 'neo4j-admin', 'set-initial-password', 'password'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Neo4j password reset to 'password'")
            return True
        else:
            print(f"❌ Failed to reset password: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error resetting password: {e}")
        return False


def test_connection_with_password(password):
    """Test connection with given password."""
    try:
        from neo4j_integration.connection import Neo4jConnection

        db = Neo4jConnection(
            uri="bolt://localhost:7687",
            username="neo4j",
            password=password
        )

        if db.test_connection():
            print(f"✅ Connection successful with password: '{password}'")
            db.close()
            return True
        else:
            print(f"❌ Connection failed with password: '{password}'")
            return False

    except Exception as e:
        print(f"❌ Connection error with password '{password}': {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Neo4j Setup and Diagnosis")
    print("=" * 50)

    # Check if Neo4j is installed
    neo4j_installed = False
    try:
        result = subprocess.run(['neo4j', '--version'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Neo4j installed: {result.stdout.strip()}")
            neo4j_installed = True
        else:
            print("❌ Neo4j not found")
    except FileNotFoundError:
        print("❌ Neo4j not installed")

    if not neo4j_installed:
        print("\n💡 Install Neo4j:")
        print("Ubuntu/Debian: sudo apt install neo4j")
        print("Or download from: https://neo4j.com/download/")
        return False

    # Check service status
    print("\n🔍 Checking Neo4j status...")
    service_running = check_neo4j_service()
    process_running = check_neo4j_process()
    port_open = check_neo4j_port()

    # Try to start if not running
    if not service_running and not process_running:
        if try_start_neo4j():
            # Wait a moment for service to start
            import time
            time.sleep(3)
            service_running = check_neo4j_service()

    if not service_running and not process_running:
        print("\n❌ Neo4j is not running and couldn't be started")
        print("💡 Try manually: sudo systemctl start neo4j")
        return False

    # Test common passwords
    print("\n🔐 Testing common passwords...")
    common_passwords = ['password', 'neo4j', 'admin', '123456', '']

    for password in common_passwords:
        if test_connection_with_password(password):
            print(f"\n🎉 Success! Use password: '{password}'")

            # Update example file with correct password
            try:
                example_file = Path(__file__).parent / 'example_test.py'
                if example_file.exists():
                    content = example_file.read_text()
                    content = content.replace(
                        'password="neo4j"', f'password="{password}"')
                    example_file.write_text(content)
                    print(f"✅ Updated example_test.py with correct password")
            except Exception as e:
                print(f"⚠️  Couldn't update example file: {e}")

            return True

    # Try to reset password
    print("\n🔄 Trying to reset password...")
    if reset_neo4j_password():
        if test_connection_with_password('password'):
            print("\n🎉 Success! Password reset to 'password'")
            return True

    print("\n❌ All connection attempts failed")
    print("💡 Manual steps:")
    print("1. sudo systemctl stop neo4j")
    print("2. sudo rm -rf /var/lib/neo4j/data/dbms/auth")
    print("3. sudo systemctl start neo4j")
    print("4. Connect with username=neo4j, password=neo4j")

    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
