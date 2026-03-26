#!/usr/bin/env python3
"""
Directory Manager for Cohort Retrieval Agent System.

This utility provides management and monitoring capabilities for the
scalable directory structure system.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Project Imports
from .config import DirectoryPathsConfig

class DirectoryManager:
    """
    Manager for cohort retrieval agent directory operations.
    
    Provides utilities for:
    - Creating and managing directory structures
    - Monitoring storage usage
    - Cleaning up old files
    - Generating reports
    """
    
    def __init__(self, config: Optional[DirectoryPathsConfig] = None):
        self.config = config or DirectoryPathsConfig()
        self.timestamp = datetime.now().isoformat()
    
    def create_agent_structure(self, agent_name: str, disease_name: str) -> Dict[str, Path]:
        """Create complete directory structure for an agent/disease combination."""
        try:
            paths = self.config.create_all_directories(agent_name, disease_name)
            print(f" Created directory structure for {agent_name.upper()}/{disease_name}")
            print(f" Base path: {paths['base']}")
            print(f" Created {len(paths)} directories")
            return paths
        except Exception as e:
            print(f" Failed to create directory structure: {e}")
            raise
    
    def list_all_structures(self) -> Dict[str, List[str]]:
        """List all existing agent/disease directory structures."""
        structures = {}
        
        try:
            agents = self.config.list_all_agents()
            for agent in agents:
                diseases = self.config.list_diseases_for_agent(agent)
                structures[agent] = diseases
                print(f"🔬 {agent.upper()}: {len(diseases)} diseases")
                for disease in diseases:
                    print(f"  - {disease}")
            
            return structures
        except Exception as e:
            print(f"❌ Failed to list structures: {e}")
            return {}
    
    def get_storage_report(self) -> Dict[str, Any]:
        """Generate comprehensive storage usage report."""
        try:
            summary = self.config.get_storage_summary()
            
            print(" Storage Usage Report")
            print("=" * 50)
            print(f"Base path: {summary['base_path']}")
            print(f"Total size: {summary['total_size_mb']:.2f} MB")
            print()
            
            for agent_name, agent_data in summary['agents'].items():
                print(f"🔬 {agent_name.upper()}: {agent_data['total_size_mb']:.2f} MB")
                for disease_name, disease_data in agent_data['diseases'].items():
                    print(f"  - {disease_name}: {disease_data['size_mb']:.2f} MB")
                    print(f"    Path: {disease_data['path']}")
                print()
            
            return summary
        except Exception as e:
            print(f"❌ Failed to generate storage report: {e}")
            return {}
    
    def cleanup_old_files(self, agent_name: str = None, disease_name: str = None, 
                         max_age_days: int = None) -> bool:
        """Clean up old archive files."""
        try:
            if max_age_days:
                self.config.max_archive_age_days = max_age_days
            
            self.config.cleanup_old_archives(agent_name, disease_name)
            
            scope = "all agents"
            if agent_name and disease_name:
                scope = f"{agent_name.upper()}/{disease_name}"
            elif agent_name:
                scope = f"{agent_name.upper()}"
            
            print(f"🧹 Cleaned up old files for {scope}")
            print(f"📅 Files older than {self.config.max_archive_age_days} days removed")
            return True
        except Exception as e:
            print(f"❌ Failed to cleanup files: {e}")
            return False
    
    def validate_structure(self, agent_name: str, disease_name: str) -> Dict[str, Any]:
        """Validate directory structure for an agent/disease combination."""
        try:
            structure = self.config.get_directory_structure(agent_name, disease_name)
            
            validation_results = {
                "agent_name": agent_name,
                "disease_name": disease_name,
                "base_path": structure["base_path"],
                "validation_time": self.timestamp,
                "directories": {},
                "files": {},
                "issues": [],
                "status": "valid"
            }
            
            # Check directories
            for dir_type, dir_info in structure["directories"].items():
                dir_path = Path(dir_info["path"])
                validation_results["directories"][dir_type] = {
                    "exists": dir_info["exists"],
                    "readable": dir_path.is_dir() and os.access(dir_path, os.R_OK) if dir_info["exists"] else False,
                    "writable": dir_path.is_dir() and os.access(dir_path, os.W_OK) if dir_info["exists"] else False,
                    "size_mb": dir_info["size_mb"]
                }
                
                if not dir_info["exists"]:
                    validation_results["issues"].append(f"Directory {dir_type} does not exist: {dir_info['path']}")
            
            # Check standard files
            for file_type, file_path in structure["files"].items():
                file_path_obj = Path(file_path)
                validation_results["files"][file_type] = {
                    "exists": file_path_obj.exists(),
                    "readable": file_path_obj.is_file() and os.access(file_path_obj, os.R_OK) if file_path_obj.exists() else False,
                    "size_bytes": file_path_obj.stat().st_size if file_path_obj.exists() else 0
                }
            
            if validation_results["issues"]:
                validation_results["status"] = "issues_found"
            
            # Print validation results
            print(f" Validation Results for {agent_name.upper()}/{disease_name}")
            print(f"Status: {' Valid' if validation_results['status'] == 'valid' else '⚠️ Issues Found'}")
            print(f"Base path: {validation_results['base_path']}")
            print(f"Directories: {len(validation_results['directories'])}")
            print(f"Files: {len(validation_results['files'])}")
            
            if validation_results["issues"]:
                print("Issues found:")
                for issue in validation_results["issues"]:
                    print(f"  - {issue}")
            
            return validation_results
        except Exception as e:
            print(f"❌ Failed to validate structure: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_config_template(self, output_path: str = None) -> str:
        """Generate a configuration template file."""
        try:
            template = {
                "directory_paths": {
                    "base_project_dir": "agentic_ai_wf",
                    "shared_data_dir": "shared",
                    "cohort_data_dir": "cohort_data",
                    "geo_agent_dir": "GEO",
                    "sra_agent_dir": "SRA",
                    "tcga_agent_dir": "TCGA",
                    "gtex_agent_dir": "GTEx",
                    "arrayexpress_agent_dir": "ArrayExpress",
                    "subdirectories": {
                        "raw_data": "raw_data",
                        "processed_data": "processed_data",
                        "metadata": "metadata",
                        "logs": "logs",
                        "temp": "temp",
                        "cache": "cache",
                        "reports": "reports",
                        "validation": "validation",
                        "archive": "archive",
                        "backup": "backup",
                        "tracking": "tracking"
                    },
                    "file_patterns": {
                        "dataset_summary": "dataset_summary.json",
                        "download_log": "download_log.txt",
                        "validation_report": "validation_report.json",
                        "metadata_index": "metadata_index.json",
                        "usage_stats": "usage_stats.json",
                        "performance_metrics": "performance_metrics.json"
                    },
                    "settings": {
                        "enable_archiving": True,
                        "max_archive_age_days": 30,
                        "enable_tracking": True
                    }
                },
                "environment_variables": {
                    "AGENTIC_AI_BASE_DIR": "Override base project directory",
                    "SHARED_DATA_DIR": "Override shared data directory name",
                    "COHORT_DATA_DIR": "Override cohort data directory name"
                },
                "usage_examples": {
                    "get_raw_data_path": "config.directory_paths.get_raw_data_path('geo', 'lupus')",
                    "create_directories": "config.directory_paths.create_all_directories('geo', 'lupus')",
                    "get_file_path": "config.directory_paths.get_file_path('geo', 'lupus', 'dataset_summary')",
                    "list_agents": "config.directory_paths.list_all_agents()",
                    "storage_summary": "config.directory_paths.get_storage_summary()"
                }
            }
            
            output_path = output_path or "directory_config_template.json"
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
            
            print(f"📄 Configuration template saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Failed to generate config template: {e}")
            return ""
    
    def export_structure_info(self, output_path: str = None) -> str:
        """Export complete directory structure information."""
        try:
            export_data = {
                "export_time": self.timestamp,
                "base_path": str(self.config.get_base_cohort_path()),
                "agents": {},
                "summary": {
                    "total_agents": 0,
                    "total_diseases": 0,
                    "total_size_mb": 0
                }
            }
            
            agents = self.config.list_all_agents()
            export_data["summary"]["total_agents"] = len(agents)
            
            for agent in agents:
                diseases = self.config.list_diseases_for_agent(agent)
                export_data["summary"]["total_diseases"] += len(diseases)
                
                agent_data = {
                    "diseases": {},
                    "total_size_mb": 0
                }
                
                for disease in diseases:
                    structure = self.config.get_directory_structure(agent, disease)
                    agent_data["diseases"][disease] = structure
                    
                    # Calculate size
                    for dir_info in structure["directories"].values():
                        agent_data["total_size_mb"] += dir_info["size_mb"]
                
                export_data["agents"][agent] = agent_data
                export_data["summary"]["total_size_mb"] += agent_data["total_size_mb"]
            
            output_path = output_path or f"directory_structure_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f" Structure information exported to: {output_path}")
            print(f" Summary: {export_data['summary']}")
            return output_path
        except Exception as e:
            print(f" Failed to export structure info: {e}")
            return ""


def main():
    """Command-line interface for directory management."""
    parser = argparse.ArgumentParser(description="Cohort Retrieval Agent Directory Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create structure command
    create_parser = subparsers.add_parser("create", help="Create directory structure")
    create_parser.add_argument("agent", help="Agent name (geo, sra, tcga, gtex, arrayexpress)")
    create_parser.add_argument("disease", help="Disease name")
    
    # List structures command
    list_parser = subparsers.add_parser("list", help="List all directory structures")
    
    # Storage report command
    storage_parser = subparsers.add_parser("storage", help="Generate storage usage report")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old files")
    cleanup_parser.add_argument("--agent", help="Specific agent to clean")
    cleanup_parser.add_argument("--disease", help="Specific disease to clean")
    cleanup_parser.add_argument("--max-age", type=int, default=30, help="Maximum age in days")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate directory structure")
    validate_parser.add_argument("agent", help="Agent name")
    validate_parser.add_argument("disease", help="Disease name")
    
    # Template command
    template_parser = subparsers.add_parser("template", help="Generate configuration template")
    template_parser.add_argument("--output", help="Output file path")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export structure information")
    export_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        manager = DirectoryManager()
        
        if args.command == "create":
            manager.create_agent_structure(args.agent, args.disease)
        elif args.command == "list":
            manager.list_all_structures()
        elif args.command == "storage":
            manager.get_storage_report()
        elif args.command == "cleanup":
            manager.cleanup_old_files(args.agent, args.disease, args.max_age)
        elif args.command == "validate":
            manager.validate_structure(args.agent, args.disease)
        elif args.command == "template":
            manager.generate_config_template(args.output)
        elif args.command == "export":
            manager.export_structure_info(args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 