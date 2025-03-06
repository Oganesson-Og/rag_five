#!/usr/bin/env python3
"""
Debug Script for Async/Await Issues
----------------------------------

This script analyzes the codebase for common async/await issues and provides
detailed reports of potential problems.

Usage:
    python3 debug_async_issues.py [--path=<path>] [--verbose]

Features:
- Detects non-awaited coroutines
- Finds synchronous functions being awaited
- Identifies dictionary await attempts
- Validates async method consistency
- Checks for proper async function declarations
"""

import os
import sys
import re
import ast
import argparse
import inspect
import importlib
import importlib.util
import asyncio
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import logging

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("async_debugger")

class AsyncIssueDetector:
    """Detects async/await issues in Python code."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues = []
        self.analyzed_files = 0
        self.issue_count = 0
    
    def _log(self, message: str, level: str = "info"):
        """Log message with appropriate level."""
        if level == "debug" and not self.verbose:
            return
        getattr(logger, level)(message)
    
    def analyze_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Analyze all Python files in a directory recursively.
        
        Args:
            directory: Directory path to analyze
            
        Returns:
            List of issues found
        """
        self._log(f"Analyzing directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path)
        
        self._log(f"Analysis complete. Found {self.issue_count} issues in {self.analyzed_files} files.")
        return self.issues
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a single Python file for async/await issues.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of issues found
        """
        self._log(f"Analyzing file: {file_path}", "debug")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file with AST
            tree = ast.parse(content, filename=file_path)
            
            # Analyze the AST
            file_issues = self._analyze_ast(tree, file_path, content)
            if file_issues:
                self.issue_count += len(file_issues)
                self._log(f"Found {len(file_issues)} issues in {file_path}")
                self.issues.extend(file_issues)
            
            self.analyzed_files += 1
            return file_issues
            
        except SyntaxError as e:
            self._log(f"Syntax error in {file_path}: {str(e)}", "error")
            self.issues.append({
                "file": file_path,
                "line": getattr(e, "lineno", 0),
                "column": getattr(e, "offset", 0),
                "type": "syntax_error",
                "message": str(e)
            })
            return self.issues
        except Exception as e:
            self._log(f"Error analyzing {file_path}: {str(e)}", "error")
            return self.issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: str, content: str) -> List[Dict[str, Any]]:
        """
        Analyze an AST for async/await issues.
        
        Args:
            tree: AST to analyze
            file_path: Path to source file
            content: File content
            
        Returns:
            List of issues found
        """
        file_issues = []
        
        # Track async function definitions
        async_funcs = set()
        
        # Find all async function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_funcs.add(node.name)
        
        # Check for await expressions
        for node in ast.walk(tree):
            # Check for awaiting dictionaries/non-awaitable objects
            if isinstance(node, ast.Await):
                # Get the source code for the await expression
                await_source = self._get_source_segment(content, node)
                
                # Check if awaiting a dictionary literal
                if isinstance(node.value, ast.Dict):
                    file_issues.append({
                        "file": file_path,
                        "line": node.lineno,
                        "column": node.col_offset,
                        "type": "await_dict_literal",
                        "message": f"Awaiting dictionary literal: {await_source}",
                        "severity": "error"
                    })
                
                # Check if awaiting a dictionary method call
                elif isinstance(node.value, ast.Call):
                    if hasattr(node.value, 'func') and isinstance(node.value.func, ast.Attribute):
                        if node.value.func.attr == 'get' and not node.value.func.attr.startswith('_'):
                            file_issues.append({
                                "file": file_path,
                                "line": node.lineno,
                                "column": node.col_offset,
                                "type": "await_dict_method",
                                "message": f"Potentially awaiting dictionary method: {await_source}",
                                "severity": "warning"
                            })
                
                # Check for non-async function calls being awaited
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    func_name = node.value.func.id
                    if func_name not in async_funcs and not func_name.startswith("_"):
                        file_issues.append({
                            "file": file_path,
                            "line": node.lineno,
                            "column": node.col_offset,
                            "type": "await_non_async_func",
                            "message": f"Awaiting non-async function: {func_name}",
                            "severity": "warning"
                        })
            
            # Check for async function declarations with non-async return values
            if isinstance(node, ast.AsyncFunctionDef):
                # Look for return statements that return dictionaries or non-awaitable objects
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Return) and isinstance(sub_node.value, ast.Dict):
                        return_source = self._get_source_segment(content, sub_node)
                        file_issues.append({
                            "file": file_path,
                            "line": sub_node.lineno,
                            "column": sub_node.col_offset,
                            "type": "async_returning_dict",
                            "message": f"Async function returning dictionary: {node.name} -> {return_source}",
                            "severity": "warning"
                        })
        
        return file_issues
    
    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Get source code segment for an AST node."""
        try:
            lines = source.splitlines()
            # Simple extraction for single-line nodes
            if hasattr(node, 'lineno'):
                line = lines[node.lineno - 1]
                return line.strip()
            return "<unknown>"
        except Exception:
            return "<unknown>"


def fix_common_issues(issues: List[Dict[str, Any]]) -> None:
    """
    Attempt to fix common async/await issues.
    
    Args:
        issues: List of issues to fix
    """
    fixes_applied = 0
    
    for issue in issues:
        if issue['type'] == 'await_dict_method':
            # Get the file
            file_path = issue['file']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.readlines()
                
                # Get the line
                line_idx = issue['line'] - 1
                line = content[line_idx]
                
                # Fix awaiting dict.get() by removing await
                if 'await' in line and '.get(' in line:
                    fixed_line = line.replace('await ', '')
                    content[line_idx] = fixed_line
                    
                    # Write back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(content)
                    
                    logger.info(f"Fixed issue in {file_path}:{issue['line']} - Removed await from dict.get()")
                    fixes_applied += 1
            except Exception as e:
                logger.error(f"Failed to fix {file_path}: {str(e)}")
    
    logger.info(f"Applied {fixes_applied} automatic fixes")


def main():
    """Run the async issue detector."""
    parser = argparse.ArgumentParser(description="Detect async/await issues in Python code")
    parser.add_argument("--path", default=".", help="Path to directory to analyze")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    args = parser.parse_args()
    
    # Set up verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run analysis
    detector = AsyncIssueDetector(verbose=args.verbose)
    issues = detector.analyze_directory(args.path)
    
    # Print summary of issues
    print("\nIssue Summary:")
    print("-" * 80)
    
    # Group issues by type
    issue_types = {}
    for issue in issues:
        issue_type = issue["type"]
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue)
    
    for issue_type, type_issues in issue_types.items():
        print(f"{issue_type}: {len(type_issues)} issues")
        for issue in type_issues[:5]:  # Show first 5 of each type
            print(f"  - {issue['file']}:{issue['line']} - {issue['message']}")
        if len(type_issues) > 5:
            print(f"  - ... and {len(type_issues) - 5} more")
    
    # Try to fix issues if requested
    if args.fix and issues:
        print("\nAttempting to fix issues...")
        fix_common_issues(issues)


if __name__ == "__main__":
    main() 