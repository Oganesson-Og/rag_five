"""
Logging Configuration Module
--------------------------

This module provides a simplified interface for configuring logging in the RAG system.
It serves as a wrapper around the more detailed logging functionality in the logging.py module.

Key Features:
- Simple configuration interface
- Default settings for common use cases
- Integration with existing logging system

Dependencies:
- logging
- pathlib
- src.utils.logging

Example Usage:
    # Configure logging with defaults
    configure_logging()
    
    # Configure with custom path
    configure_logging(log_dir="custom/log/path")
    
    # Configure with custom level
    configure_logging(log_level="DEBUG")

Author: Keith Satuku
Version: 1.0.0
Created: 2023
License: MIT
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

from src.utils.logging import setup_logging, LogConfig

def configure_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: int = 1024 * 1024,  # 1MB
    backup_count: int = 3
) -> logging.Logger:
    """
    Configure logging for the RAG system with sensible defaults.
    
    Args:
        log_dir: Directory to store log files. Defaults to 'logs' in the current directory.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.
        log_format: Log message format. Defaults to timestamp, name, level, and message.
        max_bytes: Maximum size of log file before rotation in bytes. Defaults to 1MB.
        backup_count: Number of backup files to keep. Defaults to 3.
        
    Returns:
        Configured logger instance
    """
    # Set default log directory if not provided
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set default log file path
    log_file = log_dir / "rag_pipeline.log"
    
    # Convert log level string to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO) if log_level else logging.INFO
    
    # Create log config
    config = LogConfig(
        log_file=log_file,
        level=level,
        format=log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        max_bytes=max_bytes,
        backup_count=backup_count
    )
    
    # Setup and return logger
    return setup_logging(config) 