"""
Logging configuration for clinical trials embedding processing
"""

import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file=None, enable_file_logging=None):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path. If None, creates timestamped log file.
        enable_file_logging: If True, creates file handler. If None, auto-detects based on environment.
    
    Returns:
        tuple: (logger, log_file_path or None)
    """
    # Auto-detect environment if not specified
    if enable_file_logging is None:
        # Try to check if we're in a Streamlit production environment
        is_production = False
        try:
            import streamlit as st
            is_production = st.secrets.get("ENVIRONMENT") == "production"
        except (ImportError, FileNotFoundError, Exception):
            # Not in Streamlit context or secrets not available - default to development
            is_production = False
        
        enable_file_logging = not is_production
    
    log_file_path = None
    
    if enable_file_logging:
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate log file name if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(log_dir, f"clinical_trials_processing_{timestamp}.log")
        else:
            log_file_path = log_file
    
    # Create logger
    logger = logging.getLogger('clinical_trials')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create file handler only if file logging is enabled
    if enable_file_logging and log_file_path:
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Create console handler (only show important messages)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    if log_file_path:
        logger.info(f"Logging configured. Log file: {log_file_path}")
    else:
        logger.info("Logging configured (console only)")
    
    return logger, log_file_path


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        logger: The clinical_trials logger
    """
    return logging.getLogger('clinical_trials')


# Progress tracking utilities
class ProgressTracker:
    """Simple progress tracker for concise console output"""
    
    def __init__(self, total, description="Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = get_logger()
        
    def update(self, increment=1):
        """Update progress and log if milestone reached"""
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        # Log detailed progress to file
        self.logger.debug(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
        
        # Show concise progress on console at certain milestones
        if percentage in [0, 25, 50, 75, 100] or self.current == self.total:
            print(f"\r{self.description}: {percentage:.0f}% ({self.current}/{self.total})", end="", flush=True)
            
        if self.current == self.total:
            print()  # New line when complete
    
    def complete(self, success_count=None):
        """Mark as complete with optional success count"""
        if success_count is not None:
            success_rate = (success_count / self.total) * 100
            self.logger.info(f"{self.description} complete: {success_count}/{self.total} successful ({success_rate:.1f}%)")
            print(f"✅ {self.description} complete: {success_count}/{self.total} successful ({success_rate:.1f}%)")
        else:
            self.logger.info(f"{self.description} complete: {self.current}/{self.total}")
            print(f"✅ {self.description} complete")
