"""
Multi-task Deep Learning for Oral Cancer Analysis
================================================

This package contains modules for:
- Data preprocessing and patch extraction from WSIs
- Multi-task model architectures for TVNT, DOI, POI, TB, PNI, MI
- Training and evaluation utilities
- Visualization and analysis tools

Author: Your Name
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import get_config, is_local_environment, print_config_summary

# Define what's available when importing the package
__all__ = [
    "get_config",
    "is_local_environment", 
    "print_config_summary",
]

"""
Source code package for Multi-task Deep Learning Oral Cancer Analysis
"""

# This file makes the src directory a Python package
pass