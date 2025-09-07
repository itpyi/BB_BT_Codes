"""Shared utility functions for BB code simulations.

Contains common helper functions used across multiple modules to avoid
code duplication and ensure consistency.
"""

from __future__ import annotations

import json
import csv
import math
from typing import Dict, Any, Optional, Tuple, cast


# Simplified error handling - let standard exceptions bubble up naturally


def safe_json_loads(s: str) -> Dict[str, Any]:
    """Parse JSON string with fallback for malformed input.
    
    Args:
        s: JSON string to parse
        
    Returns:
        Parsed JSON as dictionary, empty dict if parsing fails
    """
    if not isinstance(s, str):
        return {}
    
    # Try standard JSON parsing first
    try:
        result = json.loads(s)
        return cast(Dict[str, Any], result) if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try fallback for malformed quotes
    try:
        result = json.loads(s.replace("''", '"').replace('""', '"'))
        return cast(Dict[str, Any], result) if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def existing_counts_for_point(
    resume_csv: Optional[str], *, meta_filter: Dict[str, Any]
) -> Tuple[int, int]:
    """Aggregate prior shots/errors from a resume CSV for a specific point.

    Args:
        resume_csv: Path to resume CSV file, or None
        meta_filter: Dictionary containing metadata to match against
        
    Returns:
        Tuple of (total_shots, total_errors) from matching rows
    """
    if not resume_csv:
        return 0, 0
    
    total_shots = 0
    total_errors = 0
    
    try:
        with open(resume_csv, newline="") as f:
            for row in csv.DictReader(f):
                meta_str = row.get("meta") or row.get("json_metadata", "{}")
                if not meta_str:
                    continue
                
                meta = safe_json_loads(meta_str)
                if not meta:
                    continue
                
                # Check if this row matches our filter
                match = True
                for key, expected_value in meta_filter.items():
                    meta_value = meta.get(key)
                    if meta_value is None:
                        match = False
                        break
                    
                    if key in ("l", "m", "rounds"):
                        if int(meta_value) != int(expected_value):
                            match = False
                            break
                    elif key == "p":
                        if not math.isclose(float(meta_value), float(expected_value), 
                                          rel_tol=1e-12, abs_tol=1e-15):
                            match = False
                            break
                    else:  # string comparison for a_poly, b_poly, etc.
                        if str(meta_value) != str(expected_value):
                            match = False
                            break
                
                if match:
                    shots = int(row.get("shots", 0) or 0)
                    errors = int(row.get("errors", 0) or 0)
                    total_shots += shots
                    total_errors += errors
    
    except (FileNotFoundError, ValueError, TypeError):
        # File doesn't exist, invalid data, or conversion errors
        # Return zeros - these are expected scenarios
        pass
    
    return total_shots, total_errors


# Constants to replace magic numbers
DEFAULT_MAX_SHOTS = 1_000_000_000
DEFAULT_MAX_ERRORS = 100
DEFAULT_RESUME_EVERY = 50
DEFAULT_BP_ITERS = 20
DEFAULT_OSD_ORDER = 3