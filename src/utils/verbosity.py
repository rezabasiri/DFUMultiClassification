"""
Verbosity control utilities for the DFU classification system.

Provides functions for verbosity-level-controlled output and progress tracking.
"""

import time
from datetime import timedelta
from tqdm import tqdm
from src.utils.production_config import VERBOSITY

# Global verbosity level (can be overridden at runtime)
_verbosity_level = VERBOSITY

# Progress bar instance (for VERBOSITY == 3)
_progress_bar = None
_progress_start_time = None
_progress_total = None
_progress_desc = None


def set_verbosity(level):
    """Set the global verbosity level.

    Args:
        level (int): Verbosity level (0-3)
            0 = MINIMAL: Only essential info (errors, final results)
            1 = NORMAL: Standard output (default)
            2 = DETAILED: Include debug info, intermediate metrics
            3 = PROGRESS_BAR: Settings at start, then only progress bar
    """
    global _verbosity_level
    _verbosity_level = level


def get_verbosity():
    """Get the current verbosity level."""
    return _verbosity_level


def vprint(message, level=1, end='\n', flush=False):
    """Print message if current verbosity level is >= specified level.

    Args:
        message (str): Message to print
        level (int): Minimum verbosity level required to print (default: 1)
        end (str): String appended after the message (default: newline)
        flush (bool): Whether to flush the output buffer
    """
    # In progress bar mode (level 3), suppress all prints except level 0 (errors/critical)
    if _verbosity_level == 3 and level > 0:
        return

    if _verbosity_level >= level:
        print(message, end=end, flush=flush)


def init_progress_bar(total, desc="Processing"):
    """Initialize progress bar (only active when verbosity == 3).

    Args:
        total (int): Total number of iterations
        desc (str): Description shown on progress bar
    """
    global _progress_bar, _progress_start_time, _progress_total, _progress_desc

    if _verbosity_level != 3:
        return

    _progress_total = total
    _progress_desc = desc
    _progress_start_time = time.time()

    _progress_bar = tqdm(
        total=total,
        desc=desc,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]',
        ncols=100,
        leave=True
    )


def update_progress(n=1, status=None):
    """Update progress bar (only active when verbosity == 3).

    Args:
        n (int): Increment amount (default: 1)
        status (str): Optional status message to append to description
    """
    global _progress_bar

    if _verbosity_level != 3 or _progress_bar is None:
        return

    _progress_bar.update(n)

    if status:
        elapsed = time.time() - _progress_start_time
        remaining = (_progress_total - _progress_bar.n) * (elapsed / max(_progress_bar.n, 1))
        _progress_bar.set_postfix_str(
            f"{status} | Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {timedelta(seconds=int(remaining))}"
        )


def close_progress():
    """Close progress bar (only active when verbosity == 3)."""
    global _progress_bar, _progress_start_time, _progress_total, _progress_desc

    if _verbosity_level != 3 or _progress_bar is None:
        return

    _progress_bar.close()
    _progress_bar = None
    _progress_start_time = None
    _progress_total = None
    _progress_desc = None


def get_elapsed_time():
    """Get elapsed time since progress bar was initialized.

    Returns:
        float: Elapsed time in seconds, or None if no progress bar active
    """
    if _progress_start_time is None:
        return None
    return time.time() - _progress_start_time
