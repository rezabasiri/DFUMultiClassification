"""
Configuration module for path management across different platforms.
Centralizes directory paths used throughout the project.
"""

import os

def get_project_paths():
    """
    Determine the appropriate directory paths based on the environment.

    Returns:
        tuple: (directory, result_dir, root) paths
    """
    # Default to current repository structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory = project_root
    result_dir = os.path.join(project_root, 'results')
    # print(f"Using local repository structure: {directory}")

    root = os.path.join(directory, "data")

    return directory, result_dir, root


def get_data_paths(root=None):
    """
    Get paths to image folders.

    Args:
        root: Root data directory (optional, will use get_project_paths if not provided)

    Returns:
        dict: Dictionary containing paths to different image modalities
    """
    # Always get result_dir from get_project_paths
    _, result_dir, default_root = get_project_paths()

    # Use provided root or default
    if root is None:
        root = default_root

    return {
        'image_folder': os.path.join(root, "raw/Depth_RGB"),
        'depth_folder': os.path.join(root, "raw/Depth_Map_IMG"),
        'thermal_folder': os.path.join(root, "raw/Thermal_Map_IMG"),
        'thermal_rgb_folder': os.path.join(root, "raw/Thermal_RGB"),
        'csv_file': os.path.join(root, "raw/DataMaster_Processed_V12_WithMissing.csv"),
        'best_matching_csv': os.path.join(result_dir, "best_matching.csv"),
        'bb_depth_csv': os.path.join(root, "raw/bounding_box_depth.csv"),
        'bb_thermal_csv': os.path.join(root, "raw/bounding_box_thermal.csv")
    }


# Image processing parameters
IMAGE_SIZE = 128
IMAGE_SIZE_MAIN = 64  # Used in main training script

# Random seeds for reproducibility
RANDOM_SEED = 42

# Class labels
CLASS_LABELS = ['I', 'P', 'R']
