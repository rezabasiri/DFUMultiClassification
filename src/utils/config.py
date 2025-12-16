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
    # Define paths based on platform
    if os.path.exists("/Volumes/Expansion/DFUCalgary"):
        directory = "/Volumes/Expansion/DFUCalgary"  # For Mac
        result_dir = directory
    elif os.path.exists("G:/DFUCalgary"):
        directory = "G:/DFUCalgary"  # For Windows
        result_dir = directory
    elif os.path.exists("/project/6086937/basirire/multimodal"):
        directory = "/project/6086937/basirire/multimodal"  # Compute Canada
        result_dir = "/scratch/basirire/multimodal/Phase_Specefic_Calssification_With_Generative_Augmentation/results_dir"
    elif os.path.exists("C:/Users/90rez/OneDrive - University of Toronto/PhDUofT/ZivotData"):
        directory = "C:/Users/90rez/OneDrive - University of Toronto/PhDUofT/ZivotData"  # For OneDrive
        result_dir = os.path.join(directory, 'Codes/MultimodalClassification/Phase_Specefic_Calssification_With_Generative_Augmentation/results_dir')
    else:
        # Default to current repository structure
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        directory = project_root
        result_dir = os.path.join(project_root, 'results')
        print(f"Using local repository structure: {directory}")

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
    if root is None:
        _, result_dir, root = get_project_paths()

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
