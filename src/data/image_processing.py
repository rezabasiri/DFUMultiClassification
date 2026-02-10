"""
Image processing and dataset preparation utilities.
Functions for matching depth/thermal images, preprocessing, and bounding box handling.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.utils.config import get_project_paths, get_data_paths
from src.utils.verbosity import vprint
from src.utils.production_config import GENERATIVE_AUG_VERSION

# Conditionally import generative augmentation module based on version
if GENERATIVE_AUG_VERSION == 'v3':
    from src.data.generative_augmentation_v3 import augment_image, AugmentationConfig
else:
    from src.data.generative_augmentation_v2 import augment_image, AugmentationConfig

# Get paths
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)
depth_folder = data_paths['depth_folder']
thermal_folder = data_paths['thermal_folder']

# Create augmentation config for general augmentation
_augmentation_config = AugmentationConfig()

def extract_info_from_filename(filename):
    match = re.search(r'_P(\d{3})(\d{2})(\d)', filename)
    if match:
        patient_num, appt_num, wound_num = match.groups()
        return int(patient_num), int(appt_num), int(wound_num)
    return None, None, None
def find_file_match(folder, filename):
    if pd.isna(filename) or not isinstance(filename, str):
        return None
    
    #Exact match
    pattern_to_match = re.search(r'_(P\w+)', filename)
    if not pattern_to_match:
        return None
    pattern_to_match = pattern_to_match.group(1)
    matching_files = glob.glob(os.path.join(folder, f'*{pattern_to_match}*'))
    
    # Alternative match
    if not matching_files:
        pattern_to_match = re.search(r'_(P\d+)', filename)
        if not pattern_to_match:
            return None
        pattern_to_match = pattern_to_match.group(1)
        matching_files = glob.glob(os.path.join(folder, f'*{pattern_to_match}*'))

    if isinstance(matching_files, list) and len(matching_files) > 0:
        for file in matching_files:
            if not os.path.basename(file).startswith('.'):
                return file
    return None
def find_best_alternative(df, depth_filename, patient_num, appt_num, wound_num):
    """
    Find alternative thermal image through stepwise matching.
    
    Args:
        df: DataFrame with thermal image information
        depth_filename: Original depth image filename
        patient_num: Patient number
        appt_num: Appointment number
        wound_num: DFU number
        
    Returns:
        DataFrame: Matching row(s) or empty DataFrame if no match found
    """
    try:
        # Parse the original filename components
        match = re.match(r'\d+_P(\d{3})(\d{2})(\d)([AB])D([RML])([WZ])', depth_filename)
        if not match:
            print(f"Warning: Could not parse filename components for {depth_filename}")
            return pd.DataFrame()
        
        _, _, _, before_after, _, distance = match.groups()
        
        # Step 1: Most strict matching - everything except R/M/L
        pattern = f'P{patient_num:03d}{appt_num:02d}{wound_num}{before_after}T[RML]{distance}'
        matches = df[df['Filename'].str.contains(pattern, regex=True)]
        if not matches.empty:
            return matches.head(1)
        
        # Step 2: Remove distance (W/Z) requirement
        pattern = f'P{patient_num:03d}{appt_num:02d}{wound_num}{before_after}T[RML]'
        matches = df[df['Filename'].str.contains(pattern, regex=True)]
        if not matches.empty:
            return matches.head(1)
        
        # Step 3: Remove Before/After (A/B) requirement
        pattern = f'P{patient_num:03d}{appt_num:02d}{wound_num}[AB]T[RML]'
        matches = df[df['Filename'].str.contains(pattern, regex=True)]
        if not matches.empty:
            return matches.head(1)
        
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Error in find_best_alternative for {depth_filename}: {str(e)}")
        return pd.DataFrame()

def create_best_matching_dataset(depth_bb_file, thermal_bb_file, csv_file, depth_folder, thermal_folder, output_file):
    """
    Create dataset matching depth and thermal images with correct DFU correspondence.
    """
    try:
        depth_bb = pd.read_csv(depth_bb_file)
        thermal_bb = pd.read_csv(thermal_bb_file)
        metadata = pd.read_csv(csv_file)
        best_matching = []

        # Group depth bounding boxes by filename to handle multiple DFUs per image
        depth_groups = depth_bb.groupby('Filename')

        for depth_filename, depth_group in depth_groups:
            patient_num, appt_num, _ = extract_info_from_filename(depth_filename)
            if patient_num is None:
                print(f"Warning: Could not extract info from filename {depth_filename}")
                continue

            depth_map = find_file_match(depth_folder, depth_filename)
            if not depth_map:
                # print(f"Warning: No depth map found for {depth_filename}")
                continue

            thermal_filename = depth_filename.replace('D', 'T')
            
            # Process each DFU in the depth image
            for _, depth_row in depth_group.iterrows():
                current_dfu = depth_row['DFU#']
                
                # Try to find direct thermal match first
                thermal_row = thermal_bb[
                    (thermal_bb['Filename'] == thermal_filename) & 
                    (thermal_bb['DFU#'] == current_dfu)
                ]

                # If no direct match found, try alternative
                if thermal_row.empty:
                    thermal_row = find_best_alternative(
                        thermal_bb, 
                        depth_filename, 
                        patient_num, 
                        appt_num, 
                        current_dfu
                    )

                if thermal_row.empty:
                    # print(f"Warning: No thermal match found for {depth_filename} DFU#{current_dfu}")
                    continue

                thermal_map = find_file_match(thermal_folder, thermal_row['Filename'].iloc[0])
                if not thermal_map:
                    # print(f"Warning: No thermal map found for {thermal_row['Filename'].iloc[0]}")
                    continue

                metadata_row = metadata[
                    (metadata['Patient#'] == patient_num) & 
                    (metadata['Appt#'] == appt_num) & 
                    (metadata['DFU#'] == current_dfu)
                ]

                if metadata_row.empty:
                    # print(f"Warning: No metadata found for P{patient_num}A{appt_num}DFU{current_dfu}")
                    continue

                try:
                    best_matching.append({
                        'Patient#': patient_num,
                        'Appt#': appt_num,
                        'DFU#': current_dfu,
                        'depth_rgb': depth_filename,
                        'depth_map': os.path.basename(depth_map),
                        'thermal_rgb': thermal_row['Filename'].iloc[0],
                        'thermal_map': os.path.basename(thermal_map),
                        'depth_xmin': depth_row['Xmin'],
                        'depth_ymin': depth_row['Ymin'],
                        'depth_xmax': depth_row['Xmax'],
                        'depth_ymax': depth_row['Ymax'],
                        'thermal_xmin': thermal_row['Xmin'].iloc[0],
                        'thermal_ymin': thermal_row['Ymin'].iloc[0],
                        'thermal_xmax': thermal_row['Xmax'].iloc[0],
                        'thermal_ymax': thermal_row['Ymax'].iloc[0],
                        **metadata_row.iloc[0].to_dict()
                    })
                except Exception as e:
                    print(f"Error creating matching entry for {depth_filename} DFU#{current_dfu}: {str(e)}")
                    continue

        best_matching_df = pd.DataFrame(best_matching)
        best_matching_df.to_csv(output_file, index=False)
        print(f"Successfully created best matching dataset with {len(best_matching_df)} entries")
        return best_matching_df

    except Exception as e:
        print(f"Error in create_best_matching_dataset: {str(e)}")
        return pd.DataFrame()
def prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, selected_modalities, filtered_df=None):
    """
    Prepare dataset for training/evaluation.

    Args:
        depth_bb_file: Path to depth bounding box file
        thermal_bb_file: Path to thermal bounding box file
        csv_file: Path to metadata CSV file
        selected_modalities: List of modality names to include
        filtered_df: Optional pre-filtered dataframe (e.g., from outlier removal).
                    If provided, uses this instead of reading from best_matching.csv.
                    This preserves the original best_matching.csv file.

    Returns:
        DataFrame with matched data for selected modalities
    """
    if filtered_df is not None:
        # Use pre-filtered dataframe (e.g., from outlier detection)
        best_matching_df = filtered_df.copy()
        vprint(f"Using pre-filtered dataset with {len(best_matching_df)} samples", level=2)
    else:
        # Load from file (original behavior)
        best_matching_csv = os.path.join(result_dir, 'best_matching.csv')

        if not os.path.exists(best_matching_csv):
            create_best_matching_dataset(depth_bb_file, thermal_bb_file, csv_file, depth_folder, thermal_folder, best_matching_csv)

        best_matching_df = pd.read_csv(best_matching_csv)
    matched_files = {}
    
    if 'depth_rgb' in selected_modalities:
        matched_files['depth_rgb'] = best_matching_df['depth_rgb'].tolist()
        matched_files['depth_bb'] = best_matching_df[['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax']].values
    
    if 'depth_map' in selected_modalities:
        matched_files['depth_map'] = best_matching_df['depth_map'].tolist()
        if 'depth_bb' not in matched_files:
            matched_files['depth_bb'] = best_matching_df[['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax']].values
    
    if 'thermal_rgb' in selected_modalities:
        matched_files['thermal_rgb'] = best_matching_df['thermal_rgb'].tolist()
        matched_files['thermal_bb'] = best_matching_df[['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']].values
    
    if 'thermal_map' in selected_modalities:
        matched_files['thermal_map'] = best_matching_df['thermal_map'].tolist()
        if 'thermal_bb' not in matched_files:
            matched_files['thermal_bb'] = best_matching_df[['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']].values
    
    if 'metadata' in selected_modalities:
        matched_files['metadata'] = best_matching_df.drop(['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                                           'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                                           'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'], axis=1).to_dict('records')
    
    # Check that at least one modality is selected
    if not matched_files:
        raise ValueError("No valid modalities selected")
    
    # Print the number of samples for each selected modality
    vprint("Number of samples for each selected modality:", level=1)
    for key in matched_files:
        vprint(f"  {key}: {len(matched_files[key])}", level=1)
    
    return best_matching_df
def preprocess_image_data(train_data, test_data, target_class, selected_modalities):
    # Function to apply preprocessing to a single dataframe
    def preprocess_single_df(data):
        # Create an explicit copy to avoid SettingWithCopyWarning
        data = data.copy()
        data[target_class] = data[target_class].map({'I': 0, 'P': 1, 'R': 2})
        return data
    # Apply preprocessing to both train and test data
    train_data_processed = preprocess_single_df(train_data)
    test_data_processed = preprocess_single_df(test_data)

    # Prepare features and target
    image_columns = []
    if 'depth_rgb' in selected_modalities:
        image_columns.extend(['depth_rgb', 'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax'])
    if 'depth_map' in selected_modalities:
        image_columns.extend(['depth_map', 'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax'])
    if 'thermal_rgb' in selected_modalities:
        image_columns.extend(['thermal_rgb', 'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'])
    if 'thermal_map' in selected_modalities:
        image_columns.extend(['thermal_map', 'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'])

    X_train = train_data_processed[image_columns]
    y_train = train_data_processed[target_class]
    X_test = test_data_processed[image_columns]
    y_test = test_data_processed[target_class]

    # No need for imputation or scaling for image data
    X_train_image = X_train
    X_test_image = X_test

    return X_train_image, X_test_image, y_train, y_test, X_train.columns
def adjust_bounding_box(xmin, ymin, xmax, ymax):
    fov_first_camera = (87, 58)  # Horizontal FOV, Vertical FOV
    fov_second_camera = (69, 42)  # Horizontal FOV, Vertical FOV
    fov_error = 3  # Error in FOV
    left_scale = (1, 1)  # Assuming no scaling, modify if needed

    adjusted_xmin = int(xmin * (1-(fov_error/min(fov_first_camera[0], fov_second_camera[0]))))
    adjusted_ymin = int(ymin * (1-(fov_error/min(fov_first_camera[1], fov_second_camera[1]))))
    adjusted_xmax = int(xmax * (1+(fov_error/min(fov_first_camera[0], fov_second_camera[0]))))
    adjusted_ymax = int(ymax * (1+(fov_error/min(fov_first_camera[1], fov_second_camera[1]))))

    xmin = adjusted_xmin // left_scale[0]
    ymin = adjusted_ymin // left_scale[1]
    xmax = adjusted_xmax // left_scale[0]
    ymax = adjusted_ymax // left_scale[1]

    return xmin, ymin, xmax, ymax
def load_and_preprocess_image(filepath, bb_data, modality, target_size=(224, 224), augment=False):
    import time
    import random

    # Sample timing every 1% of images
    should_time = random.random() < 0.01

    if should_time:
        t_start = time.time()

    try:
        # Load image and convert to array
        img = load_img(filepath, color_mode="rgb", target_size=None)

        if should_time:
            t_after_load = time.time()
            print(f"[TIME_DEBUG] Image load (PIL): {(t_after_load - t_start)*1000:.1f}ms", flush=True)

        img_array = img_to_array(img)
        
        # Get image dimensions
        img_height, img_width = img_array.shape[:2]
        
        # Get and validate bounding box coordinates
        try:
            # Handle different types of bb_data
            if isinstance(bb_data, np.ndarray):
                if bb_data.shape == (4, 2):  # Handle case where we have duplicated coordinates
                    bb_coords = bb_data[:, 0]  # Take first set of coordinates
                else:
                    bb_coords = bb_data
            else:
                bb_coords = np.array(bb_data)
            
            # Extract coordinates
            xmin, ymin, xmax, ymax = [int(coord) for coord in bb_coords]
            
            # Apply modality-specific adjustments
            if modality == 'depth_map':
                xmin, ymin, xmax, ymax = adjust_bounding_box(xmin, ymin, xmax, ymax)
            elif modality == 'thermal_map':
                xmin, ymin, xmax, ymax = xmin-30, ymin-30, xmax+30, ymax+30
            
            # Ensure coordinates are within bounds
            xmin = max(0, min(xmin, img_width - 1))
            xmax = max(xmin + 1, min(xmax, img_width))
            ymin = max(0, min(ymin, img_height - 1))
            ymax = max(ymin + 1, min(ymax, img_height))
            
        except Exception as e:
            print(f"Error processing bounding box for {filepath}: {str(e)}")
            print(f"bb_data type: {type(bb_data)}")
            print(f"bb_data shape: {bb_data.shape if hasattr(bb_data, 'shape') else 'no shape'}")
            print(f"bb_data content: {bb_data}")
            return create_default_image(target_size)
        
        # Validate bounding box dimensions
        if xmax <= xmin or ymax <= ymin:
            print(f"Invalid bounding box dimensions for {filepath}, content: {xmin}, {ymin}, {xmax}, {ymax} or {bb_data}")
            return create_default_image(target_size)
        
        # Ensure minimum size of bounding box
        if (xmax - xmin) < 2 or (ymax - ymin) < 2:
            print(f"Bounding box too small for {filepath}")
            return create_default_image(target_size)
        
        try:
            # Crop image to bounding box
            cropped_img = img_array[ymin:ymax, xmin:xmax]
            
            # Verify cropped image dimensions
            if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                print(f"Invalid crop dimensions for {filepath}")
                return create_default_image(target_size)
            
            # Check if the cropped image is valid
            if not is_valid_bounding_box(cropped_img):
                print(f"Invalid bounding box content for {filepath}, content: {xmin}, {ymin}, {xmax}, {ymax} or {bb_data}")
                return create_default_image(target_size)
            
            # Use PIL for initial resizing to maintain aspect ratio
            cropped_pil = Image.fromarray(np.uint8(cropped_img))
            
            # Calculate new dimensions maintaining aspect ratio
            original_width, original_height = cropped_pil.size
            aspect_ratio = original_width / original_height
            
            if aspect_ratio > 1:
                new_width = target_size[1]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target_size[0]
                new_width = int(new_height * aspect_ratio)
            
            # Ensure minimum dimensions
            new_width = max(new_width, 1)
            new_height = max(new_height, 1)
            
            # Resize using PIL
            resized_pil = cropped_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with padding
            final_img = Image.new('RGB', target_size, (0, 0, 0))
            
            # Calculate padding
            left = (target_size[1] - new_width) // 2
            top = (target_size[0] - new_height) // 2
            
            # Paste resized image onto padded background
            final_img.paste(resized_pil, (left, top))
            
            # Convert to numpy array
            final_array = np.array(final_img)
            
            try:
                img_tensor = tf.convert_to_tensor(final_array, dtype=tf.float32)
            except Exception as e:
                print(f"Error tensor conversion failed for {filepath}: {str(e)} and tensor size {img_tensor.shape[:2]}")
                return create_default_image(target_size)
            try:
                # Note: Augmentation is now applied ONLY at batch time in create_enhanced_augmentation_fn()
                # to avoid double augmentation and to properly handle generative images.
                # The 'augment' parameter is kept for API compatibility but no longer used here.

                if modality in ['depth_rgb', 'thermal_rgb']:
                    # Normalize image
                    img_tensor = img_tensor / 255.
                elif modality in ['depth_map', 'thermal_map']:
                    # Normalize MAP
                    img_tensor = img_tensor / tf.reduce_max(img_tensor) if tf.reduce_max(img_tensor) != 0 else 255.
                
                img_tensor = tf.reshape(img_tensor, (*target_size, 3))

                if should_time:
                    t_end = time.time()
                    print(f"[TIME_DEBUG] Image preprocess total: {(t_end - t_start)*1000:.1f}ms", flush=True)

                return img_tensor
            except Exception as e:
                print(f"Error processing augmentations for {filepath}: {str(e)} and tensor size {img_tensor.shape[:2]}")
                return create_default_image(target_size)
            
        except Exception as e:
            print(f"Error processing crop for {filepath}: {str(e)}")
            return create_default_image(target_size)
            
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        return create_default_image(target_size)

def create_default_image(target_size):
    """Create a default black image of the target size"""
    return tf.zeros((*target_size, 3), dtype=tf.float32)

def is_valid_bounding_box(img_array, threshold=0.95):
    """Check if the bounding box content is valid"""
    if img_array.ndim == 3:
        is_black = np.all(np.mean(img_array < 10, axis=(0,1)) > threshold)
        is_white = np.all(np.mean(img_array > 245, axis=(0,1)) > threshold)
    else:
        is_black = np.mean(img_array < 10) > threshold
        is_white = np.mean(img_array > 245) > threshold
    
    return not (is_black or is_white)
