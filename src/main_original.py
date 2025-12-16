#%% Import Libraries and Define Paths    
import os
import glob
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate, Concatenate, GlobalAveragePooling2D, Multiply, Layer, BatchNormalization, Dropout, Lambda, GlobalAveragePooling1D, Flatten, Add, Attention, LayerNormalization, Reshape, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib
from sklearn.model_selection import KFold, StratifiedKFold
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
import shap
import random
import logging
import pickle
from sklearn.model_selection import GroupKFold
import seaborn as sns
import itertools
import csv
from PIL import Image
from src.data.generative_augmentation_v2 import (
    GenerativeAugmentationManager,
    GenerativeAugmentationCallback,
    create_enhanced_augmentation_fn,
    AugmentationConfig
)
from src.evaluation.metrics import filter_frequent_misclassifications, track_misclassifications
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Add at the beginning of your script
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

logging.basicConfig(level=logging.INFO)
# import warnings
# warnings.filterwarnings('ignore', message='The calling iterator did not fully read the dataset being cached.*')
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(["GPU:0"])
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Define paths using centralized config
from src.utils.config import get_project_paths, get_data_paths

directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

os.environ["OMP_NUM_THREADS"] = "2"
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

ck_path = os.path.join(result_dir, "checkpoints")
os.makedirs(ck_path, exist_ok=True)

# Get data folder paths from config
image_folder = data_paths['image_folder']
depth_folder = data_paths['depth_folder']
thermal_folder = data_paths['thermal_folder']
thermal_rgb_folder = data_paths['thermal_rgb_folder']
csv_file = data_paths['csv_file']
depth_bb_file = data_paths['bb_depth_csv']
thermal_bb_file = data_paths['bb_thermal_csv']

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load bounding box data
depth_bb = pd.read_csv(depth_bb_file)
thermal_bb = pd.read_csv(thermal_bb_file)

# Image processing parameters
image_size = 64 #128
global_batch_size = 30
batch_size = global_batch_size // strategy.num_replicas_in_sync
n_epochs = 1000
CLASS_LABELS = ['I', 'P', 'R']
#%% debugging utilities
import gc
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
def clear_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.backend.clear_session()
            gc.collect()
        except RuntimeError as e:
            print('clear_gpu_memory Warning:', e)
def reset_keras():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
def clear_cuda_memory():
    if 'cuda' in tf.test.gpu_device_name():
        tf.keras.backend.clear_session()
        gc.collect()
#%% Data Processing
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
def prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, selected_modalities):
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
    print("Number of samples for each selected modality:")
    for key in matched_files:
        print(f"  {key}: {len(matched_files[key])}")
    
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

    try:
        # Load image and convert to array
        img = load_img(filepath, color_mode="rgb", target_size=None)
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
                # Apply augmentations if requested
                if augment:
                    img_tensor = augment_image(img_tensor, modality, 
                                            tf.random.uniform([], maxval=1000000, dtype=tf.int32))
                
                if modality in ['depth_rgb', 'thermal_rgb']:
                    # Normalize image
                    img_tensor = img_tensor / 255.
                elif modality in ['depth_map', 'thermal_map']:
                    # Normalize MAP
                    img_tensor = img_tensor / tf.reduce_max(img_tensor) if tf.reduce_max(img_tensor) != 0 else 255.
                
                img_tensor = tf.reshape(img_tensor, (*target_size, 3))
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
#%% Cached Dataset Creation
def create_cached_dataset(best_matching_df, selected_modalities, batch_size, 
                         is_training=True, cache_dir=None, augmentation_fn=None):
    """
    Create a cached TF dataset optimized for training/validation with support for generative augmentation.
    
    Args:
        best_matching_df: DataFrame with matching data
        selected_modalities: List of selected modalities
        batch_size: Batch size
        is_training: Whether this is for training
        cache_dir: Directory for caching
        augmentation_fn: Custom augmentation function (including generative augmentations)
    """
    def process_single_sample(filename, bb_coords, modality_name):
        """Process a single image sample using py_function"""
        def _process_image(filename_tensor, bb_coords_tensor, modality_tensor):
            try:
                # Convert tensors to numpy/python types
                filename_str = filename_tensor.numpy().decode('utf-8')
                bb_coords_float = bb_coords_tensor.numpy()
                modality_str = modality_tensor.numpy().decode('utf-8')
                
                base_folders = {
                    'depth_rgb': image_folder,
                    'depth_map': depth_folder,
                    'thermal_rgb': thermal_rgb_folder,
                    'thermal_map': thermal_folder
                }
                
                img_path = os.path.join(base_folders[modality_str], filename_str)
                img_tensor = load_and_preprocess_image(
                    img_path, 
                    bb_coords_float,
                    modality_str,
                    target_size=(image_size, image_size),
                    augment=False
                )
                
                # Convert TensorFlow tensor to numpy array
                if isinstance(img_tensor, tf.Tensor):
                    img_array = img_tensor.numpy()
                else:
                    img_array = np.array(img_tensor)
                
                return img_array
            
            except Exception as e:
                print(f"Error in _process_image: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                return np.zeros((image_size, image_size, 3), dtype=np.float32)

        processed_image = tf.py_function(
            _process_image,
            [filename, bb_coords, modality_name],
            tf.float32
        )
        # Set the shape that was lost during py_function
        processed_image.set_shape((image_size, image_size, 3))
        return processed_image

    def load_and_preprocess_single_sample(row):
        features = {}
        # Handle image modalities
        for modality in [m for m in selected_modalities if m != 'metadata']:
            # Extract and validate bounding box coordinates
            if modality in ['depth_rgb', 'depth_map']:
                # Get all depth-related coordinates
                bb_coords = []
                for prefix in ['depth_']:
                    bb_coords.extend([
                        float(row[f'{prefix}xmin']),
                        float(row[f'{prefix}ymin']),
                        float(row[f'{prefix}xmax']),
                        float(row[f'{prefix}ymax'])
                    ])
                
                bb_coords = tf.stack(bb_coords)
            else:  # thermal modalities
                bb_coords = tf.stack([
                    float(row['thermal_xmin']),
                    float(row['thermal_ymin']),
                    float(row['thermal_xmax']),
                    float(row['thermal_ymax'])
                ])
            
            # Convert modality to tensor
            modality_tensor = tf.convert_to_tensor(modality, dtype=tf.string)
            
            # Process image
            img_tensor = process_single_sample(
                row[modality], 
                bb_coords,
                modality_tensor
            )
            
            features[f'{modality}_input'] = img_tensor

        # Handle metadata if selected
        if 'metadata' in selected_modalities:
            metadata_features = tf.stack([
                tf.cast(row['rf_prob_I'], tf.float32),
                tf.cast(row['rf_prob_P'], tf.float32),
                tf.cast(row['rf_prob_R'], tf.float32)
            ])
            features['metadata_input'] = metadata_features
        
        # Add sample identifiers for visualization only (not used in training)
        features['sample_id'] = tf.stack([
            tf.cast(row['Patient#'], tf.float32),
            tf.cast(row['Appt#'], tf.float32),
            tf.cast(row['DFU#'], tf.float32)
        ])
        
        # Extract label
        label = tf.cast(row['Healing Phase Abs'], tf.int32)
        label = tf.one_hot(label, depth=3)
        
        return features, label

    def df_to_dataset(dataframe):
        # Make a copy to avoid modifying the original
        df = dataframe.copy()
        tensor_slices = {}
        for col in df.columns:
            # Convert to appropriate numpy dtype
            if col in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                tensor_slices[col] = df[col].astype(str).values
                print(f"{col} type: {type(tensor_slices[col])} shape: {tensor_slices[col].shape}")
            elif col in ['Healing Phase Abs']:
                tensor_slices[col] = df[col].astype(np.int32).values
                print(f"{col} type: {type(tensor_slices[col])} shape: {tensor_slices[col].shape}")
            else:
                tensor_slices[col] = df[col].astype(np.float32).values
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)
        return dataset

    # Initialize dataset from DataFrame
    dataset = df_to_dataset(best_matching_df)
    
    # Apply preprocessing to each sample
    dataset = dataset.map(
        load_and_preprocess_single_sample,
        num_parallel_calls=tf.data.AUTOTUNE
        # num_parallel_calls=4
    )
    
    # Calculate how many samples we need
    n_samples = len(best_matching_df)
    steps = np.ceil(n_samples / batch_size)
    k = int(steps * batch_size)  # Total number of samples needed
    
    # Cache the dataset
    cache_filename = 'tf_cache_train' if is_training else 'tf_cache_valid'
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        dataset = dataset.cache(os.path.join(cache_dir, cache_filename))
    else:
        dataset = dataset.cache(cache_filename)
    
    with tf.device('/CPU:0'):
        pre_aug_dataset = dataset
        pre_aug_dataset = pre_aug_dataset.batch(batch_size)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000 if len(best_matching_df) > 1000 else len(best_matching_df), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Apply augmentation after batching
    if is_training:
        if augmentation_fn:
            # Use provided augmentation function (includes generative augmentations)
            dataset = dataset.map(
                augmentation_fn,
                num_parallel_calls=tf.data.AUTOTUNE,
                # num_parallel_calls=4
                )
        # else:                                         #TODO: Add back default augmentations
        #     # Fall back to regular augmentation
        #     dataset = dataset.map(
        #         create_augmentation_fn(prob=0.25),
        #         num_parallel_calls=tf.data.AUTOTUNE
        #     )

    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    # dataset = dataset.prefetch(2)
    return dataset, pre_aug_dataset, steps
# First, add this helper function near the start of your prepare_cached_datasets function
def check_split_validity(train_data, valid_data, max_ratio_diff=0.3, verbose=False):
    """
    Check if both splits contain all classes and have reasonable class distributions.
    
    Args:
        train_data: Training dataset
        valid_data: Validation dataset
        max_ratio_diff: Maximum allowed difference in class ratios between splits
        verbose: Whether to print detailed distribution information
        
    Returns:
        bool: True if split is valid, False otherwise
    """
    # Check if all classes present in both splits
    train_classes = set(train_data['Healing Phase Abs'].unique())
    valid_classes = set(valid_data['Healing Phase Abs'].unique())
    
    if len(train_classes) != 3 or len(valid_classes) != 3:
        if verbose:
            print(f"Missing classes - Train: {train_classes}, Valid: {valid_classes}")
        return False
    
    # Compare class distributions
    train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
    valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)
    
    if verbose:
        print("\nClass distributions:")
        print("Training:", dict(train_dist.round(3)))
        print("Validation:", dict(valid_dist.round(3)))
    
    # Check if class ratios are reasonably similar
    max_diff = 0
    for cls in [0, 1, 2]:
        train_ratio = train_dist.get(cls, 0)
        valid_ratio = valid_dist.get(cls, 0)
        diff = abs(train_ratio - valid_ratio)
        max_diff = max(max_diff, diff)
        if diff > max_ratio_diff:
            if verbose:
                print(f"Class {cls} ratio difference too high: {diff:.3f}")
            return False
    
    if verbose:
        print(f"Maximum ratio difference: {max_diff:.3f}")
    return True

def prepare_cached_datasets(data1, selected_modalities, train_patient_percentage=0.8,
                          batch_size=32, cache_dir=None, gen_manager=None, aug_config=None, run=0):
    """
    Prepare cached datasets with proper metadata handling based on selected modalities.
    """
    random.seed(42 + run * (run + 3))
    tf.random.set_seed(42 + run * (run + 3))
    np.random.seed(42 + run * (run + 3))
    os.environ['PYTHONHASHSEED'] = f"{42 + run * (run + 3)}"
    # Create a deep copy of the data
    data = data1.copy(deep=True)
    data = data.reset_index(drop=True)
    
    # Then in your splitting code:
    max_retries = 2000
    best_split_diff = float('inf')
    best_split = None

    for attempt in range(max_retries):
        # Shuffle and split patients
        patient_numbers = sorted(data['Patient#'].unique())
        n_train_patients = int(len(patient_numbers) * train_patient_percentage)
        np.random.shuffle(patient_numbers)
        train_patients = patient_numbers[:n_train_patients]
        valid_patients = patient_numbers[n_train_patients:]
        
        # Convert labels
        if 'Healing Phase Abs' in data.columns:
            data['Healing Phase Abs'] = data['Healing Phase Abs'].astype(str)
            data['Healing Phase Abs'] = data['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})
        
        # Split data
        train_data = data[data['Patient#'].isin(train_patients)]
        valid_data = data[data['Patient#'].isin(valid_patients)]
        
        # Calculate maximum distribution difference for this split
        train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
        valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)
        max_diff = max(abs(train_dist.get(cls, 0) - valid_dist.get(cls, 0)) for cls in [0, 1, 2])
        
        # Keep track of best split even if not perfect
        if max_diff < best_split_diff and len(set(train_data['Healing Phase Abs'].unique())) == 3 and len(set(valid_data['Healing Phase Abs'].unique())) == 3:
            best_split_diff = max_diff
            best_split = (train_data.copy(), valid_data.copy())
        
        # Check if split is valid
        if check_split_validity(train_data, valid_data, max_ratio_diff=0.05):
            print(f"Found valid split after {attempt + 1} attempts")
            print("\nFinal class distributions:")
            # Create ordered distributions
            ordered_train = {i: train_dist[i] if i in train_dist else 0 for i in [0, 1, 2]}
            ordered_valid = {i: valid_dist[i] if i in valid_dist else 0 for i in [0, 1, 2]}
            print("Training:", {k: round(v, 3) for k, v in ordered_train.items()})
            print("Validation:", {k: round(v, 3) for k, v in ordered_valid.items()})
            break
            
        if attempt == max_retries - 1:
            print(f"Warning: Could not find optimal split after {max_retries} attempts.")
            print(f"Using best found split with max difference of {best_split_diff:.3f}")
            if best_split is not None:
                train_data, valid_data = best_split
                train_dist = train_data['Healing Phase Abs'].value_counts(normalize=True)
                valid_dist = valid_data['Healing Phase Abs'].value_counts(normalize=True)
                
                # Create ordered distributions
                ordered_train = {i: train_dist[i] if i in train_dist else 0 for i in [0, 1, 2]}
                ordered_valid = {i: valid_dist[i] if i in valid_dist else 0 for i in [0, 1, 2]}
                
                print("\nBest found class distributions:")
                print("Training:", {k: round(v, 3) for k, v in ordered_train.items()})
                print("Validation:", {k: round(v, 3) for k, v in ordered_valid.items()})
                
            else:
                print("No valid split found with all classes present in both sets")
                raise ValueError("Could not create valid data split")
    
    # Determine columns to keep based on selected modalities
    if 'metadata' in selected_modalities:
        columns_to_keep = ['Healing Phase Abs']
    else:
        columns_to_keep = ['Patient#', 'Appt#', 'DFU#','Healing Phase Abs']
    
    # Add image and bounding box columns only once
    added_depth_bb = False
    added_thermal_bb = False
    
    for modality in selected_modalities:
        if modality == 'metadata':
            # Keep all metadata columns excluding image and bounding box columns
            metadata_columns = [col for col in data.columns if col not in [
                'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'
            ]]
            columns_to_keep.extend(metadata_columns)
        else:
            # Add image filename
            columns_to_keep.append(modality)
            
            # Add bounding box coordinates only once per type
            if modality in ['depth_rgb', 'depth_map'] and not added_depth_bb:
                columns_to_keep.extend(['depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax'])
                added_depth_bb = True
            elif modality in ['thermal_rgb', 'thermal_map'] and not added_thermal_bb:
                columns_to_keep.extend(['thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'])
                added_thermal_bb = True

    # Keep only necessary columns
    train_data = train_data[columns_to_keep].copy()
    valid_data = valid_data[columns_to_keep].copy()
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    
    # Apply sampling to training data only
    def apply_mixed_sampling_to_df(df, apply_sampling=True, mix=False):
        """
        Apply both random over and undersampling to balance classes with explicit ordering
        """
        alpha_unique_cases = df[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates().copy()
        X_alpha = df.drop('Healing Phase Abs', axis=1)
        y_alpha = df['Healing Phase Abs']
        X = df.drop('Healing Phase Abs', axis=1)
        y = df['Healing Phase Abs']
        resampled_df = df.copy()
        
        # Print original distribution with ordered classes
        counts = Counter(y_alpha)
        print("Original class distribution (ordered):")
        for class_idx in [0, 1, 2]:  # Explicit ordering
            print(f"Class {class_idx}: {counts[class_idx]}")
        # Calculate alpha values from original distribution
        total_samples = sum(counts.values())
        class_frequencies = {cls: count/total_samples for cls, count in counts.items()}
        median_freq = np.median(list(class_frequencies.values()))
        alpha_values = [median_freq/class_frequencies[i] for i in [0, 1, 2]]  # Keep ordered
        alpha_sum = sum(alpha_values)
        alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]

        print("\nCalculated alpha values from original distribution:")
        print(f"Alpha values (ordered) [I, P, R]: {[round(a, 3) for a in alpha_values]}")
        if not apply_sampling:
            # Return same format as sampling case
            resampled_df = df.copy()
            resampled_df = resampled_df.reset_index(drop=True)
            return resampled_df, alpha_values
        
        count_items = [(count, index) for index, count in counts.items()]
        count_items.sort()
        try:
            if not mix:
                OverSampleOnly
                
            del resampled_df
            # Calculate intermediate targets maintaining class order
            intermediate_target = {
                count_items[1][1]: count_items[1][0],          # Keep I class (0) as is
                count_items[2][1]: count_items[1][0],          # Reduce P class (1) to match I class
                count_items[0][1]: counts[2]           # Keep R class (2) as is
            }
            
            undersampler = RandomUnderSampler(
                sampling_strategy=intermediate_target,
                random_state=42 + run * (run + 3)
            )
            X_under, y_under = undersampler.fit_resample(X, y)
            
            # Print intermediate results with ordered classes
            under_counts = Counter(y_under)
            print("\nAfter undersampling (ordered):")
            for class_idx in [0, 1, 2]:
                print(f"Class {class_idx}: {under_counts[class_idx]}")
            
            # Set final targets
            final_target = {
                count_items[1][1]: count_items[1][0],          # Keep I class as is
                count_items[2][1]: count_items[1][0],          # Keep P class as reduced
                count_items[0][1]: count_items[1][0]           # Boost R class to match
            }
            
            oversampler = RandomOverSampler(
                sampling_strategy=final_target,
                random_state=42 + run * (run + 3)
            )
            X_resampled, y_resampled = oversampler.fit_resample(X_under, y_under)
            
            # Print final results with ordered classes
            final_counts = Counter(y_resampled)
            print("\nAfter oversampling (ordered):")
            for class_idx in [0, 1, 2]:
                print(f"Class {class_idx}: {final_counts[class_idx]}")
            
            # Reconstruct DataFrame
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df['Healing Phase Abs'] = y_resampled
            resampled_df = resampled_df.reset_index(drop=True)
        
            # Verify final order of classes
            print("\nVerifying final class distribution...")
            final_verify = Counter(resampled_df['Healing Phase Abs'])
            assert all(final_verify[i] == count_items[1][0] for i in [0, 1, 2]), "Classes not properly balanced"
            
            return resampled_df, alpha_values
            
        except Exception as e:
            print(f"Error in mixed sampling: {str(e)}")
            print("Falling back to simple random oversampling...")
            # Print original distribution with ordered classes
            counts = Counter(y_alpha)
            print("Original class distribution (ordered):")
            for class_idx in [0, 1, 2]:  # Explicit ordering
                print(f"Class {class_idx}: {counts[class_idx]}")
            # Calculate alpha values from original distribution
            total_samples = sum(counts.values())
            class_frequencies = {cls: count/total_samples for cls, count in counts.items()}
            median_freq = np.median(list(class_frequencies.values()))
            alpha_values = [median_freq/class_frequencies[i] for i in [0, 1, 2]]  # Keep ordered
            alpha_sum = sum(alpha_values)
            alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]
            
            oversampler = RandomOverSampler(random_state=42 + run * (run + 3))
            X_resampled, y_resampled = oversampler.fit_resample(X, y)
            
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df['Healing Phase Abs'] = y_resampled
            resampled_df = resampled_df.reset_index(drop=True)
            
            # Print final results with ordered classes
            final_counts = Counter(y_resampled)
            print("\nAfter oversampling (ordered):")
            for class_idx in [0, 1, 2]:
                print(f"Class {class_idx}: {final_counts[class_idx]}")
            
            return resampled_df, alpha_values
    if 'metadata' in selected_modalities:
        # Calculate class weights for Random Forest models
        unique_cases = train_data[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates().copy()
        print(f"\nUnique cases: {len(unique_cases)} (before oversampling)")
        
        # Create binary labels on unique cases
        unique_cases['label_bin1'] = (unique_cases['Healing Phase Abs'] > 0).astype(int)
        unique_cases['label_bin2'] = (unique_cases['Healing Phase Abs'] > 1).astype(int)
        
        # Print true class distributions
        print("\nTrue binary label distributions (unique cases):")
        print("Binary1:", unique_cases['label_bin1'].value_counts())
        print("Binary2:", unique_cases['label_bin2'].value_counts())
        
        # Calculate weights using only unique cases
        class_weights_binary1 = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=unique_cases['label_bin1']
        )
        class_weight_dict_binary1 = dict(zip([0, 1], class_weights_binary1))
        
        class_weights_binary2 = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=unique_cases['label_bin2']
        )
        class_weight_dict_binary2 = dict(zip([0, 1], class_weights_binary2))
        
        # print("\nClass weights (based on unique cases):")
        # print("Binary1:", class_weight_dict_binary1)
        # print("Binary2:", class_weight_dict_binary2)   
    else:
        class_weight_dict_binary1=None
        class_weight_dict_binary2=None
    
    train_data, alpha_values = apply_mixed_sampling_to_df(train_data, apply_sampling=False, mix=False)
    
    def preprocess_split(split_data, is_training=True, class_weight_dict_binary1=None, 
                            class_weight_dict_binary2=None, rf_model1=None, rf_model2=None, imputation_data=None):
            """Preprocess data with proper handling of metadata and image columns"""
            # Create a copy of the data
            split_data = split_data.copy()
            if imputation_data is not None:
                source_data = imputation_data.copy()
            else:
                source_data = split_data.copy()

            # Only process metadata if it's in selected modalities
            if 'metadata' in selected_modalities:
                # Identify columns for metadata processing
                image_related_columns = [
                    'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                    'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                    'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax'
                ]
                
                # Create a copy with only metadata columns
                metadata_df = split_data.drop(columns=[col for col in image_related_columns if col in split_data.columns])
                source_df = source_data.drop(columns=[col for col in image_related_columns if col in source_data.columns])
                
                # Feature engineering on metadata columns only
                metadata_df['BMI'] = metadata_df['Weight (Kg)'] / ((metadata_df['Height (cm)'] / 100) ** 2)
                source_df['BMI'] = source_df['Weight (Kg)'] / ((source_df['Height (cm)'] / 100) ** 2)
                metadata_df['Age above 60'] = (metadata_df['Age'] > 60).astype(int)
                source_df['Age above 60'] = (source_df['Age'] > 60).astype(int)
                metadata_df['Age Bin'] = pd.cut(metadata_df['Age'], bins=range(0, int(metadata_df['Age'].max()) + 20, 20), right=False, labels=range(len(range(0, int(metadata_df['Age'].max()) + 20, 20)) - 1))
                source_df['Age Bin'] = pd.cut(source_df['Age'], bins=range(0, int(source_df['Age'].max()) + 20, 20), right=False, labels=range(len(range(0, int(source_df['Age'].max()) + 20, 20)) - 1))
                metadata_df['Weight Bin'] = pd.cut(metadata_df['Weight (Kg)'], bins=range(0, int(metadata_df['Weight (Kg)'].max()) + 20, 20), right=False, labels=range(len(range(0, int(metadata_df['Weight (Kg)'].max()) + 20, 20)) - 1))
                source_df['Weight Bin'] = pd.cut(source_df['Weight (Kg)'], bins=range(0, int(source_df['Weight (Kg)'].max()) + 20, 20), right=False, labels=range(len(range(0, int(source_df['Weight (Kg)'].max()) + 20, 20)) - 1))
                metadata_df['Height Bin'] = pd.cut(metadata_df['Height (cm)'], bins=range(0, int(metadata_df['Height (cm)'].max()) + 10, 10), right=False, labels=range(len(range(0, int(metadata_df['Height (cm)'].max()) + 10, 10)) - 1))
                source_df['Height Bin'] = pd.cut(source_df['Height (cm)'], bins=range(0, int(source_df['Height (cm)'].max()) + 10, 10), right=False, labels=range(len(range(0, int(source_df['Height (cm)'].max()) + 10, 10)) - 1))
                
                # Handle categorical columns
                categorical_columns = ['Sex (F:0, M:1)', 'Side (Left:0, Right:1)', 'Foot Aspect', 'Odor', 'Type of Pain Grouped']
                for col in categorical_columns:
                    if col in metadata_df.columns:
                        metadata_df[col] = pd.Categorical(metadata_df[col]).codes
                        source_df[col] = pd.Categorical(source_df[col]).codes
                
                # Other categorical mappings
                categorical_mappings = {
                    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': {'ankle': 4, 'Heel': 3, 'middle': 2, 'toes': 1, 'Hallux': 0},
                    'Dressing Grouped': {'NoDressing': 0, 'BandAid': 1, 'BasicDressing': 1, 'AbsorbantDressing': 2, 'Antiseptic': 3, 'AdvanceMethod': 4, 'other': 4},
                    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': {'Serous': 0, 'Haemoserous': 1, 'Bloody': 2, 'Thick': 3}
                }
                
                for col, mapping in categorical_mappings.items():
                    if col in metadata_df.columns:
                        metadata_df[col] = metadata_df[col].map(mapping)
                        source_df[col] = source_df[col].map(mapping)
                # Remove unnecessary columns
                features_to_drop = [
                    'ID', 'Location', 'Healing Phase', 'Phase Confidence (%)', 'DFU#', 'Appt#',
                    'Appt Days', 'Type of Pain2', 'Type of Pain_Grouped2', 'Type of Pain', 
                    'Peri-Ulcer Temperature (°C)', 'Wound Centre Temperature (°C)', 'Dressing',
                    'Dressing Grouped', "No Offloading", "Offloading: Therapeutic Footwear", 
                    "Offloading: Scotcast Boot or RCW", "Offloading: Half Shoes or Sandals", 
                    "Offloading: Total Contact Cast", "Offloading: Crutches, Walkers or Wheelchairs", 
                    "Offloading Score"
                ]
                integer_columns = [
                    "Sex (F:0, M:1)", "Smoking","Alcohol Consumption", "Habits Score", "Type of Diabetes", "Heart Conditions", 
                    "Cancer History", "Sensory Peripheral", "Clinical Score", "Number of DFUs","Side (Left:0, Right:1)", "Foot Aspect", "Location", 
                    "Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)","No Toes Deformities", "Bunion", "Claw", "Hammer", "Charcot Arthropathy", 
                    "Flat (Pes Planus) Arch", "Abnormally High Arch","No Arch Deformities", "Foot Score", 
                    "Pain Level", "Type of Pain", "Type of Pain Grouped", "Type of Pain2","Type of Pain_Grouped2", "Wound Tunneling", 
                    "Exudate Amount (None:0,Minor,Medium,Severe:3)","Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)", "Odor", 
                    "No Peri-ulcer Conditions (False:0, True:1)", "Erythema at Peri-ulcer","Edema at Peri-ulcer", "Pale Colour at Peri-ulcer", "Maceration at Peri-ulcer", 
                    "Wound Score", "Dressing","Dressing Grouped", 
                    "No Foot Abnormalities", "Foot Hair Loss", "Foot Dry Skin","Foot Fissure Cracks", "Foot Callus", "Thickened Toenail", 
                    "Foot Fungal Nails", "Leg Score", "No Offloading","Offloading: Therapeutic Footwear", "Offloading: Scotcast Boot or RCW", 
                    "Offloading: Half Shoes or Sandals", "Offloading: Total Contact Cast","Offloading: Crutches, Walkers or Wheelchairs", "Offloading Score", "Healing Phase Abs", "Healing Phase Abs Regression", 'Age Bin', 'Age above 60', 'Weight Bin', 'Height Bin'
                ]
                metadata_df = metadata_df.drop(columns=[col for col in features_to_drop if col in metadata_df.columns])
                source_df = source_df.drop(columns=[col for col in features_to_drop if col in source_df.columns])
                # Impute missing values
                columns_to_impute = [col for col in metadata_df.columns if col not in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                                                                        'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                                                                        'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']+['Healing Phase Abs']]
                
                # numeric_columns = split_data.select_dtypes(include=[np.number]).columns
                imputer = KNNImputer(n_neighbors=5)
                source_df[columns_to_impute] = imputer.fit_transform(source_df[columns_to_impute])
                metadata_df[columns_to_impute] = imputer.transform(metadata_df[columns_to_impute])
                
                for column in integer_columns:
                    if column in metadata_df.columns:
                        metadata_df[column] = metadata_df[column].astype(int)
                        source_df[column] = source_df[column].astype(int)
                    else:
                        continue
                
                # Scale features
                scaler = StandardScaler()
                source_df[columns_to_impute] = scaler.fit_transform(source_df[columns_to_impute])
                metadata_df[columns_to_impute] = scaler.transform(metadata_df[columns_to_impute])
                
                # Random Forest processing
                if is_training:
                    try:
                        import tensorflow_decision_forests as tfdf
                        print("Using TensorFlow Decision Forests")
                        
                        # Create models
                        rf_model1 = tfdf.keras.RandomForestModel(
                            num_trees=300,
                            task=tfdf.keras.Task.CLASSIFICATION,
                            random_seed=42 + run * (run + 3),
                            verbose=0
                        )
                        rf_model2 = tfdf.keras.RandomForestModel(
                            num_trees=300,
                            task=tfdf.keras.Task.CLASSIFICATION,
                            random_seed=42 + run * (run + 3),
                            verbose=0
                        )
                        
                        # Prepare features for RF
                        train_df = metadata_df.copy()
                        
                        # Create binary labels and verify their values
                        train_df['label_bin1'] = (train_df['Healing Phase Abs'] > 0).astype(int)
                        train_df['label_bin2'] = (train_df['Healing Phase Abs'] > 1).astype(int)
                        
                        # # Print value counts to verify
                        # print("\nLabel binary 1 distribution:", train_df['label_bin1'].value_counts())
                        # print("Label binary 2 distribution:", train_df['label_bin2'].value_counts())
                        # print("\nClass weights 1:", class_weight_dict_binary1)
                        # print("Class weights 2:", class_weight_dict_binary2)
                        
                        # Add weights with explicit mapping
                        train_df['weight1'] = train_df['label_bin1'].apply(lambda x: class_weight_dict_binary1[x])
                        train_df['weight2'] = train_df['label_bin2'].apply(lambda x: class_weight_dict_binary2[x])
                        
                        # # Print weight distribution to verify
                        # print("\nWeight1 unique values:", train_df['weight1'].unique())
                        # print("Weight2 unique values:", train_df['weight2'].unique())
                        
                        # Remove unnecessary columns
                        cols_to_drop = ['Patient#', 'Healing Phase Abs']
                        
                        # Create datasets
                        dataset1 = tfdf.keras.pd_dataframe_to_tf_dataset(
                            train_df.drop(columns=cols_to_drop + ['label_bin2', 'weight2']),
                            label='label_bin1',
                            weight='weight1'
                        )
                        
                        dataset2 = tfdf.keras.pd_dataframe_to_tf_dataset(
                            train_df.drop(columns=cols_to_drop + ['label_bin1', 'weight1']),
                            label='label_bin2',
                            weight='weight2'
                        )
                        
                        # Train models
                        rf_model1.fit(dataset1)
                        rf_model2.fit(dataset2)
                    except ImportError:
                        print("Using Scikit-learn RandomForestClassifier")
                        from sklearn.ensemble import RandomForestClassifier
                        rf_model1 = RandomForestClassifier(
                            n_estimators=300,
                            random_state=42 + run * (run + 3),
                            class_weight=class_weight_dict_binary1,
                            n_jobs=-1,
                            # max_features=None
                        )
                        rf_model2 = RandomForestClassifier(
                            n_estimators=300, #100,
                            random_state=42 + run * (run + 3),
                            class_weight=class_weight_dict_binary2,
                            n_jobs=-1,
                            # max_features=None
                        )
                        # Prepare features for RF
                        X = metadata_df.drop(['Patient#', 'Healing Phase Abs'], axis=1)
                        # y = split_data['Healing Phase Abs'].map({'I': 0, 'P': 1, 'R': 2})
                        y = metadata_df['Healing Phase Abs']
                        y_bin1 = (y > 0).astype(int)
                        y_bin2 = (y > 1).astype(int)
                        # Train RF models
                        rf_model1.fit(X, y_bin1)
                        rf_model2.fit(X, y_bin2)    
                try:
                    import tensorflow_decision_forests as tfdf
                    dataset1 = tfdf.keras.pd_dataframe_to_tf_dataset(
                        metadata_df.drop(['Patient#', 'Healing Phase Abs'], axis=1),
                        label=None  # No label needed for prediction
                    )
                    
                    # Get predictions
                    with tf.device('/CPU:0'):
                        pred1 = rf_model1.predict(dataset1)
                        pred2 = rf_model2.predict(dataset1)
                    # with tf.device('/CPU:0'):
                    #     prob1 = rf_predict_function(rf_model1, dataset1)
                    #     prob2 = rf_predict_function(rf_model2, dataset1)
                        # # Get probabilities for positive class (class 1)
                        prob1 = np.squeeze(pred1)
                        prob2 = np.squeeze(pred2)
                except ImportError:
                    dataset = metadata_df.drop(['Patient#', 'Healing Phase Abs'], axis=1)
                    # dataset_pd = tf_to_pd(dataset)
                    prob1 = rf_model1.predict_proba(dataset)[:, 1]
                    prob2 = rf_model2.predict_proba(dataset)[:, 1]
                
                # Calculate final probabilities
                prob_I = 1 - prob1
                prob_P = prob1 * (1 - prob2)
                prob_R = prob2
                
                # Store RF probabilities in the DataFrame
                split_data['rf_prob_I'] = prob_I
                split_data['rf_prob_P'] = prob_P
                split_data['rf_prob_R'] = prob_R
            
            metadata_columns = [col for col in split_data.columns if col not in [
                'Healing Phase Abs',
                'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax',
                'rf_prob_I', 'rf_prob_P', 'rf_prob_R',
                'Patient#', 'Appt#', 'DFU#'
            ]]
            split_data = split_data.drop(columns=metadata_columns)
            
            return split_data, rf_model1, rf_model2

    # Preprocess both splits
    source_data = train_data.copy()
    del train_data
    train_data, rf_model1, rf_model2 = preprocess_split(source_data, is_training=True, class_weight_dict_binary1=class_weight_dict_binary1, class_weight_dict_binary2=class_weight_dict_binary2)
    valid_data, _, _ = preprocess_split(valid_data, is_training=False, rf_model1=rf_model1, rf_model2=rf_model2, imputation_data=source_data)
    del rf_model1, rf_model2
    
    # Create cached datasets
    train_dataset, pre_aug_dataset, steps_per_epoch = create_cached_dataset(
        train_data,
        selected_modalities,
        batch_size,
        is_training=True,
        cache_dir=result_dir,
        augmentation_fn=create_enhanced_augmentation_fn(gen_manager, aug_config) if gen_manager else None)

    valid_dataset, _, validation_steps = create_cached_dataset(
        valid_data,
        selected_modalities,
        batch_size,
        is_training=False,
        cache_dir=result_dir,
        augmentation_fn=None)
    
    return train_dataset, pre_aug_dataset, valid_dataset, steps_per_epoch, validation_steps, alpha_values
class BatchVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, modalities, freq=5, max_samples=5, run=1, save_dir='batch_visualizations'):
        """
        Args:
            dataset: TensorFlow dataset to visualize
            modalities: List of modality names
            freq: Frequency of epochs to create visualizations
            max_samples: Maximum number of samples to visualize
            save_dir: Directory to save visualizations
        """
        super(BatchVisualizationCallback, self).__init__()
        self.dataset = dataset
        self.modalities = modalities
        self.freq = freq
        self.max_samples = max_samples
        self.save_dir = save_dir
        self.run = run
        os.makedirs(save_dir, exist_ok=True)
        # Create modality abbreviation mapping
        self.modality_abbrev = {
            'metadata': 'md',
            'depth_rgb': 'dr',
            'depth_map': 'dm',
            'thermal_rgb': 'tr',
            'thermal_map': 'tm'
        }
    def _create_short_filename(self, epoch):
        """Create a shorter filename using modality abbreviations"""
        # Get abbreviated modalities string
        mod_str = '_'.join(self.modality_abbrev[m] for m in sorted(self.modalities))
        return f'e{epoch}_{mod_str}_{self.run}.png'
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            try:
                num_modalities = len(self.modalities)
                fig, axes = plt.subplots(self.max_samples, num_modalities, 
                                    figsize=(6*num_modalities, 5*self.max_samples))
                
                if self.max_samples == 1:
                    axes = axes.reshape(1, -1)
                elif num_modalities == 1:
                    axes = axes.reshape(-1, 1)
                
                label_map = {0: 'I', 1: 'P', 2: 'R'}
                
                # Get one batch from the dataset
                for batch in self.dataset.take(1):
                    batch_inputs, batch_labels = batch
                    
                    # Convert batch_labels to numpy if it's a tensor
                    if isinstance(batch_labels, tf.Tensor):
                        batch_labels = batch_labels.numpy()
                    
                    # Adjust max_samples if batch is smaller
                    actual_batch_size = batch_labels.shape[0]
                    num_samples = min(self.max_samples, actual_batch_size)
                    
                    for i in range(num_samples):
                        for j, modality in enumerate(self.modalities):
                            ax = axes[i, j]
                            input_key = f'{modality}_input'
                            
                            if input_key not in batch_inputs:
                                ax.text(0.5, 0.5, f"Missing modality: {modality}", 
                                    ha='center', va='center')
                                continue
                            
                            # Get the input data for this modality
                            modality_data = batch_inputs[input_key][i]
                            if isinstance(modality_data, tf.Tensor):
                                modality_data = modality_data.numpy()
                            
                            if modality == 'metadata':
                                # Plot RF probabilities
                                ax.clear()  # Clear previous plot
                                ax.bar(['I', 'P', 'R'], modality_data)
                                ax.set_ylim(0, 1)
                                ax.set_title('RF Probabilities')
                            else:
                                # Handle image data
                                ax.clear()  # Clear previous plot
                                if modality_data.ndim == 3 and modality_data.shape[2] == 3:
                                    if modality_data.max() <= 1.0:
                                        modality_data = (modality_data * 255).astype(np.uint8)
                                    ax.imshow(modality_data)
                                elif modality_data.ndim == 2 or (modality_data.ndim == 3 and modality_data.shape[2] == 1):
                                    ax.imshow(modality_data.squeeze(), cmap='gray')
                                else:
                                    ax.text(0.1, 0.5, f"Unsupported shape: {modality_data.shape}", 
                                        wrap=True, fontsize=8)
                                
                                ax.axis('off')
                            
                            if i == 0:
                                ax.set_title(modality)
                        
                        # Add label information
                        label_index = np.argmax(batch_labels[i])
                        label = label_map[label_index]
                        # Get model predictions
                        pred_batch = self.model.predict(batch_inputs, verbose=0)
                        pred_index = np.argmax(pred_batch[i])
                        pred_label = label_map[pred_index]
                        
                        # Get patient info from the passed identifiers
                        patient_info = ""
                        try:
                            ids = batch_inputs['sample_id'][i].numpy()
                            patient_info = f"\nP{int(ids[0]):03d}A{int(ids[1]):02d}D{int(ids[2]):d}"
                        except Exception as e:
                            print(f"Error getting patient info: {str(e)}")
                        
                        axes[i, 0].text(-0.1, -0.15, 
                                    f"Sample {i}\nTrue: {label}\nPred: {pred_label}{patient_info}", 
                                    transform=axes[i, 0].transAxes, 
                                    fontsize=8, 
                                    verticalalignment='top')
                    
                    break  # Only process one batch
                
                plt.suptitle(f'Epoch {epoch + 1}')
                plt.tight_layout()
                save_path = os.path.join(self.save_dir,  self._create_short_filename(epoch + 1))
                plt.savefig(save_path, bbox_inches='tight', dpi=400)
                plt.close(fig)
                
                print(f"\nBatch visualization saved for epoch {epoch + 1}")
            except Exception as e:
                print(f"Error during batch visualization: {str(e)}")
def visualize_dataset(train_dataset, selected_modalities, save_dir, 
                     max_samples_per_page=5, dataset_portion=100, 
                     dpi=400, max_batches=None, total_samples=None):
    """
    Visualize samples from a training dataset with multiple modalities.
    
    Args:
        train_dataset: TensorFlow dataset to visualize
        selected_modalities: List of modality names
        save_dir: Directory to save visualizations
        max_samples_per_page: Maximum number of samples per visualization page
        dataset_portion: Percentage of dataset to visualize (1-100)
        dpi: DPI for saved images
        max_batches: Optional maximum number of batches to process
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create modality abbreviation mapping
    modality_abbrev = {
        'metadata': 'md',
        'depth_rgb': 'dr',
        'depth_map': 'dm',
        'thermal_rgb': 'tr',
        'thermal_map': 'tm'
    }
    
    # Create shorter filename
    def create_filename(page_num):
        mod_str = '_'.join(modality_abbrev[m] for m in sorted(selected_modalities))
        return f'dataset_p{page_num}_{mod_str}.png'
    
    label_map = {0: 'I', 1: 'P', 2: 'R'}
    num_modalities = len(selected_modalities)
    processed_samples = 0
    page_num = 1
    
    # Initialize variables
    current_page_samples = []
    batch_count = 0
    
    # Get one batch to estimate total samples
    for first_batch in train_dataset.take(1):
        batch_size = first_batch[1].shape[0]
        break
    
    print(f"Processing dataset (batch size: {batch_size})")
    print(f"Will create pages with up to {max_samples_per_page} samples each")
    
    # Process the dataset
    for batch in train_dataset:
        if max_batches and batch_count >= max_batches:
            break
            
        batch_inputs, batch_labels = batch
        batch_size = batch_labels.shape[0]
        
        # Convert batch_labels to numpy if needed
        if isinstance(batch_labels, tf.Tensor):
            batch_labels = batch_labels.numpy()
        
        # Process each sample in the batch
        for i in range(batch_size):
            if total_samples and processed_samples >= total_samples:
                break
            # Skip samples based on dataset_portion
            if dataset_portion < 100:
                if np.random.rand() > dataset_portion / 100:
                    continue
            
            # Store sample data
            sample_data = {
                'inputs': {},
                'label': batch_labels[i],
                'sample_id': batch_inputs['sample_id'][i].numpy()
            }
            
            for modality in selected_modalities:
                input_key = f'{modality}_input'
                if input_key in batch_inputs:
                    modality_data = batch_inputs[input_key][i]
                    if isinstance(modality_data, tf.Tensor):
                        modality_data = modality_data.numpy()
                    sample_data['inputs'][modality] = modality_data
            
            current_page_samples.append(sample_data)
            processed_samples += 1

        
            # Create visualization when page is full
            if len(current_page_samples) >= max_samples_per_page:
                # Create figure
                fig, axes = plt.subplots(len(current_page_samples), num_modalities,
                                       figsize=(6*num_modalities, 5*len(current_page_samples)))
                
                # Handle single row/column cases
                if len(current_page_samples) == 1:
                    axes = axes.reshape(1, -1)
                elif num_modalities == 1:
                    axes = axes.reshape(-1, 1)
                
                # Plot each sample
                for row, sample in enumerate(current_page_samples):
                    for col, modality in enumerate(selected_modalities):
                        ax = axes[row, col]
                        
                        if modality not in sample['inputs']:
                            ax.text(0.5, 0.5, f"Missing modality: {modality}",
                                  ha='center', va='center')
                            continue
                        
                        modality_data = sample['inputs'][modality]
                        
                        if modality == 'metadata':
                            ax.clear()
                            ax.bar(['I', 'P', 'R'], modality_data)
                            ax.set_ylim(0, 1)
                            ax.set_title('RF Probabilities')
                        else:
                            ax.clear()
                            if modality_data.ndim == 3 and modality_data.shape[2] == 3:
                                if modality_data.max() <= 1.0:
                                    modality_data = (modality_data * 255).astype(np.uint8)
                                ax.imshow(modality_data)
                            elif modality_data.ndim == 2 or (modality_data.ndim == 3 and modality_data.shape[2] == 1):
                                ax.imshow(modality_data.squeeze(), cmap='gray')
                            else:
                                ax.text(0.1, 0.5, f"Unsupported shape: {modality_data.shape}",
                                      wrap=True, fontsize=8)
                            ax.axis('off')
                        
                        if row == 0:
                            ax.set_title(modality)
                    
                    # Add sample information
                    label_index = np.argmax(sample['label'])
                    label = label_map[label_index]
                    ids = sample['sample_id']
                    patient_info = f"P{int(ids[0]):03d}A{int(ids[1]):02d}D{int(ids[2]):d}"
                    
                    axes[row, 0].text(-0.1, -0.15,
                                    f"Sample {processed_samples-len(current_page_samples)+row+1}\n"
                                    f"Label: {label}\n{patient_info}",
                                    transform=axes[row, 0].transAxes,
                                    fontsize=8,
                                    verticalalignment='top')
                
                # Save figure
                plt.suptitle(f'Dataset Visualization - Page {page_num}')
                plt.tight_layout()
                save_path = os.path.join(save_dir, create_filename(page_num))
                plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
                plt.close(fig)
                
                print(f"Saved page {page_num} ({processed_samples} samples)")
                
                # Reset for next page
                current_page_samples = []
                page_num += 1
                
        if total_samples and processed_samples >= total_samples:
            break
        batch_count += 1
    
    # Save any remaining samples
    if current_page_samples:
        # Create figure for remaining samples
        fig, axes = plt.subplots(len(current_page_samples), num_modalities,
                               figsize=(6*num_modalities, 5*len(current_page_samples)))
        
        # Handle single row/column cases
        if len(current_page_samples) == 1:
            axes = axes.reshape(1, -1)
        elif num_modalities == 1:
            axes = axes.reshape(-1, 1)
                
        for row, sample in enumerate(current_page_samples):
            for col, modality in enumerate(selected_modalities):
                ax = axes[row, col]
                
                if modality not in sample['inputs']:
                    ax.text(0.5, 0.5, f"Missing modality: {modality}",
                          ha='center', va='center')
                    continue
                
                modality_data = sample['inputs'][modality]
                
                if modality == 'metadata':
                    ax.clear()
                    ax.bar(['I', 'P', 'R'], modality_data)
                    ax.set_ylim(0, 1)
                    ax.set_title('RF Probabilities')
                else:
                    ax.clear()
                    if modality_data.ndim == 3 and modality_data.shape[2] == 3:
                        if modality_data.max() <= 1.0:
                            modality_data = (modality_data * 255).astype(np.uint8)
                        ax.imshow(modality_data)
                    elif modality_data.ndim == 2 or (modality_data.ndim == 3 and modality_data.shape[2] == 1):
                        ax.imshow(modality_data.squeeze(), cmap='gray')
                    else:
                        ax.text(0.1, 0.5, f"Unsupported shape: {modality_data.shape}",
                              wrap=True, fontsize=8)
                    ax.axis('off')
                
                if row == 0:
                    ax.set_title(modality)
            
            # Add sample information
            label_index = np.argmax(sample['label'])
            label = label_map[label_index]
            ids = sample['sample_id']
            patient_info = f"P{int(ids[0]):03d}A{int(ids[1]):02d}D{int(ids[2]):d}"
            
            axes[row, 0].text(-0.1, -0.15,
                            f"Sample {processed_samples-len(current_page_samples)+row+1}\n"
                            f"Label: {label}\n{patient_info}",
                            transform=axes[row, 0].transAxes,
                            fontsize=8,
                            verticalalignment='top')
        
        # Save final figure
        plt.suptitle(f'Dataset Visualization - Page {page_num} (Final)')
        plt.tight_layout()
        save_path = os.path.join(save_dir, create_filename(page_num))
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        print(f"Saved final page {page_num} ({processed_samples} total samples)")
    
    print(f"\nVisualization complete! Created {page_num} pages with {processed_samples} samples")
    return processed_samples
class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir='training_plots', update_freq=1):
        """
        Args:
            save_dir: Directory to save the plots
            update_freq: Frequency of epochs to update plots
        """
        super(TrainingHistoryCallback, self).__init__()
        self.save_dir = save_dir
        self.update_freq = update_freq
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize history storage
        self.history = {
            'loss': [], 'accuracy': [], 'weighted_f1_score': [],
            'val_loss': [], 'val_accuracy': [], 'val_weighted_f1_score': []
        }

    def on_epoch_end(self, epoch, logs=None):
        try:
            # Update history
            for metric in self.history.keys():
                if metric in logs:
                    self.history[metric].append(logs[metric])
            
            if (epoch + 1) % self.update_freq == 0:
                metrics = ['loss', 'accuracy', 'weighted_f1_score']
                
                for metric in metrics:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot training metrics
                    if len(self.history[metric]) > 0:
                        plt.plot(self.history[metric], label=f'Train', alpha=0.7)
                    if len(self.history[f'val_{metric}']) > 0:
                        plt.plot(self.history[f'val_{metric}'], label=f'Validation', 
                                linestyle='--', alpha=0.7)
                    
                    plt.title(f'Model {metric.replace("_", " ").title()} (Epoch {epoch + 1})')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric.replace('_', ' ').title())
                    plt.legend(loc='best')
                    plt.grid(True, alpha=0.3)
                    
                    save_path = os.path.join(self.save_dir, f'{metric}_history.png')
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close()
        except Exception as e:
            print(f"Error during training history plot: {str(e)}")
            
def plot_net_confusion_matrix(y_true, y_pred, save_dir='evaluation_plots'):
    """
    Plot confusion matrix and print classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_dir: Directory to save the confusion matrix plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    present_classes = np.unique(y_true)
    present_class_names = [CLASS_LABELS[i] for i in present_classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_class_names, 
                yticklabels=present_class_names)
    plt.title('Net Confusion Matrix Across All Runs')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    save_path = os.path.join(save_dir, 'net_confusion_matrix.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print("\nNet Confusion Matrix Results:")
    print(classification_report(y_true, y_pred, 
                              target_names=present_class_names, 
                              labels=present_classes))
    print(f"Net Cohen's Kappa: {cohen_kappa_score(y_true, y_pred, weights='quadratic'):.4f}")

#%% Model Generators
def create_image_branch(input_shape, modality):
    print("\nDebug create_image_branch")
    
    image_input = Input(shape=(image_size, image_size, 3), name=f'{modality}_input')
    if modality in ['depth_rgb', 'thermal_rgb']:
    # # if True:
    #     if os.path.exists(os.path.join(directory, "local_weights/efficientnetb3_notop.h5")):
    #         base_model = tf.keras.applications.EfficientNetB3(weights=None, include_top=False, input_tensor=image_input, pooling='avg', drop_connect_rate=0.1, classes=500)
    #         base_model.load_weights(os.path.join(directory, "local_weights/efficientnetb3_notop.h5"))
    #     else: 
    #         base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=image_input, pooling='avg', drop_connect_rate=0.1, classes=500)
    #     base_model.trainable = True
    #     print("\nRGB Branch Configuration:")
    #     print(f"Input shape: {input_shape}")
    #     print(f"EfficientNetB3 trainable status: {base_model.trainable}")
    #     print(f"Number of trainable weights for EfficentNetB3: {len(base_model.trainable_weights)}")
    #     for layer in base_model.layers:
    #         layer._name = f'{modality}_{layer.name}'
    #     x = base_model.output
        
        x = Conv2D(256, (3, 3), activation='relu',  kernel_initializer='he_normal', name=f'{modality}_Conv2D_0')(image_input)
        x = Conv2D(128, (3, 3), activation='relu',  kernel_initializer='he_normal', name=f'{modality}_Conv2D_1')(x)
        x = Conv2D(64, (3, 3), activation='relu',  kernel_initializer='he_normal', name=f'{modality}_Conv2D_2')(x)
        x = Conv2D(32, (3, 3), activation='relu',  kernel_initializer='he_normal', name=f'{modality}_Conv2D_3')(x)
    else:
        # if os.path.exists(os.path.join(directory, "local_weights/efficientnetb0_notop.h5")):
        #     base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_tensor=image_input, pooling='avg', drop_connect_rate=0.1, classes=100)
        #     base_model.load_weights(os.path.join(directory, "local_weights/efficientnetb0_notop.h5"))
        # else: 
        #     base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_tensor=image_input, pooling='avg', drop_connect_rate=0.1, classes=100)
        # base_model.trainable = True
        # print("\nSpatial Branch Configuration:")
        # print(f"Input shape: {input_shape}")
        # print(f"EfficientNetB0 trainable status: {base_model.trainable}")
        # print(f"Number of trainable weights for EfficentNetB0: {len(base_model.trainable_weights)}")
        # for layer in base_model.layers:
        #     layer._name = f'{modality}_{layer.name}'
        # x = base_model.output
        
        x = Conv2D(128, (3, 3), activation='relu',  kernel_initializer='he_normal', name=f'{modality}_Conv2D_1')(image_input)
        x = Conv2D(64, (3, 3), activation='relu',  kernel_initializer='he_normal', name=f'{modality}_Conv2D_2')(x)
        x = Conv2D(32, (3, 3), activation='relu',  kernel_initializer='he_normal', name=f'{modality}_Conv2D_3')(x)
        
    # x = Conv2D(128, (3, 3), activation='relu', name=f'{modality}_Conv2D_1')(image_input)
    # x = Conv2D(64, (3, 3), activation='relu', name=f'{modality}_Conv2D_2')(x)
    x = GlobalAveragePooling2D(name=f'{modality}_GAP2D')(x)
    # x = Dense(64, activation='relu', name=f'{modality}_projection3')(x)
        
    # # Projection layer to ensure consistent dimensionality across modalities
    x = Dense(512, activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.001),  kernel_initializer='he_normal', name=f'{modality}_projection512')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj512')(x)
    x = tf.keras.layers.Dropout(0.15, name=f'{modality}_dropout_512')(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),  kernel_initializer='he_normal', name=f'{modality}_projection1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj1')(x)
    x = tf.keras.layers.Dropout(0.10, name=f'{modality}_dropout_1')(x)
    x = Dense(128, activation='relu',  kernel_initializer='he_normal', name=f'{modality}_projection2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj2')(x)
    # x = tf.keras.layers.Dropout(0.10, name=f'{modality}_dropout_2')(x)
    x = Dense(64, activation='relu',  kernel_initializer='he_normal', name=f'{modality}_projection3')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{modality}_BN_proj3')(x)
    # x = tf.keras.layers.Dropout(0.10, name=f'{modality}_DP_proj3')(x)
    
    # Apply modular attention
    modular_attention = OptimizedModularAttention(name=f'{modality}_modular_attention')
    attention_output = modular_attention(x)
    
    
    return image_input, attention_output
    # return image_input, x

class ConfidenceBasedMetadataAttention(Layer):
    """Attention mechanism that scales based on metadata confidence"""
    def __init__(self, min_scale=1.5, max_scale=3.0, **kwargs):
        super(ConfidenceBasedMetadataAttention, self).__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def build(self, input_shape):
        # Calculate feature dimension (total dim minus RF probs)
        self.feature_dim = input_shape[-1] - 3
        
        # Learnable attention weights for feature dimension
        self.attention_weights = self.add_weight(
            name='metadata_attention',
            shape=(self.feature_dim,),  # Only for features, not RF probs
            initializer='ones',
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            dtype='float32'
        )
        super(ConfidenceBasedMetadataAttention, self).build(input_shape)
    
    def compute_confidence_score(self, probabilities):
        """Calculate confidence score from RF probabilities"""
        max_prob = tf.reduce_max(probabilities, axis=-1, keepdims=True)
        entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-10), axis=-1, keepdims=True)
        normalized_entropy = entropy / tf.math.log(3.0)
        confidence = max_prob * (1 - normalized_entropy)
        scaled_confidence = self.min_scale + (self.max_scale - self.min_scale) * confidence
        return scaled_confidence
        
    def call(self, inputs):
        # Split inputs into RF probs and features
        rf_probs = inputs[:, :3]  # First 3 values are RF probabilities
        features = inputs[:, 3:]   # Rest are processed features
        
        # Compute confidence-based scaling
        confidence_scale = self.compute_confidence_score(rf_probs)
        
        # Apply learnable attention to features only
        normalized_weights = tf.nn.softmax(self.attention_weights)
        scaled_features = features * tf.expand_dims(normalized_weights, 0)
        
        # Apply confidence scaling to attended features
        return scaled_features * confidence_scale
    
def create_metadata_branch(input_shape, index):
    """Metadata branch with basic processing"""
    metadata_input = Input(shape=input_shape, name=f'metadata_input')
    
    # Add minimal processing
    x = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32), name=f'metadata_cast_{index}')(metadata_input)
    x = BatchNormalization(name=f'metadata_BN_{index}')(x)
    # x = tf.repeat(x, repeats=20, axis=1)
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
    # # Simple Attention Mechanism directly on the 3 inputs
    # attention_weights = Dense(3, activation='softmax', name=f'metadata_attention')(x) # output (None, 3)
    # attended_metadata = Multiply(name=f'metadata_weighted_{index}')([x, attention_weights]) # (None, 3)

    return metadata_input, x

class OptimizedModularAttention(Layer):
    """Optimized version of ModularAttention"""
    def __init__(self, **kwargs):
        super(OptimizedModularAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name='modality_attention',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            dtype='float32'
        )
        super(OptimizedModularAttention, self).build(input_shape)
        
    def call(self, inputs):
        normalized_weights = tf.nn.softmax(self.attention_weights)
        return inputs * tf.expand_dims(normalized_weights, 0)
def create_fusion_layer(branches, num_branches):
    """Updated fusion layer with improved metadata scaling and skip connections"""
    if num_branches > 1:
        attention_outputs = []
        has_metadata = any('metadata' in branch.name for branch in branches)
        metadata_branch = None

        if has_metadata:
            metadata_branch = next((branch for branch in branches if 'metadata' in branch.name), None)

        for i, branch in enumerate(branches):
            # Apply skip connection before attention mechanism
            original_branch = branch  # Store the original branch for skip connection

            query = Dense(64, activation='tanh', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001), name=f'fusion_query_{i}')(branch)

            attended_branches = []

            for j, other_branch in enumerate(branches):
                if i != j:
                    interaction = tf.multiply(query, other_branch)
                    score = Dense(1, activation='sigmoid', name=f'fusion_score_{i}_{j}')(interaction)

                    # Improved metadata scaling based on confidence
                    if has_metadata and metadata_branch is not None:
                        if branch is metadata_branch:
                            # Metadata querying other modalities, scale down attention
                            weight = tf.nn.sigmoid(score * 0.8)  # Reduced scaling factor
                        elif other_branch is metadata_branch:
                            # Other modalities querying metadata, scale up attention
                            weight = tf.nn.sigmoid(score * 1.5)  # Increased scaling factor
                        else:
                            weight = tf.nn.sigmoid(score)
                    else:
                        weight = tf.nn.sigmoid(score)

                    attended = tf.multiply(other_branch, weight)
                    attended_branches.append(attended)

            if attended_branches:
                attended_sum = tf.add_n(attended_branches)
                attention_outputs.append(tf.add(original_branch, attended_sum))  # Add the original branch for skip connection
            else:
                attention_outputs.append(branch)

        return concatenate(attention_outputs, name='modal_fusion')
    else:
        return branches[0]
# Add visualization callback
class MetadataConfidenceCallback(tf.keras.callbacks.Callback):
    """Callback to monitor metadata confidence and influence when metadata is present"""
    def __init__(self, selected_modalities, log_dir='metadata_confidence_logs'):
        super(MetadataConfidenceCallback, self).__init__()
        self.has_metadata = 'metadata' in selected_modalities
        if self.has_metadata:
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.confidence_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        if not self.has_metadata:
            return  # Skip if no metadata
            
        metadata_layer = next(
            (layer for layer in self.model.layers 
             if isinstance(layer, ConfidenceBasedMetadataAttention)),
            None
        )
        
        if metadata_layer and hasattr(metadata_layer, 'last_confidence'):
            avg_confidence = np.mean(metadata_layer.last_confidence)
            self.confidence_history.append(avg_confidence)
            
            if (epoch + 1) % 10 == 0:  # Plot every 10 epochs
                plt.figure(figsize=(10, 5))
                plt.plot(self.confidence_history)
                plt.title('Metadata Confidence History')
                plt.xlabel('Epoch')
                plt.ylabel('Average Confidence Score')
                plt.savefig(os.path.join(self.log_dir, f'metadata_confidence_epoch_{epoch+1}.png'))
                plt.close()
def create_multimodal_model(input_shapes, selected_modalities, class_weights):
    with strategy.scope():
        inputs = {}
        branches = []
        # Process each modality
        for i, modality in enumerate(selected_modalities):
            if modality == 'metadata':
                metadata_input, branch_output = create_metadata_branch(input_shapes[modality], i)
                inputs[f'metadata_input'] = metadata_input
                # branch_output = tf.keras.layers.Lambda(lambda x: tf.cast(x[:, :3], tf.float32), name=f'metadata_branch_{i}')(branch_output)
                # Add a dense layer to project metadata to desired dimension (e.g., 64)
                # branch_output = Dense(64, activation='relu', name=f'{modality}_projection')(branch_output)
                branches.append(branch_output)
            elif modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
                image_input, branch_output = create_image_branch(input_shapes[modality], f'{modality}')
                inputs[f'{modality}_input'] = image_input
                # branch_output = tf.keras.layers.Dropout(0.10, name=f'{modality}_fdropout_{i}')(branch_output)
                branches.append(branch_output)
    
        # Add sample_id input but don't connect it to the model
        inputs['sample_id'] = Input(shape=(3,), name='sample_id')

        if len(selected_modalities) == 1:
            output = Dense(3, activation='softmax', name='output')(branches[0])
        elif len(selected_modalities) == 2:
            merged = concatenate(branches, name='concat_branches')
            output = Dense(3, activation='softmax', name='output')(merged)
        elif len(selected_modalities) == 3:
            merged = concatenate(branches, name='concat_branches')
            x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(merged)
            x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
            x = tf.keras.layers.Dropout(0.10, name='final_dropout_4')(x)
            output = Dense(3, activation='softmax', name='output')(x)
        elif len(selected_modalities) == 4:
            merged = concatenate(branches, name='concat_branches')
            x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_3')(merged)
            x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
            x = tf.keras.layers.Dropout(0.10, name='final_dropout_3')(x)
            x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(x)
            x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
            x = tf.keras.layers.Dropout(0.10, name='final_dropout_4')(x)
            output = Dense(3, activation='softmax', name='output')(x)
        elif len(selected_modalities) == 5:
            merged = concatenate(branches, name='concat_branches')
            x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_2')(merged)
            x = tf.keras.layers.BatchNormalization(name='final_BN_2')(x)
            x = tf.keras.layers.Dropout(0.25, name='final_dropout_2')(x)
            x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_3')(x)
            x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
            x = tf.keras.layers.Dropout(0.10, name='final_dropout_3')(x)
            x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(x)
            x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
            x = tf.keras.layers.Dropout(0.10, name='final_dropout_4')(x)
            output = Dense(3, activation='softmax', name='output')(x)

        # else:
            
        #     image_tensor = tf.stack(branches, axis=1)
        #     averaged_branches = tf.reduce_mean(image_tensor, axis=1)
            
        #     # Apply cross-modal fusion
        #     # fused = create_fusion_layer(branches, len(branches))
        #     # merged = concatenate(averaged_branches, name='concat_branches')
            
        #     # Final classification layers
        #     # x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_1')(fused)
        #     # x = tf.keras.layers.BatchNormalization(name='final_BN_1')(x)
        #     # x = tf.keras.layers.Dropout(0.10, name='final_dropout_1')(x)
        #     # x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_2')(x)
        #     # x = tf.keras.layers.BatchNormalization(name='final_BN_2')(x)
        #     # x = tf.keras.layers.Dropout(0.25, name='final_dropout_2')(x)
        #     # x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_3')(x)
        #     # x = tf.keras.layers.BatchNormalization(name='final_BN_3')(x)
        #     # x = tf.keras.layers.Dropout(0.10, name='final_dropout_3')(x)
        #     # x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='final_dense_4')(fused)
        #     # x = tf.keras.layers.BatchNormalization(name='final_BN_4')(x)
        #     # x = tf.keras.layers.Dropout(0.15, name='final_dropout_4')(x)
        #     output = Dense(3, activation='softmax', name='output')(averaged_branches)

        model = Model(inputs=inputs, outputs=output)

        return model 
#%% Loss Functions
from tensorflow.keras import backend as K
def weighted_f1_score(y_true, y_pred):
    y_true_cls = tf.argmax(y_true, axis=1)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    
    conf_matrix = tf.math.confusion_matrix(y_true_cls, y_pred_cls, num_classes=3)
    conf_matrix = tf.cast(conf_matrix, tf.float32)
    true_positives = tf.linalg.diag_part(conf_matrix)
    predicted_positives = tf.reduce_sum(conf_matrix, axis=0)
    actual_positives = tf.reduce_sum(conf_matrix, axis=1)
    precision = true_positives / (predicted_positives + 1e-7)
    recall = true_positives / (actual_positives + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    weights = actual_positives / tf.reduce_sum(actual_positives)
    weighted_f1 = tf.reduce_sum(f1 * weights)
    
    return weighted_f1
def weighted_ordinal_crossentropy(y_true, y_pred, num_classes=3, ordinal_weight=0.5):
    # Standard categorical crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # Ordinal penalty
    true_class = tf.argmax(y_true, axis=-1)
    pred_class = tf.argmax(y_pred, axis=-1)
    ordinal_penalty = tf.square(tf.cast(true_class, tf.float32) - tf.cast(pred_class, tf.float32))
    # Combine losses
    total_loss = cce + ordinal_weight * ordinal_penalty
    return total_loss

# Wrapper function to set hyperparameters
def get_weighted_ordinal_crossentropy(num_classes=3, ordinal_weight=0.5):
    def loss(y_true, y_pred):
        return tf.cast(weighted_ordinal_crossentropy(y_true, y_pred, num_classes, ordinal_weight), tf.float32)
    return loss
def get_weighted_ordinal_crossentropyF1(num_classes=3, ordinal_weight=0.5, f1_weight=0.1):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Weighted ordinal cross-entropy
        wocce = weighted_ordinal_crossentropy(y_true, y_pred, num_classes, ordinal_weight)
        # Weighted F1 score
        wf1score = weighted_f1_score(y_true, y_pred)
        # Combine losses
        total_loss = wocce - f1_weight * tf.math.log(wf1score + 1e-7)
        # Ensure the loss is finite
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.zeros_like(total_loss))
        
        return total_loss
    return loss
def focal_ordinal_loss(y_true, y_pred, num_classes=3, ordinal_weight=0.5, gamma=3.0, alpha=[0.598, 0.315, 1.597]):
    # Clip prediction values to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Focal loss
    cross_entropy = -y_true * tf.math.log(y_pred)
    focal_weight = alpha * tf.math.pow(1 - y_pred, gamma)
    focal_loss = focal_weight * cross_entropy
    
    # Sum over classes
    focal_loss = tf.reduce_sum(focal_loss, axis=-1)
    
    # Ordinal penalty
    true_class = tf.argmax(y_true, axis=-1)
    pred_class = tf.argmax(y_pred, axis=-1)
    ordinal_penalty = tf.square(tf.cast(true_class, tf.float32) - tf.cast(pred_class, tf.float32))
    
    # Combine losses
    total_loss = focal_loss + ordinal_weight * ordinal_penalty
    return total_loss
# Wrapper function to set hyperparameters
def get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.5, gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        return focal_ordinal_loss(y_true, y_pred, num_classes, ordinal_weight, gamma, alpha)
    return loss
# def get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.5, gamma=2.0, alpha=[1,1,1]):
#     return lambda y_true, y_pred: focal_ordinal_loss(y_true, y_pred, num_classes, ordinal_weight, gamma, alpha)
#%% Cross Validation
class EpochMemoryCallback(tf.keras.callbacks.Callback):
    """Memory management callback that's compatible with distribution strategy"""
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        # tf.keras.backend.clear_session()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, _ in enumerate(gpus):
                tf.config.experimental.reset_memory_stats(f'GPU:{i}')
class NaNMonitorCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor for NaN values in validation metrics
    and trigger training restart if detected.
    """
    def __init__(self):
        super(NaNMonitorCallback, self).__init__()
        self.nan_detected = False
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'val_weighted_f1_score' in logs:
            if np.isnan(logs['val_weighted_f1_score']):
                print("\nNaN detected in validation weighted F1 score. Triggering training restart...")
                self.nan_detected = True
                self.model.stop_training = True
def clean_up_training_resources():
    """Helper function to clean up resources before restarting training"""
    # Clean GPU memory
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Try to clear GPU memory stats
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, gpu in enumerate(gpus):
                tf.config.experimental.reset_memory_stats(f'GPU:{i}')
    except Exception as e:
        print(f"Error clearing memory stats: {str(e)}")
    
    # Remove cache files
    cache_patterns = [
        os.path.join(result_dir, 'tf_cache_train*'),
        os.path.join(result_dir, 'tf_cache_valid*'),
        'tf_cache_train*',
        'tf_cache_valid*'
    ]
    
    for pattern in cache_patterns:
        try:
            cache_files = glob.glob(pattern)
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                except Exception as e:
                    print(f"Warning: Could not remove cache file {cache_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error while processing pattern {pattern}: {str(e)}")

def create_checkpoint_filename(selected_modalities, run=1, config_name=0):
    modality_str = '_'.join(sorted(selected_modalities))
    checkpoint_name = f'{modality_str}_{run}_{config_name}.h5'
    return os.path.join(result_dir, checkpoint_name)
class EscapingReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=5, 
                 escape_factor=5.0, min_lr=1e-6, escape_patience=2):
        """
        Modified ReduceLROnPlateau that temporarily increases LR to escape plateaus.
        
        Args:
            monitor: Quantity to monitor
            factor: Factor by which to reduce LR when plateauing
            patience: Number of epochs with no improvement before reducing LR
            escape_factor: Factor by which to increase LR when trying to escape
            min_lr: Minimum learning rate
            escape_patience: Number of plateau detections before trying to escape
        """
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.escape_factor = escape_factor
        self.min_lr = min_lr
        self.escape_patience = escape_patience
        
        # Internal state
        self.best = float('inf')
        self.wait = 0
        self.plateau_count = 0
        self.is_escaping = False
        self.escape_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        # Get and store current learning rate at the start of epoch
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        # Make sure LR is included in logs
        logs['lr'] = current_lr
        
        # If we're in escape mode, check if it helped
        if self.is_escaping:
            if current < self.best:
                # Escape was successful, reset state
                print(f'\nEscape successful! New best {self.monitor}: {current:.6f}')
                self.best = current
                self.wait = 0
                self.plateau_count = 0
                self.is_escaping = False
            elif epoch - self.escape_epoch >= 2:  # Give escape 2 epochs to work
                # Escape didn't work, revert and reduce LR
                # print(f'\nEscape unsuccessful, reducing LR from {current_lr:.2e} to {current_lr * self.factor:.2e}')
                new_lr = max(current_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.is_escaping = False
                self.wait = 0
            return
            
        # Check if improved
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
        # Check if we should try escaping
        if self.wait >= self.patience:
            self.plateau_count += 1
            
            # If we've hit enough plateaus, try escaping
            if self.plateau_count >= self.escape_patience:
                new_lr = current_lr * self.escape_factor
                print(f'\nAttempting to escape plateau by increasing LR from {current_lr:.2e} to {new_lr:.2e}')
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.is_escaping = True
                self.escape_epoch = epoch
                self.plateau_count = 0
            else:
                # Otherwise, reduce LR as normal
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr != current_lr:
                    # print(f'\nReducing LR from {current_lr:.2e} to {new_lr:.2e}')
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            self.wait = 0
class ProcessedDataManager:
    def __init__(self, data, directory):
        """
        Initialize the ProcessedDataManager.
        
        Args:
            data: The input data DataFrame
            directory: Base directory for the project
        """
        self.directory = directory
        self.data = data.copy(deep=True)
        self.all_modality_shapes = {}
        self.cached_datasets = {}
        
    def process_all_modalities(self):
        """Process all modalities and store their shapes."""
        # Get metadata shape after preprocessing
        print("Processing metadata shape...")
        temp_data = self.data.copy()
        temp_train, _, _, _, _, _ = prepare_cached_datasets(
            temp_data, 
            ['metadata'], 
            train_patient_percentage=0.8,
            batch_size=1,
            run=0,
        )
        for batch in temp_train.take(1):
            self.all_modality_shapes['metadata'] = batch[0]['metadata_input'].shape[1:]
            break
        
        # Set image shapes
        print("Setting image shapes...")
        for modality in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']:
            self.all_modality_shapes[modality] = (image_size, image_size, 3)
        
        del temp_train, temp_data
        gc.collect()
        clear_gpu_memory()
        clear_cache_files()
        
        
    def get_shapes_for_modalities(self, selected_modalities):
        """Get input shapes for selected modalities."""
        return {mod: self.all_modality_shapes[mod] for mod in selected_modalities if mod in self.all_modality_shapes}
class CohenKappa(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name='cohen_kappa', **kwargs):
        super(CohenKappa, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        # Initialize confusion matrix
        self.confusion_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and true values to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        # Update confusion matrix
        confusion = tf.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                mask = tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, j))
                count = tf.reduce_sum(tf.cast(mask, tf.float32))
                confusion = tf.tensor_scatter_nd_add(
                    confusion,
                    [[i, j]],
                    [count]
                )
        
        self.confusion_matrix.assign_add(confusion)
        
    def result(self):
        # Calculate observed agreement
        n = tf.reduce_sum(self.confusion_matrix)
        observed = tf.reduce_sum(tf.linalg.diag_part(self.confusion_matrix)) / n
        
        # Calculate expected agreement
        row_sums = tf.reduce_sum(self.confusion_matrix, axis=1)
        col_sums = tf.reduce_sum(self.confusion_matrix, axis=0)
        expected = tf.reduce_sum((row_sums * col_sums) / (n * n))
        
        # Calculate kappa
        kappa = (observed - expected) / (1.0 - expected + tf.keras.backend.epsilon())
        return kappa
        
    def reset_state(self):
        # Reset confusion matrix to zeros
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))
class WeightedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, alpha_values, name='weighted_accuracy', **kwargs):
        super(WeightedAccuracy, self).__init__(name=name, **kwargs)
        self.alpha_values = tf.constant(alpha_values, dtype=tf.float32)
        self.weighted_true_positives = self.add_weight(name='weighted_tp', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        # Create a mask for correct predictions
        correct_predictions = tf.cast(tf.equal(y_pred, y_true), tf.float32)
        
        # Apply class weights based on true labels
        weights = tf.gather(self.alpha_values, y_true)
        weighted_correct = correct_predictions * weights
        
        # Update metrics
        self.weighted_true_positives.assign_add(tf.reduce_sum(weighted_correct))
        self.total_samples.assign_add(tf.reduce_sum(weights))

    def result(self):
        return self.weighted_true_positives / self.total_samples

    def reset_state(self):
        self.weighted_true_positives.assign(0.0)
        self.total_samples.assign(0.0)

def analyze_modality_contributions(attention_outputs, modality_names):
    """Normalize attention values from each modality to [0, 1] range"""
    normalized_outputs = []
    
    for attention in attention_outputs:
        # Flatten for easier processing
        flat_values = attention.reshape(-1)
        
        # Get min and max
        min_val = np.min(flat_values)
        max_val = np.max(flat_values)
        
        # Normalize to [0, 1]
        normalized = (flat_values - min_val) / (max_val - min_val + 1e-7)
        
        # Reshape back to original shape
        normalized = normalized.reshape(attention.shape)
        normalized_outputs.append(normalized)
    
    return normalized_outputs
class ModalityContributionCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, save_dir='modality_analysis', monitor='val_weighted_accuracy', mode='max', run_number=None):
        super().__init__()
        self.save_dir = save_dir
        self.val_data = val_data
        os.makedirs(save_dir, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.run_number = run_number
        self.best_attention_values = None
        self.best_modality_names = None
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'max':
            improved = current > self.best
        else:
            improved = current < self.best
            
        if improved:
            self.best = current
            print(f"\nGenerating modality contribution analysis for epoch {epoch + 1}")
            
            # # Find layers with attention outputs
            # modality_outputs = {}
            # for layer in self.model.layers:
            #     if 'modular_attention' in layer.name or 'metadata_attention' in layer.name:
            #         modality_name = layer.name.split('_')[0] + "_" + layer.name.split('_')[1]
            #         if "metadata" in modality_name:
            #             modality_name = "metadata"
            #         modality_outputs[modality_name] = layer
            
            modality_outputs = {}
            outputs = []
            for layer in self.model.layers:
                # Look for any layer with final outputs from each modality branch
                # if any(mod in layer.name for mod in ['metadata_BN', 'depth_rgb_projection3', 'depth_map_projection3', 'thermal_map_projection3']):
                if any(mod in layer.name for mod in ['metadata_BN', 'depth_rgb_BN_proj3', 'depth_map_BN_proj3', 'thermal_map_BN_proj3']):
                # if any(mod in layer.name for mod in ['metadata_attention', 'depth_rgb_modular_attention', 'depth_map_modular_attention', 'thermal_map_modular_attention']):
                    modality_name = layer.name.split('_')[0] + "_" + layer.name.split('_')[1]
                    if "metadata" in modality_name:
                        modality_name = "metadata"
                    modality_outputs[modality_name] = layer
                    outputs.append(layer.output)
            
            if modality_outputs:
                # Create visualization model
                vis_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=[self.model.output] + outputs
                )
                
                # Get predictions and attention values
                predictions = vis_model.predict(self.val_data, verbose=0)
                overall_pred = predictions[0]  # Model predictions
                attention_outputs = predictions[1:]  # Attention outputs
                
                # Normalize attention outputs
                attention_outputs = analyze_modality_contributions(
                    attention_outputs, 
                    list(modality_outputs.keys())
                )
                
                # Store best attention values for this run
                self.best_attention_values = attention_outputs
                self.best_modality_names = list(modality_outputs.keys())
                
                # Create multiple visualizations
                fig = plt.figure(figsize=(15, 10))
                gs = plt.GridSpec(2, 2)
                
                # 1. Average attention magnitude per modality
                ax1 = fig.add_subplot(gs[0, 0])
                avg_magnitudes = [np.mean(att) for att in attention_outputs]
                std_magnitudes = [np.std(att) for att in attention_outputs]
                bars = ax1.bar(modality_outputs.keys(), avg_magnitudes, yerr=std_magnitudes)
                ax1.set_title('Average Attention Magnitude per Modality')
                ax1.set_ylabel('Magnitude')
                
                # 2. Attention distribution violin plot
                ax2 = fig.add_subplot(gs[0, 1])
                violin_data = [att.flatten() for att in attention_outputs]
                ax2.violinplot(violin_data)
                ax2.set_xticks(range(1, len(modality_outputs) + 1))
                ax2.set_xticklabels(modality_outputs.keys())
                ax2.set_title('Distribution of Attention Values')
                ax2.set_ylabel('Attention Value')
                ax2.grid(True, axis='y', linestyle='-', linewidth=0.5, color='gray')
                ax2.tick_params(axis='both', which='major', labelsize=18)
                
                # # 3. Feature importance heatmap
                # ax3 = fig.add_subplot(gs[1, :])
                # feature_importance = np.array([np.mean(att, axis=0) for att in attention_outputs[1]])
                # sns.heatmap(feature_importance, 
                #           xticklabels=range(feature_importance.shape[1]),
                #           yticklabels=list(modality_outputs.keys())[1],
                #           cmap='viridis',
                #           ax=ax3)
                # ax3.set_title('Feature Importance per Modality')
                # ax3.set_xlabel('Feature Dimension')
                # ax3.set_ylabel('Modality')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f'modality_analysis_epoch_{epoch+1}.png'))
                plt.close()
                
                # Save numerical results
                with open(os.path.join(self.save_dir, f'modality_stats_epoch_{epoch+1}.txt'), 'w') as f:
                    f.write(f"Epoch {epoch + 1} Modality Analysis\n")
                    f.write("=" * 50 + "\n\n")
                    for mod_name, attention in zip(modality_outputs.keys(), attention_outputs):
                        f.write(f"\n{mod_name.upper()} MODALITY:\n")
                        f.write(f"Mean attention: {np.mean(attention):.4f}\n")
                        f.write(f"Std attention: {np.std(attention):.4f}\n")
                        f.write(f"Max attention: {np.max(attention):.4f}\n")
                        f.write(f"Min attention: {np.min(attention):.4f}\n")
                        
                # Save the raw attention values for later averaging
                np.save(os.path.join(self.save_dir, f'attention_values_run{self.run_number}.npy'), 
                       {'attention_outputs': attention_outputs, 
                        'modality_names': list(modality_outputs.keys())})
def average_attention_values(result_dir, num_runs):
    """Calculate and visualize average attention values across all runs using violin plots."""
    all_attention_outputs = []
    modality_names = None
    name_mapping = {
        'metadata': 'Metadata',
        'depth_rgb': 'RGB',
        'depth_map': 'Depth',
        'thermal_map': 'Thermal'
    }
    # Load attention values from each run
    for run in range(1, num_runs + 1):
        file_path = os.path.join(result_dir, 'modality_analysis', f'attention_values_run{run}.npy')
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True).item()
            # Store both means and raw values
            all_attention_outputs.append({
                'means': [np.mean(att) for att in data['attention_outputs']],
                'raw_values': data['attention_outputs']
            })
            if modality_names is None:
                modality_names = data['modality_names']
    
    if not all_attention_outputs:
        print("No attention values found!")
        return

    # Prepare data for violin plots
    num_modalities = len(modality_names)
    attention_data = [[] for _ in range(num_modalities)]
    run_means_per_modality = [[] for _ in range(num_modalities)]
    
    # Collect all values and means
    for run_data in all_attention_outputs:
        for i in range(num_modalities):
            mean_value = run_data['means'][i]
            run_means_per_modality[i].append(mean_value)
            
            # Only take the attention scores, not the full raw values
            # Assuming attention scores are what we want to visualize
            attention_scores = run_data['raw_values'][i]
            if isinstance(attention_scores, np.ndarray):
                # Take the mean across appropriate dimensions if needed
                if attention_scores.ndim > 1:
                    attention_scores = np.mean(attention_scores, axis=tuple(range(attention_scores.ndim-1)))
                attention_data[i].extend(attention_scores.flatten())

    # Create visualization with violin plots
    plt.figure(figsize=(15, 8))
    
    # Create violin plot with quartile lines
    parts = plt.violinplot(attention_data, positions=range(1, num_modalities + 1),
                          showmeans=True, showextrema=True, showmedians=True)
    
    # Customize violin plot appearance
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    
    # Add individual points for run means
    for i in range(num_modalities):
        plt.scatter([i + 1] * len(run_means_per_modality[i]), 
                   run_means_per_modality[i],
                   color='white', edgecolor='black', 
                   s=100, zorder=3, alpha=0.6)
    
    # Add min, max, and quartile annotations
    for i, data in enumerate(attention_data):
        x_pos = i + 1
        min_val = np.min(data)
        max_val = np.max(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        plt.text(x_pos, min_val, f'{min_val:.3f}', ha='center', va='bottom')
        plt.text(x_pos, max_val, f'{max_val:.3f}', ha='center', va='top')
        plt.text(x_pos + 0.2, q1, f'Q1: {q1:.3f}', ha='left', va='center')
        plt.text(x_pos + 0.2, q3, f'Q3: {q3:.3f}', ha='left', va='center')
    
    modality_namest = [name_mapping.get(name, name) for name in modality_names]
    modality_names = modality_namest
    # Customize plot
    plt.title('Distribution of Attention Values per Modality', fontsize=16, pad=20)
    plt.ylabel('Attention Value', fontsize=16)
    plt.xlabel('Modality', fontsize=16)
    plt.xticks(range(1, num_modalities + 1), modality_names, rotation=45, fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, label='Distribution'),
        plt.Line2D([0], [0], color='red', label='Mean', linewidth=2),
        plt.Line2D([0], [0], color='black', label='Median', linewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='w', markeredgecolor='black',
                  markersize=10, label='Run Means')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'modality_analysis', 'attention_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    with open(os.path.join(result_dir, 'modality_analysis', 'attention_distribution_stats.txt'), 'w') as f:
        f.write("Attention Distribution Analysis Across All Runs\n")
        f.write("=" * 50 + "\n\n")
        for i, mod_name in enumerate(modality_names):
            f.write(f"\n{mod_name.upper()} MODALITY:\n")
            data = attention_data[i]
            f.write(f"Overall mean: {np.mean(data):.4f}\n")
            f.write(f"Overall median: {np.median(data):.4f}\n")
            f.write(f"Overall std: {np.std(data):.4f}\n")
            f.write(f"Q1 (25th percentile): {np.percentile(data, 25):.4f}\n")
            f.write(f"Q3 (75th percentile): {np.percentile(data, 75):.4f}\n")
            f.write(f"Min value: {np.min(data):.4f}\n")
            f.write(f"Max value: {np.max(data):.4f}\n")
            
            f.write("\nRun means:\n")
            for run_idx, run_mean in enumerate(run_means_per_modality[i]):
                f.write(f"Run {run_idx + 1}: {run_mean:.4f}\n")
            f.write("\n")
def cross_validation_manual_split(data, configs, train_patient_percentage=0.8, n_runs=3):
    """
    Perform cross-validation using cached dataset pipeline.
    
    Args:
        data: Input DataFrame
        configs: Dictionary of configurations for different modality combinations
        train_patient_percentage: Percentage of data to use for training
        n_runs: Number of cross-validation runs
    
    Returns:
        Tuple of (all_metrics, all_confusion_matrices, all_histories)
    """
    all_metrics = []
    all_confusion_matrices = []
    all_histories = []
    all_gating_results = []
    all_runs_metrics = []
    
    for run in range(n_runs):
        # Clean up after each modality combination
        try:
            clear_gpu_memory()
            clear_cache_files()
        except Exception as e:
            print(f"Error clearing memory stats: {str(e)}")
        # Reset random seeds for next run
        random.seed(42 + run * (run + 3))
        tf.random.set_seed(42 + run * (run + 3))
        np.random.seed(42 + run * (run + 3))
        os.environ['PYTHONHASHSEED'] = str(42 + run * (run + 3))
        
        print(f"\nRun {run + 1}/{n_runs}")
        
        # Check if this run is already complete
        if is_run_complete(run + 1, ck_path):
            print(f"\nRun {run + 1} is already complete. Moving to next run...")
            continue

        # Try to load aggregated predictions first
        run_predictions_list_t, run_true_labels_t = load_aggregated_predictions(run + 1, ck_path, dataset_type='train')
        run_predictions_list_v, run_true_labels_v = load_aggregated_predictions(run + 1, ck_path, dataset_type='valid')
        if run_predictions_list_t is not None and run_predictions_list_v is not None and run_true_labels_t is not None and run_true_labels_v is not None:
            print(f"\nLoaded aggregated predictions for run {run + 1}")
            print(f"Number of models: {len(run_predictions_list_t)}")
            print(f"Shape of predictions from first model: {run_predictions_list_t[0].shape}")
            print(f"Labels shape: {run_true_labels_t.shape}")
            
            # Proceed directly to gating network training with loaded predictions
            if len(run_predictions_list_t) == len(configs) and len(run_predictions_list_v) == len(configs):
                print(f"\nTraining gating network for run {run + 1}...")
                try:
                    # Convert labels back to class indices for gating network
                    gating_labels_t = np.argmax(run_true_labels_t, axis=1) if len(run_true_labels_t.shape) > 1 else run_true_labels_t
                    gating_labels_v = np.argmax(run_true_labels_v, axis=1) if len(run_true_labels_v.shape) > 1 else run_true_labels_v
                    combined_predictions, gating_labels = train_gating_network(
                        run_predictions_list_t,
                        run_predictions_list_v,
                        gating_labels_t,
                        gating_labels_v,
                        run + 1,
                        find_optimal=False,
                        min_models=2,
                        max_tries=200
                    )

                    # Calculate and store gating network results
                    final_predictions = np.argmax(combined_predictions, axis=1)
                    gating_metrics = {
                        'run': run + 1,
                        'accuracy': accuracy_score(gating_labels, final_predictions),
                        'f1_macro': f1_score(gating_labels, final_predictions, average='macro'),
                        'f1_weighted': f1_score(gating_labels, final_predictions, average='weighted'),
                        'kappa': cohen_kappa_score(gating_labels, final_predictions, weights='quadratic'),
                        'confusion_matrix': confusion_matrix(gating_labels, final_predictions)
                    }
                    
                    print(f"\nGating Network Results for Run {run + 1}:")
                    print(f"Accuracy: {gating_metrics['accuracy']:.4f}")
                    print(f"F1 Macro: {gating_metrics['f1_macro']:.4f}")
                    print(f"Kappa: {gating_metrics['kappa']:.4f}")
                    
                    all_gating_results.append(gating_metrics)
                    save_run_results(gating_metrics, run + 1, result_dir)
                    continue  # Move to next run
                    
                except Exception as e:
                    print(f"Error in gating network training: {str(e)}")
                    # Fall through to regenerate predictions
                    run_predictions_list_t = []
                    run_predictions_list_v = []
                    run_true_labels_t = None
                    run_true_labels_v = None
            else:
                print(f"Found incomplete set of predictions ({len(run_predictions_list_t)} of {len(configs)})")
                run_predictions_list_t = []
                run_predictions_list_v = []
                run_true_labels_t = None
                run_true_labels_v = None
        else:
            run_predictions_list_t = []
            run_predictions_list_v = []
            run_true_labels_t = None
            run_true_labels_v = None
            
        # Get list of completed configs for this run
        completed_configs = get_completed_configs_for_run(run + 1, configs.keys(), ck_path, dataset_type='valid')
        if completed_configs:
            print(f"\nFound completed configs for run {run + 1}: {completed_configs}")
        
        # Initialize data manager for this run
        data_manager = ProcessedDataManager(data.copy(), directory)
        data_manager.process_all_modalities()
        
        # Setup augmentation once per run
        aug_config = AugmentationConfig()
        aug_config.generative_settings['output_size']['width'] = image_size
        aug_config.generative_settings['output_size']['height'] = image_size
        
        gen_manager = GenerativeAugmentationManager(
            base_dir=os.path.join(directory, 'Codes/MultimodalClassification/ImageGeneration/models_5_7'),
            config=aug_config
        )
        
        # Get all unique modalities from all configs
        all_modalities = set()
        for config in configs.values():
            all_modalities.update(config['modalities'])
        all_modalities = list(all_modalities)
        
        print(f"\nPreparing datasets for run {run + 1} with all modalities: {all_modalities}")
        # Create cached datasets once for all modalities
        master_train_dataset, pre_aug_dataset, master_valid_dataset, master_steps_per_epoch, master_validation_steps, master_alpha_value = prepare_cached_datasets(
            data_manager.data,
            all_modalities,  # Use all modalities
            train_patient_percentage=train_patient_percentage,
            batch_size=batch_size,
            gen_manager=gen_manager,
            aug_config=aug_config,
            run=run
        )
        
        run_metrics = []
        
        # For each modality combination
        for config_name, config in configs.items():
            # Before starting new config
            try:
                gc.collect()  # Keep basic garbage collection
                if gpus:
                    for i in range(len(gpus)):
                        try:
                            tf.config.experimental.reset_memory_stats(f'GPU:{i}')  # Reset GPU stats without clearing session
                        except:
                            pass
            except Exception as e:
                print(f"Error in cleanup between configs: {str(e)}")
            # First check if this config is in completed_configs
            if config_name in completed_configs:
                predictions_t, labels_t = load_run_predictions(run + 1, config_name, ck_path, dataset_type='train')
                predictions_v, labels_v = load_run_predictions(run + 1, config_name, ck_path, dataset_type='valid')
                print(f"\nLoading existing predictions for {config_name}")
                run_predictions_list_t.append(predictions_t)
                if run_true_labels_t is None:
                    run_true_labels_t = labels_t
                run_predictions_list_v.append(predictions_v)
                if run_true_labels_v is None:
                    run_true_labels_v = labels_v
                continue
            elif os.path.exists(create_checkpoint_filename(config['modalities'], run+1, config_name)):
                # If not in completed_configs but weights exist, we need to regenerate predictions
                print(f"\nFound weights but no predictions for {config_name}, regenerating predictions")
            else:
                print(f"\nNo existing data found for {config_name}, starting fresh")
            
            selected_modalities = config['modalities']
            print(f"\nTraining {config_name} with modalities: {selected_modalities}, run {run + 1} of {n_runs}")
            
            training_successful = False
            max_retries = 3
            retry_count = 0
            
            while not training_successful and retry_count < max_retries:
                try:
                    # Filter the master datasets for the selected modalities
                    train_dataset = filter_dataset_modalities(master_train_dataset, selected_modalities)
                    pre_aug_train_dataset = filter_dataset_modalities(pre_aug_dataset, selected_modalities)
                    valid_dataset = filter_dataset_modalities(master_valid_dataset, selected_modalities)
                    # Get a single epoch's worth of data by taking the specified number of steps
                    all_labels = []
                    for batch in pre_aug_train_dataset.take(master_steps_per_epoch):
                        _, labels = batch
                        all_labels.extend(np.argmax(labels.numpy(), axis=1))

                    # Calculate class weights
                    master_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
                    master_class_weights_dict = {i: weight for i, weight in enumerate(master_class_weights)}

                    # # Print dataset information for inspection
                    # print("\nInspecting train_dataset:")
                    # for batch in train_dataset.take(1):
                    #     features, labels = batch
                    #     print("Features:")
                    #     for key, value in features.items():
                    #         print(f"  {key}: shape={value.shape}")
                    #     print(f"Labels: shape={labels.shape}")
                    
                    # print("\nInspecting valid_dataset:")
                    # for batch in valid_dataset.take(1):
                    #     features, labels = batch
                    #     print("Features:")
                    #     for key, value in features.items():
                    #         print(f"  {key}: shape={value.shape}")
                    #     print(f"Labels: shape={labels.shape}")
                        
                    steps_per_epoch = master_steps_per_epoch
                    validation_steps = master_validation_steps
                    if config_name.endswith('1'):    
                        alpha_value = master_alpha_value # Proportional class weights (When no mixed_sampling is used)
                        class_weights_dict = {i: 1 for i in range(3)}
                        class_weights = [1, 1, 1]
                    if config_name.endswith('2'):
                        alpha_value = [1, 1, 1]
                        class_weights_dict = master_class_weights_dict
                        class_weights = master_class_weights
                    if config_name.endswith('3'):
                        alpha_value = [4, 1, 4]
                        class_weights_dict = {0: 4, 1: 1, 2: 4}
                        class_weights = [4, 1, 4]
                    # alpha_value = [4, 1, 4]  # Equal class weights
                    print(f"Alpha values (ordered) [I, P, R]: {[round(a, 3) for a in alpha_value]}")
                    print(f"Class weights: {class_weights_dict} or {class_weights}")
                    
                    # Create and train model
                    with strategy.scope():
                        weighted_acc = WeightedAccuracy(alpha_values=class_weights)
                        input_shapes = data_manager.get_shapes_for_modalities(selected_modalities)
                        model = create_multimodal_model(input_shapes, selected_modalities, None)
                        loss = get_focal_ordinal_loss(num_classes=3, ordinal_weight=0.05, gamma=2.0, alpha=class_weights)
                        model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss=loss,
                            metrics=['accuracy', weighted_f1_score, weighted_acc, CohenKappa(num_classes=3)]
                        )
                        # Create distributed datasets
                        train_dataset_dis = strategy.experimental_distribute_dataset(train_dataset)
                        valid_dataset_dis = strategy.experimental_distribute_dataset(valid_dataset)
                        callbacks = [
                            EarlyStopping(
                                patience=20,
                                restore_best_weights=True,
                                monitor='val_loss', #'loss',
                                min_delta=0.01,
                                mode='min',
                                verbose=1
                            ),
                            ReduceLROnPlateau(
                                factor=0.50,
                                patience=5,
                                monitor='val_loss', #'loss',
                                min_delta=0.01,
                                min_lr=1e-10,
                                mode='min',
                            ),
                            tf.keras.callbacks.ModelCheckpoint(
                                create_checkpoint_filename(selected_modalities, run+1, config_name),
                                monitor='val_weighted_accuracy',
                                save_best_only=True,
                                mode='max',
                                save_weights_only=True
                            ),
                            EpochMemoryCallback(strategy), #ACTIVATE THIS WHEN GENERATIVE NOT USED
                            GenerativeAugmentationCallback(gen_manager),
                            NaNMonitorCallback()
                        ]

                        # visualize_dataset(
                        #     train_dataset=train_dataset,
                        #     selected_modalities=selected_modalities,
                        #     save_dir=os.path.join(result_dir, 'train_visualizations_check'),
                        #     max_samples_per_page=20,  # Number of samples per page
                        #     dataset_portion=100,      # Visualize 10% of the dataset
                        #     dpi=75,                 # Image quality
                        #     total_samples=2500
                        # )
                        # visualize_dataset(
                        #     train_dataset=valid_dataset,
                        #     selected_modalities=selected_modalities,
                        #     save_dir=os.path.join(result_dir, 'valid_visualizations_check'),
                        #     max_samples_per_page=20,  # Number of samples per page
                        #     dataset_portion=100,      # Visualize 10% of the dataset
                        #     dpi=75,                 # Image quality
                        #     total_samples=700
                        # )
                        # Add visualization callbacks for first run
                        if run == 0:
                            callbacks.insert(3, BatchVisualizationCallback(
                                dataset=train_dataset,
                                modalities=selected_modalities,
                                freq=1000,
                                max_samples=20,
                                run=run + 1,
                                save_dir=os.path.join(result_dir, 'batch_visualizations_generative')
                            ))
                            callbacks.insert(4, TrainingHistoryCallback(
                                save_dir=os.path.join(result_dir, 'training_plots_generative'),
                                update_freq=5000
                            ))
                            if 'metadata' in selected_modalities:
                                callbacks.insert(5, MetadataConfidenceCallback(
                                    selected_modalities=selected_modalities,
                                    log_dir=os.path.join(result_dir, 'metadata_confidence_logs')
                                ))
                        VISUALIZE_MODALITIES = False
                        if VISUALIZE_MODALITIES:
                                callbacks.append(ModalityContributionCallback(
                                    val_data=valid_dataset,
                                    save_dir=os.path.join(result_dir, 'modality_analysis'),
                                    monitor='val_weighted_accuracy',
                                    mode='max',
                                    run_number=run + 1
                                ))

                        # Train model
                        if os.path.exists(create_checkpoint_filename(selected_modalities, run+1, config_name)):
                            model.load_weights(create_checkpoint_filename(selected_modalities, run+1, config_name))
                            print("Loaded existing weights")
                        else:
                            print("No existing pretrained weights found")
                            print(f"Total model trainable weights: {len(model.trainable_weights)}")
                            history = model.fit(
                                train_dataset_dis,
                                epochs=n_epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=valid_dataset_dis,
                                validation_steps=validation_steps,
                                callbacks=callbacks,
                                verbose=0
                            )
                        
                        model.load_weights(create_checkpoint_filename(selected_modalities, run+1, config_name)) # Load best Validation weights
                        # Evaluate training data
                        y_true_t = []
                        y_pred_t = []
                        probabilities_t = []
                        all_sample_ids_t = []

                        with strategy.scope():
                            for batch in pre_aug_train_dataset.take(steps_per_epoch):
                                batch_inputs, batch_labels = batch
                                with strategy.scope():
                                    batch_pred = model.predict(batch_inputs, verbose=0)
                                y_true_t.extend(np.argmax(batch_labels, axis=1))
                                y_pred_t.extend(np.argmax(batch_pred, axis=1))
                                probabilities_t.extend(batch_pred)
                                all_sample_ids_t.extend(batch_inputs['sample_id'].numpy())
                                
                                del batch_inputs, batch_labels, batch_pred
                                gc.collect()

                        save_run_predictions(run + 1, config_name, np.array(probabilities_t), np.array(y_true_t), ck_path, dataset_type='train')
                        # Store probabilities for gating network
                        run_predictions_list_t.append(np.array(probabilities_t))
                        if run_true_labels_t is None:
                            run_true_labels_t = np.array(y_true_t)

                        # Track misclassifications
                        sample_ids_t = np.array(all_sample_ids_t)
                        track_misclassifications(np.array(y_true_t), np.array(y_pred_t), sample_ids_t, selected_modalities, result_dir)
                        
                        # Evaluate model
                        y_true_v = []
                        y_pred_v = []
                        probabilities_v = []
                        all_sample_ids_v = []

                        with strategy.scope():
                            for batch in valid_dataset.take(validation_steps):
                                batch_inputs, batch_labels = batch
                                with strategy.scope():
                                    batch_pred = model.predict(batch_inputs, verbose=0)
                                y_true_v.extend(np.argmax(batch_labels, axis=1))
                                y_pred_v.extend(np.argmax(batch_pred, axis=1))
                                probabilities_v.extend(batch_pred)
                                all_sample_ids_v.extend(batch_inputs['sample_id'].numpy())
                                
                                del batch_inputs, batch_labels, batch_pred
                                gc.collect()

                        save_run_predictions(run + 1, config_name, np.array(probabilities_v), np.array(y_true_v), ck_path, dataset_type='valid')
                        # Store probabilities for gating network
                        run_predictions_list_v.append(np.array(probabilities_v))
                        if run_true_labels_v is None:
                            run_true_labels_v = np.array(y_true_v)
                        
                        # Track misclassifications
                        sample_ids_v = np.array(all_sample_ids_v)
                        track_misclassifications(np.array(y_true_v), np.array(y_pred_v), sample_ids_v, selected_modalities, result_dir)

                        # Calculate metrics
                        accuracy = accuracy_score(y_true_v, y_pred_v)
                        f1_macro = f1_score(y_true_v, y_pred_v, average='macro', zero_division=0)
                        f1_weighted = f1_score(y_true_v, y_pred_v, average='weighted', zero_division=0)
                        f1_classes = f1_score(y_true_v, y_pred_v, average=None, zero_division=0, labels=[0, 1, 2])
                        kappa = cohen_kappa_score(y_true_v, y_pred_v, weights='quadratic')
                        
                        # Store results
                        cm = confusion_matrix(y_true_v, y_pred_v, labels=[0, 1, 2])
                        all_confusion_matrices.append(cm)
                        
                        metrics_dict = {
                            'run': run + 1,
                            'config': config_name,
                            'modalities': selected_modalities,
                            'accuracy': accuracy,
                            'f1_macro': f1_macro,
                            'f1_weighted': f1_weighted,
                            'f1_classes': f1_classes,
                            'kappa': kappa,
                            'y_true': y_true_v,
                            'y_pred': y_pred_v,
                            'probabilities': probabilities_v
                        }
                        save_run_metrics(metrics_dict, run + 1, result_dir)
                        run_metrics.append(metrics_dict)
                        
                        # Print results
                        print(f"\nRun {run + 1} Results for {config_name}:")
                        print(classification_report(y_true_v, y_pred_v,
                                                target_names=CLASS_LABELS,
                                                labels=[0, 1, 2],
                                                zero_division=0))
                        print(f"Cohen's Kappa: {kappa:.4f}")
                        
                        training_successful = True

                except Exception as e:
                    print(f"Error during training: {str(e)}")
                    clean_up_training_resources()
                    retry_count += 1
                    continue
                
                finally:
                    # Clean up
                    gen_manager.cleanup()
                    gc.collect()
                    
        
        save_run_metrics(run_metrics, run + 1, result_dir)
        save_aggregated_predictions(run + 1, run_predictions_list_t, run_true_labels_t, ck_path, dataset_type='train')
        save_aggregated_predictions(run + 1, run_predictions_list_v, run_true_labels_v, ck_path, dataset_type='valid')
        
        # Train gating network if we have all predictions
        if len(run_predictions_list_v) == len(configs) and len(run_predictions_list_t) == len(configs):
            print(f"\nTraining gating network for run {run + 1}...")
            try:
                # # First validate and correct predictions
                # truncated_predictions_t, run_true_labels_t = correct_and_validate_predictions(
                #     run_predictions_list_t, run_true_labels_t, "train")
                
                # # Process validation data
                # truncated_predictions_v, run_true_labels_v = correct_and_validate_predictions(
                #     run_predictions_list_v, run_true_labels_v, "valid")
                
                print(f"\nNumber of models: {len(run_predictions_list_t)}")
            except Exception as e:
                print(f"Error in prediction validation: {str(e)}")
            try:
                # Convert labels back to class indices for gating network
                gating_labels_t = np.argmax(run_true_labels_t, axis=1) if len(run_true_labels_t.shape) > 1 else run_true_labels_t
                gating_labels_v = np.argmax(run_true_labels_v, axis=1) if len(run_true_labels_v.shape) > 1 else run_true_labels_v
                combined_predictions, gating_labels = train_gating_network(
                    run_predictions_list_t,
                    run_predictions_list_v,
                    gating_labels_t,
                    gating_labels_v,
                    run + 1,
                    find_optimal=False,
                    min_models=2,
                    max_tries=200
                )
                
                # Calculate and store gating network results for this run
                final_predictions = np.argmax(combined_predictions, axis=1)
                gating_metrics = {
                    'run': run + 1,
                    'accuracy': accuracy_score(gating_labels, final_predictions),
                    'f1_macro': f1_score(gating_labels, final_predictions, average='macro'),
                    'f1_weighted': f1_score(gating_labels, final_predictions, average='weighted'),
                    'kappa': cohen_kappa_score(gating_labels, final_predictions, weights='quadratic'),
                    'confusion_matrix': confusion_matrix(gating_labels, final_predictions)
                }
                
                print(f"\nGating Network Results for Run {run + 1}:")
                print(f"Accuracy: {gating_metrics['accuracy']:.4f}")
                print(f"F1 Macro: {gating_metrics['f1_macro']:.4f}")
                print(f"Kappa: {gating_metrics['kappa']:.4f}")
                
                all_gating_results.append(gating_metrics)
                # Save individual run results
                save_run_results(gating_metrics, run + 1, result_dir)
                
            except Exception as e:
                print(f"Error in gating network training for run {run + 1}: {str(e)}")
                gating_metrics = None
            
            all_runs_metrics.extend(run_metrics)
            # Clean up after the run
            try:
                tf.keras.backend.clear_session()
                gc.collect()
                clear_gpu_memory()
            except Exception as e:
                print(f"Error clearing memory stats: {str(e)}")
        
    # Save aggregated results
    save_aggregated_results(all_runs_metrics, configs, result_dir)
    save_gating_results(all_gating_results, result_dir)
    
    return all_metrics, all_confusion_matrices, all_histories
def correct_and_validate_predictions(predictions_list, true_labels, dataset_type="train"):
    """
    Correct and validate model predictions.
    
    Args:
        predictions_list: List of model predictions
        true_labels: True labels
        dataset_type: String indicating "train" or "valid" for logging
    
    Returns:
        tuple: (truncated_predictions, truncated_labels)
    """
    corrected_predictions = []
    
    # Validate and correct predictions
    for i, preds in enumerate(predictions_list):
        if not isinstance(preds, np.ndarray) or preds is None:
            print(f"Warning: Invalid {dataset_type} predictions from model {i}. Skipping...")
            continue
            
        preds = np.array(preds, dtype=np.float32)
        if len(preds.shape) != 2:
            print(f"Warning: Invalid {dataset_type} shape {preds.shape} from model {i}. Skipping...")
            continue
        
        # Check and correct predictions
        if preds.shape[1] != 3:
            print(f"Warning: Model {i} {dataset_type} predictions have {preds.shape[1]} classes instead of 3")
            corrected = np.zeros((preds.shape[0], 3), dtype=np.float32)
            for c in range(min(preds.shape[1], 3)):
                corrected[:, c] = preds[:, c]
            for c in range(preds.shape[1], 3):
                corrected[:, c] = 1e-7
            row_sums = corrected.sum(axis=1, keepdims=True)
            corrected = corrected / row_sums
            corrected_predictions.append(corrected)
        else:
            corrected_predictions.append(preds)
    
    if not corrected_predictions:
        raise ValueError(f"No valid {dataset_type} predictions found after correction")
    
    # Find minimum length
    min_length = min(p.shape[0] for p in corrected_predictions)
    # print(f"Minimum length of {dataset_type} predictions: {min_length}")
    
    # Truncate predictions and labels
    truncated_predictions = [p[:min_length] for p in corrected_predictions]
    truncated_labels = true_labels[:min_length]
    
    # Convert labels to one-hot if needed
    if len(truncated_labels.shape) == 1:
        truncated_labels = tf.keras.utils.to_categorical(truncated_labels, num_classes=3)
    
    # print(f"Shape of {dataset_type} predictions from first model: {truncated_predictions[0].shape}")
    # print(f"Shape of {dataset_type} labels: {truncated_labels.shape}")
    
    # Verify all predictions have correct shape
    for i, preds in enumerate(truncated_predictions):
        if preds.shape != (min_length, 3):
            raise ValueError(f"Model {i} {dataset_type} predictions have incorrect shape: {preds.shape}")
    
    return truncated_predictions, truncated_labels
def save_run_results(metrics, run_number, result_dir):
    """Save gating network results for an individual run."""
    csv_filename = os.path.join(result_dir, f'gating_network_run_{run_number}_results.csv')
    
    # Format metrics for CSV
    results = [{
        'run': int(metrics['run']),
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'kappa': float(metrics['kappa'])
    }]
    
    # Save confusion matrix separately
    cm_filename = os.path.join(result_dir, f'gating_network_run_{run_number}_confusion_matrix.csv')
    np.savetxt(cm_filename, metrics['confusion_matrix'], delimiter=',', fmt='%d')
    
    # Save metrics to CSV
    fieldnames = list(results[0].keys())
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
def save_run_metrics(run_metrics, run_number, result_dir):
    # Handle single metrics dict vs list of metrics
    is_single_metric = not isinstance(run_metrics, list)
    
    # Format the metrics
    if is_single_metric:
        csv_filename = os.path.join(result_dir, f'modality_results_run_{run_number}.csv')
        formatted_result = {
            'config': run_metrics['config'],
            'modalities': '+'.join(run_metrics['modalities']),
            'accuracy': run_metrics['accuracy'],
            'f1_macro': run_metrics['f1_macro'],
            'f1_weighted': run_metrics['f1_weighted'],
            'I_f1': run_metrics['f1_classes'][0],
            'P_f1': run_metrics['f1_classes'][1],
            'R_f1': run_metrics['f1_classes'][2],
            'kappa': run_metrics['kappa']
        }
        fieldnames = list(formatted_result.keys())
        
        # Save to CSV with append mode
        write_mode = 'w' if not os.path.exists(csv_filename) else 'a'
        with open(csv_filename, write_mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_mode == 'w':
                writer.writeheader()
            writer.writerow(formatted_result)
            
    else:
        csv_filename = os.path.join(result_dir, f'modality_results_run_{run_number}_list.csv')
        # Format list of metrics
        formatted_results = []
        for m in run_metrics:
            formatted_results.append({
                'config': m['config'],
                'modalities': '+'.join(m['modalities']),
                'accuracy': m['accuracy'],
                'f1_macro': m['f1_macro'],
                'f1_weighted': m['f1_weighted'],
                'I_f1': m['f1_classes'][0],
                'P_f1': m['f1_classes'][1],
                'R_f1': m['f1_classes'][2],
                'kappa': m['kappa']
            })
        
        # Save list to CSV (overwrite mode)
        if formatted_results:
            fieldnames = list(formatted_results[0].keys())
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(formatted_results)
def save_gating_results(all_gating_results, result_dir):
    """Save aggregated gating network results to CSV."""
    if not all_gating_results:
        return
        
    # Calculate average metrics
    results = [{
        'accuracy_mean': np.mean([r['accuracy'] for r in all_gating_results]),
        'accuracy_std': np.std([r['accuracy'] for r in all_gating_results]),
        'f1_macro_mean': np.mean([r['f1_macro'] for r in all_gating_results]),
        'f1_macro_std': np.std([r['f1_macro'] for r in all_gating_results]),
        'f1_weighted_mean': np.mean([r['f1_weighted'] for r in all_gating_results]),
        'f1_weighted_std': np.std([r['f1_weighted'] for r in all_gating_results]),
        'kappa_mean': np.mean([r['kappa'] for r in all_gating_results]),
        'kappa_std': np.std([r['kappa'] for r in all_gating_results])
    }]
    
    # Save to file
    csv_filename = os.path.join(result_dir, 'gating_network_averaged_results.csv')
    fieldnames = list(results[0].keys())
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def save_aggregated_results(all_metrics, configs, result_dir):
    """Save aggregated results to CSV file."""
    results = []
    
    # Group metrics by configuration
    for config_name in configs.keys():
        config_metrics = [m for m in all_metrics if m['config'] == config_name]
        
        if not config_metrics:
            continue
            
        avg_accuracy = np.mean([m['accuracy'] for m in config_metrics])
        std_accuracy = np.std([m['accuracy'] for m in config_metrics])
        avg_f1_macro = np.mean([m['f1_macro'] for m in config_metrics])
        std_f1_macro = np.std([m['f1_macro'] for m in config_metrics])
        avg_f1_weighted = np.mean([m['f1_weighted'] for m in config_metrics])
        std_f1_weighted = np.std([m['f1_weighted'] for m in config_metrics])
        avg_kappa = np.mean([m['kappa'] for m in config_metrics])
        std_kappa = np.std([m['kappa'] for m in config_metrics])
        
        # Calculate average F1-scores for each class
        f1_classes_list = [m['f1_classes'] for m in config_metrics]
        avg_f1_classes = np.mean(f1_classes_list, axis=0)
        
        results.append({
            'Modalities': '+'.join(configs[config_name]['modalities']),
            'Accuracy (Mean)': avg_accuracy,
            'Accuracy (Std)': std_accuracy,
            'Macro Avg F1-score (Mean)': avg_f1_macro,
            'Macro Avg F1-score (Std)': std_f1_macro,
            'Weighted Avg F1-score (Mean)': avg_f1_weighted,
            'Weighted Avg F1-score (Std)': std_f1_weighted,
            'I F1-score (Mean)': avg_f1_classes[0],
            'P F1-score (Mean)': avg_f1_classes[1],
            'R F1-score (Mean)': avg_f1_classes[2],
            "Cohen's Kappa (Mean)": avg_kappa,
            "Cohen's Kappa (Std)": std_kappa
        })
    
    # Save to CSV
    csv_filename = os.path.join(result_dir, 'modality_results_averaged.csv')
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {csv_filename}")
def save_run_predictions(run_number, config_name, predictions, true_labels, ck_path, dataset_type='valid'):
    """Save predictions and true labels for a specific run and config."""
    pred_file = os.path.join(ck_path, f'pred_run{run_number}_{config_name}_{dataset_type}.npy')
    labels_file = os.path.join(ck_path, f'true_label_run{run_number}_{config_name}_{dataset_type}.npy')
    np.save(pred_file, predictions)
    np.save(labels_file, true_labels)

def load_run_predictions(run_number, config_name, ck_path, dataset_type='valid'):
    """Load predictions and true labels for a specific run and config."""
    pred_file = os.path.join(ck_path, f'pred_run{run_number}_{config_name}_{dataset_type}.npy')
    labels_file = os.path.join(ck_path, f'true_label_run{run_number}_{config_name}_{dataset_type}.npy')
    
    if os.path.exists(pred_file) and os.path.exists(labels_file):
        return np.load(pred_file), np.load(labels_file)
    return None, None

def get_completed_configs_for_run(run_number, config_names, ck_path, dataset_type='valid'):
    """Get list of configs that have completed predictions for a specific run."""
    completed = []
    for config_name in config_names:
        pred_file = os.path.join(ck_path, f'pred_run{run_number}_{config_name}_{dataset_type}.npy')
        labels_file = os.path.join(ck_path, f'true_label_run{run_number}_{config_name}_{dataset_type}.npy')
        if os.path.exists(pred_file) and os.path.exists(labels_file):
            completed.append(config_name)
    return completed

def load_aggregated_predictions(run_number, ck_path, dataset_type='valid'):
    """Load all predictions for a run if they exist."""
    aggregated_preds_file = os.path.join(ck_path, f'sum_pred_run{run_number}_{dataset_type}.npy')
    aggregated_labels_file = os.path.join(ck_path, f'sum_label_run{run_number}_{dataset_type}.npy')
    
    if os.path.exists(aggregated_preds_file) and os.path.exists(aggregated_labels_file):
        return np.load(aggregated_preds_file), np.load(aggregated_labels_file)
    return None, None

def save_aggregated_predictions(run_number, predictions_list, true_labels, ck_path, dataset_type='valid'):
    """Save aggregated predictions with validation and correction."""
    aggregated_preds_file = os.path.join(ck_path, f'sum_pred_run{run_number}_{dataset_type}.npy')
    aggregated_labels_file = os.path.join(ck_path, f'sum_label_run{run_number}_{dataset_type}.npy')
    
    # Validate and correct predictions
    corrected_predictions = []
    for i, preds in enumerate(predictions_list):
        if preds is None:
            print(f"Warning: Predictions from model {i} are None. Skipping...")
            continue
            
        preds = np.array(preds)
        if len(preds.shape) != 2:
            print(f"Warning: Invalid shape {preds.shape} from model {i}. Skipping...")
            continue
            
        # Check if we have predictions for all three classes
        if preds.shape[1] != 3:
            print(f"Warning: Model {i} predictions have {preds.shape[1]} classes instead of 3")
            # Create corrected array with proper shape
            corrected_preds = np.zeros((preds.shape[0], 3))
            
            # Copy existing predictions
            for c in range(min(preds.shape[1], 3)):
                corrected_preds[:, c] = preds[:, c]
                
            # For missing classes, set very low confidence
            for c in range(preds.shape[1], 3):
                # Use small non-zero value to avoid numerical issues
                corrected_preds[:, c] = 1e-7
                
            # Renormalize probabilities to sum to 1
            row_sums = corrected_preds.sum(axis=1, keepdims=True)
            corrected_preds = corrected_preds / row_sums
            
            print(f"Corrected predictions shape: {corrected_preds.shape}")
            corrected_predictions.append(corrected_preds)
        else:
            corrected_predictions.append(preds)
    
    # Final validation
    if not corrected_predictions:
        raise ValueError("No valid predictions to save")
    
    shapes = [p.shape for p in corrected_predictions]
    if len(set(shapes)) > 1:
        print(f"Warning: Inconsistent shapes after correction: {shapes}")
        # Find minimum number of samples across all predictions
        min_samples = min(s[0] for s in shapes)
        # Truncate all predictions to minimum length
        corrected_predictions = [p[:min_samples] for p in corrected_predictions]
        if true_labels is not None:
            true_labels = true_labels[:min_samples]
            
    try:
        np.save(aggregated_preds_file, corrected_predictions)
        if true_labels is not None:
            np.save(aggregated_labels_file, true_labels)
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        raise

def is_run_complete(run_number, ck_path):
    """Check if a run has completed gating metrics."""
    gating_metrics_file = os.path.join(ck_path, f'gating_network_run_{run_number}_results.csv')
    return os.path.exists(gating_metrics_file)
def main_with_specialized_evaluation(data_percentage=100, train_patient_percentage=0.8, n_runs=3):
    """
    Run specialized evaluation with the new cross-validation structure.
    """

    configs = {
        'a32': {#0
            'modalities': ['metadata'],
        },
        'b32': {#1
            'modalities': ['depth_rgb'],
        },
        'c32': {#2
            'modalities': ['depth_map'],
        },
        'd32': {#3
            'modalities': ['thermal_map'],
        },
        # 'e32': {#4*
        #     'modalities': ['metadata','depth_rgb'],
        # },
        # 'f32': {#5*
        #     'modalities': ['metadata','depth_map'],
        # },
        # 'g32': {#6*
        #     'modalities': ['metadata','thermal_map'],
        # },
        # 'h32': {#7
        #     'modalities': ['depth_rgb','depth_map'],
        # },
        # 'i32': {#8
        #     'modalities': ['depth_rgb','thermal_map'],
        # },
        # # 'i32': {#8 TEMPT
        # #     'modalities': ['depth_rgb','thermal_rgb'],
        # # },
        'j32': {#9*
            'modalities': ['depth_map','thermal_map'],
        },
        'k32': {#10
            'modalities': ['metadata','depth_rgb','depth_map'],
        },
        'z32': {#11
            'modalities': ['metadata','depth_rgb','thermal_map'],
        },
        'm32': {#12*
            'modalities': ['metadata','depth_map','thermal_map'],
        },
        'n32': {#13
            'modalities': ['depth_rgb','depth_map','thermal_map'],
        },
        # 'p3222': {#14
        #     'modalities': ['metadata','depth_rgb','depth_map','thermal_map'],
        # },
    }
    
    # Clear any existing cache files
    clear_cache_files()
    
    # Prepare initial dataset
    print("Preparing initial dataset...")
    data = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, 
                         list(set([mod for config in configs.values() 
                                 for mod in config['modalities']])))
    
    # Filter frequent misclassifications
    print("Filtering frequent misclassifications...")
    data = filter_frequent_misclassifications(
        data, result_dir, 
        thresholds={'I': 3, 'P': 2, 'R': 3}
    )
    
    if data_percentage < 100:
        data = data.sample(frac=data_percentage / 100, random_state=42).reset_index(drop=True)
    
    # Run cross-validation
    print("\nStarting cross-validation...")
    metrics, confusion_matrices, histories = cross_validation_manual_split(
        data, configs, train_patient_percentage, n_runs
    )
    # After all runs are complete
    # average_attention_values(result_dir, num_runs=n_runs)
    
    # # Train gating network if needed
    # if os.path.exists(os.path.join(result_dir, 'predictions_list.npy')):
    #     print("\nTraining gating network...")
    #     predictions_list = np.load(os.path.join(result_dir, 'predictions_list.npy'), allow_pickle=True)
    #     true_labels_onehot = np.load(os.path.join(result_dir, 'true_labels_onehot.npy'), allow_pickle=True)
    #     true_labels = np.argmax(true_labels_onehot, axis=1)
        
    #     combined_predictions = train_gating_network(
    #         predictions_list, true_labels, run_number=1,
    #         find_optimal=True, min_models=2, max_tries=150
    #     )
        
    #     final_predictions = np.argmax(combined_predictions, axis=1)
    #     final_metrics = calculate_and_save_metrics(true_labels, final_predictions, result_dir, configs)
    
    return metrics

def filter_dataset_modalities(dataset, selected_modalities):
    """
    Filter a dataset to only include the selected modalities.
    
    Args:
        dataset: A tf.data.Dataset that contains all modalities
        selected_modalities: List of modalities to keep
    
    Returns:
        A new dataset containing only the selected modalities
    """
    def filter_features(features, labels):
        filtered_features = {}
        # Always include sample_id
        filtered_features['sample_id'] = features['sample_id']
        
        # Include only selected modalities
        for modality in selected_modalities:
            modality_key = f'{modality}_input'
            if modality_key in features:
                filtered_features[modality_key] = features[modality_key]
            else:
                raise KeyError(f"Modality {modality} not found in dataset")
        
        return filtered_features, labels
    
    # return dataset.map(filter_features, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.map(filter_features, num_parallel_calls=2)

def clear_cache_files():
    """Clear any existing cache files."""
    import glob
    
    cache_patterns = [
        os.path.join(result_dir, 'tf_cache_train*'),
        os.path.join(result_dir, 'tf_cache_valid*'),
        'tf_cache_train*',
        'tf_cache_valid*'
    ]
    
    for pattern in cache_patterns:
        try:
            cache_files = glob.glob(pattern)
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                    print(f"Removed cache file: {cache_file}")
                except Exception as e:
                    print(f"Warning: Could not remove cache file {cache_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error while processing pattern {pattern}: {str(e)}")
#%% Main Function
def create_focal_ordinal_loss_with_params(ordinal_weight, gamma, alpha):
    """Create a focal ordinal loss function with specified parameters"""
    print("\nInitializing focal ordinal loss:")
    print(f"Ordinal weight: {ordinal_weight}")
    print(f"Gamma: {gamma}")
    print(f"Alpha: {alpha}")
    
    def loss_fn(y_true, y_pred):
        # Print shapes for debugging
        print(f"\nLoss function shapes:")
        print(f"y_true shape: {tf.shape(y_true)}")
        print(f"y_pred shape: {tf.shape(y_pred)}")
        
        # Clip prediction values to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Convert alpha to tensor with proper shape
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        
        # Focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.expand_dims(alpha_tensor, 0) * tf.math.pow(1 - y_pred, gamma)
        focal_loss = focal_weight * cross_entropy
        
        # Print intermediate values for first sample
        print("\nIntermediate values (first sample):")
        print(f"Cross entropy: {cross_entropy[0]}")
        print(f"Focal weight: {focal_weight[0]}")
        print(f"Focal loss: {focal_loss[0]}")
        
        # Sum over classes
        focal_loss = tf.reduce_sum(focal_loss, axis=-1)
        
        # Ordinal penalty
        true_class = tf.argmax(y_true, axis=-1)
        pred_class = tf.argmax(y_pred, axis=-1)
        ordinal_penalty = tf.square(tf.cast(true_class, tf.float32) - tf.cast(pred_class, tf.float32))
        
        # Print ordinal penalty information
        print(f"Ordinal penalty (first 5 samples): {ordinal_penalty[:5]}")
        
        # Combine losses
        total_loss = focal_loss + ordinal_weight * ordinal_penalty
        
        # Print final loss
        print(f"Final loss (first 5 samples): {total_loss[:5]}")
        return total_loss
    
    return loss_fn

def create_attention_visualization_callback(val_data, val_labels, save_dir='attention_weights'):
    """Enhanced callback with explicit validation data"""
    os.makedirs(save_dir, exist_ok=True)
    
    class AttentionVisualizerCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_data, val_labels, save_dir):
            super().__init__()
            self.save_dir = save_dir
            self.class_attention_history = {
                'I': [], 'P': [], 'R': []
            }
            self.val_data = val_data
            self.val_labels = val_labels


        def on_train_begin(self, logs=None):
            """Store validation data at the start of training"""
            if hasattr(self.model, 'validation_data'):
                self.val_data = self.model.validation_data[0]
                self.val_labels = self.model.validation_data[1]
            # else:
            #     print("Warning: No validation data found in model")

        def analyze_class_specific_attention(self, model_weights, class_weights, predictions, true_labels):
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(true_labels, axis=1)
            
            # Analyze attention patterns for each class
            class_analysis = {}
            for class_idx, class_name in enumerate(['I', 'P', 'R']):
                # Get samples for this class
                class_mask = true_classes == class_idx
                correct_mask = (pred_classes == class_idx) & class_mask
                
                if np.any(class_mask):
                    class_analysis[class_name] = {
                        'model_attention': {
                            'all': np.mean(model_weights[class_mask], axis=(0,1)),
                            'correct': np.mean(model_weights[correct_mask], axis=(0,1)) if np.any(correct_mask) else None
                        },
                        'class_attention': {
                            'all': np.mean(class_weights[class_mask], axis=(0,1)),
                            'correct': np.mean(class_weights[correct_mask], axis=(0,1)) if np.any(correct_mask) else None
                        },
                        'accuracy': np.mean(correct_mask[class_mask]) if np.any(class_mask) else 0
                    }
            
            return class_analysis

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 1 == 0:
                print(f"\nProcessing attention visualization for epoch {epoch + 1}")
                
                attention_layer = None
                for layer in self.model.layers:
                    if isinstance(layer, DualLevelAttentionLayer):
                        attention_layer = layer
                        break
                
                if attention_layer:
                    try:
                        attention_values = attention_layer.get_attention_values()
                        
                        if attention_values['model_attention'] is None:
                            print("Warning: model_attention is None")
                            return
                        if attention_values['class_attention'] is None:
                            print("Warning: class_attention is None")
                            return
                            
                        model_weights = attention_values['model_attention']
                        class_weights = attention_values['class_attention']
                        
                        # Average over batch and heads if needed
                        Avg_model_weights = np.mean(model_weights, axis=(0, 1))
                        Avg_class_weights = np.mean(class_weights, axis=(0, 1))
                        
                        # Define global min and max values for consistent scaling
                        model_vmin = 0.075  # minimum from all model attention weights
                        model_vmax = 0.275  # maximum from all model attention weights
                        class_vmin = 0.20   # minimum from all class attention weights
                        class_vmax = 0.45   # maximum from all class attention weights
                        # Create model name mapping
                        model_names = {
                            '1': 'Metadata',
                            '2': 'RGB',
                            '3': 'Depth',
                            '4': 'Thermal',
                            '5': 'Thermal+Depth',
                            '6': 'Metadata+RGB+Depth',
                            '7': 'Metadata+RGB+Thermal',
                            '8': 'Metadata+Thermal+Depth',
                            '9': 'RGB+Thermal+Depth'
                        }
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                        
                        # Plot model attention with fixed scale
                        sns.heatmap(Avg_model_weights, ax=ax1, cmap='viridis',
                                    vmin=model_vmin, vmax=model_vmax,
                                    xticklabels=range(1, Avg_model_weights.shape[1] + 1),
                                    yticklabels=range(1, Avg_model_weights.shape[0] + 1),
                                    cbar_kws={'label': 'Attention Weight'})
                        ax1.set_title('Model-Level Attention', fontsize=14)
                        ax1.set_xlabel('Target Models', fontsize=14)
                        ax1.set_ylabel('Source Models', fontsize=14)
                        ax1.tick_params(axis='both', which='major', labelsize=14)
                        
                        # Add legend for model names
                        legend_elements = [f"{k}: {v}" for k, v in model_names.items()]
                        ax1.text(1.3, 0.5, '\n'.join(legend_elements),
                                transform=ax1.transAxes, fontsize=14,
                                verticalalignment='center')

                        # Plot class attention with fixed scale
                        sns.heatmap(Avg_class_weights, ax=ax2, cmap='viridis',
                                    vmin=class_vmin, vmax=class_vmax,
                                    xticklabels=['I', 'P', 'R'],
                                    yticklabels=['I', 'P', 'R'],
                                    cbar_kws={'label': 'Attention Weight'})
                        ax2.set_title('Class-Level Attention', fontsize=14)
                        ax2.set_xlabel('Target Classes', fontsize=14)
                        ax2.set_ylabel('Source Classes', fontsize=14)
                        ax2.tick_params(axis='both', which='major', labelsize=14)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.save_dir, f'Avg_attention_weights_epoch_{epoch+1}.png'))
                        plt.close()
                        
                        if self.val_data is None or self.val_labels is None:
                            print("No validation data available. Trying to get from model...")
                            if hasattr(self.model, 'validation_data'):
                                self.val_data = self.model.validation_data[0]
                                self.val_labels = self.model.validation_data[1]
                            else:
                                print("Error: Cannot access validation data")
                                return
                        
                        # Get predictions
                        val_predictions = self.model.predict(self.val_data, verbose=0)
                        val_labels = self.val_labels
                        
                        # Analyze per-class attention
                        class_analysis = self.analyze_class_specific_attention(
                            model_weights, class_weights, val_predictions, val_labels
                        )
                        
                        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
                        for idx, class_name in enumerate(['I', 'P', 'R']):
                            if class_name in class_analysis:
                                analysis = class_analysis[class_name]
                                
                                # Plot model attention for this class with fixed scale
                                sns.heatmap(analysis['model_attention']['all'].reshape(-1, model_weights.shape[-1]), 
                                            ax=axes[idx,0], cmap='viridis',
                                            vmin=model_vmin, vmax=model_vmax,
                                            xticklabels=range(1, model_weights.shape[-1] + 1),
                                            yticklabels=range(1, model_weights.shape[-1] + 1),
                                            cbar_kws={'label': 'Attention Weight'})
                                axes[idx,0].set_title(f'{class_name} Class Model Attention', fontsize=14)
                                axes[idx,0].tick_params(axis='both', which='major', labelsize=14)
                                
                                # Add legend for the first plot only
                                if idx == 0:
                                    legend_elements = [f"{k}: {v}" for k, v in model_names.items()]
                                    axes[idx,0].text(1.3, 0.5, '\n'.join(legend_elements),
                                                transform=axes[idx,0].transAxes, fontsize=14,
                                                verticalalignment='center')
                                
                                # Plot class attention for this class with fixed scale
                                sns.heatmap(analysis['class_attention']['all'].reshape(-1, class_weights.shape[-1]),
                                            ax=axes[idx,1], cmap='viridis',
                                            vmin=class_vmin, vmax=class_vmax,
                                            xticklabels=['I', 'P', 'R'],
                                            yticklabels=['I', 'P', 'R'],
                                            cbar_kws={'label': 'Attention Weight'})
                                axes[idx,1].set_title(f'{class_name} Class Attention Pattern', fontsize=14)
                                axes[idx,1].tick_params(axis='both', which='major', labelsize=14)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.save_dir, f'class_attention_epoch_{epoch+1}.png'))
                        plt.close()
                        
                        # Print analysis
                        # print(f"\nEpoch {epoch + 1} Per-Class Attention Analysis:")
                        for class_name in ['I', 'P', 'R']:
                            if class_name in class_analysis:
                                analysis = class_analysis[class_name]
                                # print(f"\n{class_name} Class:")
                                # print(f"Accuracy: {analysis['accuracy']:.4f}")
                                # print(f"Average model attention: {np.mean(analysis['model_attention']['all']):.4f}")
                                # print(f"Average class attention: {np.mean(analysis['class_attention']['all']):.4f}")
                                # if analysis['model_attention']['correct'] is not None:
                                #     print("Attention patterns for correct predictions:")
                                #     print(f"Model attention: {np.mean(analysis['model_attention']['correct']):.4f}")
                                #     print(f"Class attention: {np.mean(analysis['class_attention']['correct']):.4f}")
                
                    except Exception as e:
                        print(f"Error in attention visualization: {str(e)}")
                        print("Detailed error information:")
                        import traceback
                        traceback.print_exc()
                        
                        # Print debug information
                        print("\nDebug Information:")
                        print(f"Attention layer found: {attention_layer is not None}")
                        if attention_layer:
                            print("Attention values available:")
                            values = attention_layer.get_attention_values()
                            print(f"Model attention: {'Yes' if values['model_attention'] is not None else 'No'}")
                            print(f"Class attention: {'Yes' if values['class_attention'] is not None else 'No'}")
                else:
                    print("Warning: Could not find attention layer in model")
    
    return AttentionVisualizerCallback(val_data, val_labels, save_dir)
            
from tensorflow.keras import layers, models

class BestModelAttentionCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_callback, monitor='val_weighted_accuracy', mode='max'):
        super().__init__()
        self.base_callback = base_callback
        self.monitor = monitor
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        
    def set_model(self, model):
        super().set_model(model)
        # Also set the model for the base callback
        self.base_callback.model = model
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.mode == 'max':
            improved = current > self.best
        else:
            improved = current < self.best
            
        if improved:
            self.best = current
            # Only call the attention visualization when metric improves
            self.base_callback.on_epoch_end(epoch, logs)
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        # Store the arguments as instance attributes
        self._input_dim = input_dim  # using _input_dim to avoid conflict
        self._dropout_rate = dropout_rate
        
        # Create layers in __init__
        self.dense1 = tf.keras.layers.Dense(input_dim, activation=None)
        self.dense2 = tf.keras.layers.Dense(input_dim, activation=None)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.activation = tf.keras.layers.ReLU()
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self._input_dim,
            "dropout_rate": self._dropout_rate,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=True):
        # First sub-block
        x = self.norm1(inputs)
        x = self.activation(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        
        # Second sub-block
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        
        # Add residual connection
        return inputs + x

class DualLevelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.key_dim = key_dim
        
        # Store numpy arrays directly
        self.model_attention_values = None
        self.class_attention_values = None
        self.model_attention_scores = None
        self.class_attention_scores = None
        
        # Add trainable temperature parameter
        self.temperature = tf.Variable(0.1, trainable=True, name='attention_temperature')
        
        # Add separate attention for different purposes
        self.model_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=0.1,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        
        self.class_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, #num_heads//2,  # Fewer heads for class attention or fix values becuase calsses dont change
            key_dim=32,       # Larger key dimension
            dropout=0.1,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        
        # Add gating mechanism
        self.gate = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "num_classes": self.num_classes,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def _store_attention(self, attention_weights, is_model_attention=True):
        """Convert attention weights to numpy immediately"""
        def convert_to_numpy(x):
            result = x.numpy()
            if is_model_attention:
                self.model_attention_values = result
            else:
                self.class_attention_values = result
            return x  # Return original tensor to maintain graph

        return tf.py_function(convert_to_numpy, [attention_weights], attention_weights.dtype)

    def call(self, inputs, training=True):
        # Scale inputs by learned temperature
        scaled_inputs = inputs / tf.math.maximum(self.temperature, 0.01)
        
        # Model-level attention
        model_attended, model_weights = self.model_attention(
            query=scaled_inputs,
            value=scaled_inputs,
            key=scaled_inputs,
            training=training,
            return_attention_scores=True
        )
        # Store model attention weights
        _ = self._store_attention(model_weights, is_model_attention=True)
        self.model_attention_scores = model_weights
        
        # Dynamic gating
        gate_value = self.gate(tf.reduce_mean(model_attended, axis=1, keepdims=True))
        model_attended = model_attended * gate_value
        
        # Class-level attention with different scaling
        transposed = tf.transpose(model_attended, perm=[0, 2, 1])
        class_attended, class_weights = self.class_attention(
            query=transposed,
            value=transposed,
            key=transposed,
            training=training,
            return_attention_scores=True
        )
        # Store class attention weights
        _ = self._store_attention(class_weights, is_model_attention=False)
        self.class_attention_scores = class_weights
        
        output = tf.transpose(class_attended, perm=[0, 2, 1])
        
        # Final processing
        return tf.reduce_mean(output, axis=1)
    def get_attention_scores(self):
        """Check if attention scores are available"""
        if self.model_attention_scores is None or self.class_attention_scores is None:
            print("Warning: One or both attention scores are None")
            if self.model_attention_scores is None:
                print("Model attention scores are None")
            if self.class_attention_scores is None:
                print("Class attention scores are None")
            return {'model_attention': None, 'class_attention': None}
        
        return {
            'model_attention': self.model_attention_scores,
            'class_attention': self.class_attention_scores
        }
    def get_attention_values(self):
        """Return stored numpy attention values"""
        return {
            'model_attention': self.model_attention_values,
            'class_attention': self.class_attention_values
        }

def attention_entropy_loss(attention_scores):
    def calculate_entropy(scores):
        # Normalize scores
        scores = tf.nn.softmax(scores, axis=-1)
        epsilon = 1e-10
        scores = tf.clip_by_value(scores, epsilon, 1.0)
        
        # Calculate entropy
        entropy = -tf.reduce_sum(scores * tf.math.log(scores), axis=-1)
        
        # Calculate standard deviation manually
        mean = tf.reduce_mean(scores, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(scores - mean), axis=-1)
        diversity = tf.sqrt(variance + epsilon)  # Add epsilon to avoid sqrt(0)
        
        return tf.reduce_mean(entropy) + tf.reduce_mean(diversity)
    
    model_entropy = calculate_entropy(attention_scores['model_attention'])
    class_entropy = calculate_entropy(attention_scores['class_attention'])
    
    # Balance between model and class attention
    return 0.7 * model_entropy + 0.3 * class_entropy

def custom_loss(model):
    def loss_function(y_true, y_pred):
        # Base classification loss
        base_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        )
        
        attention_layer = next((layer for layer in model.layers 
                              if isinstance(layer, DualLevelAttentionLayer)), None)
        
        if attention_layer is not None:
            attention_scores = attention_layer.get_attention_scores()
            if all(v is not None for v in attention_scores.values()):
                entropy_loss = attention_entropy_loss(attention_scores)
                
                # Dynamic entropy weight based on training progress
                entropy_weight = 0.2 * (1.0 - tf.exp(-base_loss))
                
                total_loss = base_loss - entropy_weight * entropy_loss
                
                # tf.print("Loss components:", {
                #     'base_loss': base_loss,
                #     'entropy_weight': entropy_weight,
                #     'entropy_loss': entropy_loss,
                #     'total_loss': total_loss,
                #     'temperature': attention_layer.temperature
                # })
                return total_loss
        
        return base_loss
    
    return loss_function
class DynamicLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, 
                 initial_lr=1e-3,
                 min_lr=1e-14,
                 exploration_epochs=10,
                 cycle_length=30,
                 cycle_multiplier=2.0):
        super(DynamicLRSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.exploration_epochs = exploration_epochs
        self.cycle_length = cycle_length
        self.cycle_multiplier = cycle_multiplier
        
        # Internal state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.current_cycle = 0
        self.last_restart = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.exploration_epochs:
            # Exploration phase: higher learning rate
            lr = self.initial_lr
        else:
            # Calculate position in current cycle
            cycle_epoch = epoch - self.last_restart
            if cycle_epoch >= self.cycle_length:
                # Start new cycle
                self.current_cycle += 1
                self.last_restart = epoch
                cycle_epoch = 0
                # Increase cycle length
                self.cycle_length = int(self.cycle_length * self.cycle_multiplier)
            
            # Cosine annealing within cycle
            progress = cycle_epoch / self.cycle_length
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * progress))
            
            # Calculate learning rate
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
            
            # Adjust based on loss improvement
            if logs and 'loss' in logs:
                current_loss = logs['loss']
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    # If loss hasn't improved for a while, reduce max learning rate
                    if self.patience_counter >= 5:
                        self.initial_lr *= 0.8
                        self.patience_counter = 0
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Log the learning rate for monitoring
        if logs is None:
            logs = {}
        logs['lr'] = lr
def ImprovedGatingNetwork(num_models=16, num_classes=3):
    input_layer = tf.keras.layers.Input(shape=(num_models, num_classes))
    
    # Main attention and processing
    dual_attention_output = DualLevelAttentionLayer(
        num_heads=max(8, num_models+1), #16 for 15 modesl (best value)
        key_dim=max(16, 2*(num_models+1)), #32 for 15 modesl (best value)
        num_classes=num_classes
    )(input_layer)
    
    # Final residual block and prediction
    final_residual = ResidualBlock(num_classes)(dual_attention_output)
    predictions = tf.nn.softmax(final_residual)  # Direct softmax since dimensions already match
    
    return tf.keras.Model(inputs=input_layer, outputs=predictions)
def train_model_combination(train_data, val_data, train_labels, val_labels):
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # print(f"\nTraining model combination with {len(train_data[0])} models")
    if len(train_labels.shape) == 1:
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=3)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=3)
    
    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(train_labels, axis=1)), y=np.argmax(train_labels, axis=1))
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    # print(f"Class weights for Gating Network: {class_weights_dict}")
    tf.keras.backend.clear_session()
    counts = Counter(np.argmax(train_labels, axis=1))
    total_samples = sum(counts.values())
    class_frequencies = {cls: count/total_samples for cls, count in counts.items()}
    median_freq = np.median(list(class_frequencies.values()))
    alpha_values = [median_freq/class_frequencies[i] for i in [0, 1, 2]]  # Keep ordered
    alpha_sum = sum(alpha_values)
    alpha_values = [alpha/alpha_sum * 3.0 for alpha in alpha_values]


    weighted_acc = WeightedAccuracy(alpha_values=class_weights, name='weighted_accuracy')
    
    # with strategy.scope():
    # Build the model
    model = ImprovedGatingNetwork(len(train_data[0]), 3)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=custom_loss(model),
        metrics=['accuracy', weighted_acc]
    )
    # # Create distributed datasets #TODO Distribute the second training for multigpu processing
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    # # Set proper batch size
    # global_batch_size = batch_size * strategy.num_replicas_in_sync
    # train_dataset = train_dataset.batch(global_batch_size)
    # val_dataset = val_dataset.batch(global_batch_size)
    
    # # Distribute datasets
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    # Setup callbacks
    callbacks2 = [
        # DynamicLRSchedule(
        #     initial_lr=1e-4,
        #     min_lr=1e-12,
        #     exploration_epochs=10,
        #     cycle_length=30,
        #     cycle_multiplier=2.0
        # ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-9,
            min_delta=2e-3,
            verbose=0,
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_weighted_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=2,
            min_delta=2e-2,
            mode='max'
        )
    ]
    # Create a custom training history object that includes learning rate
    class CustomHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None and hasattr(self.model.optimizer, 'lr'):
                logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
    callbacks2.append(create_attention_visualization_callback(val_data=val_data, val_labels=val_labels, save_dir=os.path.join(result_dir, 'attention_analysis')))
    # Add custom history to callbacks
    callbacks2.append(CustomHistory())
    
    # Add deterministic data handling
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    dataset = dataset.batch(64, drop_remainder=True)  # Fixed batch size
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(len(val_labels), drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Train the model
    history = model.fit(
        dataset,
        class_weight=class_weights_dict,
        epochs=1000,
        # batch_size=64, Only use when using train_data and val_data directly
        # batch_size=len(val_data), # do this ONLY when running the create_attention_visualization_callback, permorance may get affected when different than 64
        validation_data=val_dataset,
        callbacks=callbacks2,
        shuffle=True,
        verbose=0
    )

    # Predict outputs
    predictions = model.predict(train_data, verbose=0)
    val_predictions = model.predict(val_data, verbose=0)
    
    # Calculate validation accuracy
    accuracy = np.mean(np.argmax(val_predictions, axis=1) == np.argmax(val_labels, axis=1))
    kappa = cohen_kappa_score(np.argmax(val_labels, axis=1), np.argmax(val_predictions, axis=1), weights='quadratic')
    val_weighted_acc = np.max(history.history['val_weighted_accuracy'])
    # print(f"Final predictions shape: {predictions.shape}")
    # print(f"Validation accuracy: {accuracy:.4f}, Cohen Kappa Score: {kappa:.4f}")

    return val_predictions, history.history['loss'][-1], accuracy, kappa, model, val_weighted_acc
import json
from multiprocessing import Pool
from functools import partial
# Create a function that processes a single trial
def process_trial(trial, all_combinations, completed_combinations, train_predictions_list, 
                 valid_predictions_list, train_labels, valid_labels, run_number, max_tries_min):
    selected_indices = all_combinations[trial]
    selected_key = tuple(sorted(selected_indices))
    current_model_count = len(selected_indices)
    
    # Count how many combinations already tried for this number of models
    existing_combinations_count = sum(1 for combo in completed_combinations if len(combo) == current_model_count)
    remaining_tries = max_tries_min - existing_combinations_count
    # If already completed all tries for this model count, skip
    if remaining_tries <= 0:
        return None
    # If this specific combination was already tried, skip
    if selected_key in completed_combinations:
        return None
    # Keep track of how many new combinations processed for this model count
    new_combinations_count = sum(1 for combo in completed_combinations if len(combo) == current_model_count and combo not in completed_combinations)     
    # Skip if processed enough new combinations to reach max_tries_min
    if new_combinations_count >= remaining_tries:
        return None
    
    train_predictions_selected = [train_predictions_list[i] for i in selected_indices]
    valid_predictions_selected = [valid_predictions_list[i] for i in selected_indices]
    
    try:
        train_data_truncated, train_labels_truncated = correct_and_validate_predictions(
            train_predictions_selected, train_labels, "train")
        val_data_truncated, val_labels_truncated = correct_and_validate_predictions(
            valid_predictions_selected, valid_labels, "valid")
        
        train_data_sliced = np.stack(train_data_truncated, axis=1).astype(np.float32)
        val_data_sliced = np.stack(val_data_truncated, axis=1).astype(np.float32)
        
        predictions, loss, accuracy, kappa, _, val_weighted_acc = train_model_combination(train_data_sliced, val_data_sliced, train_labels_truncated, val_labels_truncated)
        
        save_progress(run_number, completed_combination=selected_key)
        
        return {
            'selected_indices': selected_indices,
            'predictions': predictions,
            'loss': loss,
            'accuracy': accuracy,
            'kappa': kappa,
            'weighted_accuracy': val_weighted_acc,
            'selected_key': selected_key
        }
    except Exception as e:
        print(f"Error in trial {trial}: {str(e)}")
        return None
def train_gating_network(train_predictions_list, valid_predictions_list, train_labels, valid_labels, run_number, find_optimal=True, min_models=2, max_tries=100):
    tf.config.experimental.enable_op_determinism()
    print("\nInitializing gating network training...")
    print(f"Number of models: {len(train_predictions_list)}")
    print(f"Shape of first model predictions: {train_predictions_list[0].shape}")
    print(f"Shape of true labels: {train_labels.shape}")
    excluded_models = set()
    excluded_temp = set()
    
    # Load previous progress
    progress, best_predictions = load_progress(run_number)
    best_accuracy = progress['best_metrics']['accuracy']
    best_weighted_accuracy = progress['best_metrics']['weighted_accuracy']
    best_loss = progress['best_metrics']['loss']
    best_kappa = progress['best_metrics']['kappa']
    best_combination = progress.get('best_combination', None)
    completed_combinations = set(tuple(comb) for comb in progress['completed_combinations'])
    if best_combination:
        excluded_models = set(range(len(train_predictions_list))) - set(best_combination)
        print(f"Excluded models from previous time code was run: {excluded_models}")
    
    # Ensure predictions are in float32
    train_predictions_list = [np.array(p, dtype=np.float32) for p in train_predictions_list]
    valid_predictions_list = [np.array(p, dtype=np.float32) for p in valid_predictions_list]
    
    # Train with all models first (only if not done before)
    all_models_key = tuple(range(len(train_predictions_list)))
    if all_models_key not in completed_combinations:
        print("\nTraining with all models...")
        train_data_truncated, train_labels_truncated = correct_and_validate_predictions(train_predictions_list, train_labels, "train")
        val_data_truncated, val_labels_truncated = correct_and_validate_predictions(valid_predictions_list, valid_labels, "valid")
        
        train_data = np.stack(train_data_truncated, axis=1).astype(np.float32)
        val_data = np.stack(val_data_truncated, axis=1).astype(np.float32)
        
        predictions, loss, accuracy, kappa, model, val_weighted_acc = train_model_combination(train_data, val_data, train_labels_truncated, val_labels_truncated) 
        if best_accuracy and accuracy+val_weighted_acc >= best_accuracy+best_weighted_accuracy:
            best_accuracy = accuracy
            best_weighted_accuracy = val_weighted_acc
            best_loss = loss
            best_kappa = kappa
            best_predictions = predictions
            best_combination = list(range(len(train_predictions_list)))
            best_model = model
            best_model.save(os.path.join(result_dir, f'best_gating_model_run{run_number}.h5'))
        elif not best_accuracy:
            best_accuracy = accuracy
            best_weighted_accuracy = val_weighted_acc
            best_loss = loss
            best_kappa = kappa
            best_predictions = predictions
            best_combination = list(range(len(train_predictions_list)))
            best_model = model
            best_model.save(os.path.join(result_dir, f'best_gating_model_run{run_number}.h5'))
            
        # Save progress
        save_progress(run_number, completed_combination=all_models_key,best_metrics={'accuracy': best_accuracy, 'weighted_accuracy': best_weighted_accuracy ,'loss': best_loss, 'kappa': best_kappa},best_combination=best_combination,best_predictions=best_predictions)
        print(f"Initial Accuracy: {best_accuracy:.4f}, Initial Weighted Accuracy: {best_weighted_accuracy:.4f}, Loss: {best_loss:.4f}, kappa: {best_kappa:.4f}")
        write_save_best_combo_results(best_predictions, best_accuracy, best_weighted_accuracy, best_loss, valid_labels, best_combination, excluded_models, result_dir, run_number)
    
    # Search for the optimal combination of models
    if find_optimal:
        print("\nSearching for the optimal model combination...")
        available_models = set(range(len(train_predictions_list)))
        # excluded_models = set()
        available_indices = list(available_models)
        # Manually exclude models
        for i in [800000]:
            if i in available_indices:
                available_indices.remove(i)
        # num_models = len(available_indices) - 1 - 0 #- 5  # Start with all models except 6 !!!!!!!! Hard CODED CHANGE
        num_models = len(available_indices) - 1 - 0 #- 5  # Start with all models except 6 !!!!!!!! Hard CODED CHANGE
        if num_models == 2:
            steps = [2]
        else:
            steps = list(np.sort(range(min_models, num_models, max(int(0.20*(num_models)), 1)))) + [len(available_indices) - 1]
        print(f"Explornig Steps: {steps}")
        while num_models >= min_models:
            # Progressively exclude models
            # available_indices = list(available_models - excluded_models)

            if len(available_indices) < num_models:
                if steps and len(steps) > 1:
                    steps = steps[:-1]
                num_models = len(available_indices)  # Adjust num_models to match available models
                if num_models < min_models:
                    break
                continue
            print(f"\nTrying combinations with {num_models} models...")
            print(f"Available indices: {available_indices}")
            from itertools import combinations
            import random
            all_combinations = list(combinations(available_indices, num_models))
            random.shuffle(all_combinations)
            max_tries_min = min(max_tries, len(all_combinations))
            print(f"Max tries: {max_tries_min}")
            
            from tqdm import tqdm
            
            num_processes = min(3, os.cpu_count() or 1)  # Use at most 6 cores
            with Pool(processes=num_processes) as pool:
                process_func = partial(process_trial, 
                                    all_combinations=all_combinations,
                                    completed_combinations=completed_combinations,
                                    train_predictions_list=train_predictions_list,
                                    valid_predictions_list=valid_predictions_list,
                                    train_labels=train_labels,
                                    valid_labels=valid_labels,
                                    run_number=run_number,
                                    max_tries_min=max_tries_min,
                                    )
                
                for result in tqdm(pool.imap(process_func, range(max_tries_min)), 
                                total=max_tries_min,
                                desc=f"Trying combinations with {num_models} models"):
                    if result is not None:
                        # If this is better than our best so far
                        if best_accuracy and result['accuracy']+result['weighted_accuracy'] >= best_accuracy+best_weighted_accuracy:
                            # Retrain the model with these indices (to get a fresh model)
                            train_predictions_selected = [train_predictions_list[i] for i in result['selected_indices']]
                            valid_predictions_selected = [valid_predictions_list[i] for i in result['selected_indices']]
                            
                            train_data_truncated, train_labels_truncated = correct_and_validate_predictions(
                                train_predictions_selected, train_labels, "train")
                            val_data_truncated, val_labels_truncated = correct_and_validate_predictions(
                                valid_predictions_selected, valid_labels, "valid")
                            
                            train_data_sliced = np.stack(train_data_truncated, axis=1).astype(np.float32)
                            val_data_sliced = np.stack(val_data_truncated, axis=1).astype(np.float32)
                            
                            # Retrain to get the model
                            _, _, _, _, best_model, _ = train_model_combination(train_data_sliced, val_data_sliced, train_labels_truncated, val_labels_truncated)
                            
                            excluded_temp = excluded_models.copy()
                            print(f"\nNew best combination found!")
                            print(f"Models {result['selected_indices']}")
                            
                            excluded_model = set(available_indices) - set(result['selected_indices'])
                            if excluded_model:
                                print(f"Excluded trial model(s) {excluded_model}")
                                excluded_temp.update(excluded_model)
                            print(f"Previous accuracy: {best_accuracy:.4f}, Previous Weighted Accuracy: {best_weighted_accuracy:.4f}, Previous loss: {best_loss:.4f}, Previous kappa: {best_kappa:.4f}")
                            
                            best_accuracy = result['accuracy']
                            best_weighted_accuracy = result['weighted_accuracy']
                            best_loss = result['loss']
                            best_predictions = result['predictions']
                            best_combination = result['selected_indices']
                            best_kappa = result['kappa']
                            
                            print(f"New best accuracy: {best_accuracy:.4f}, New best Weighted Accuracy: {best_weighted_accuracy:.4f}, New best loss: {best_loss:.4f}, New best kappa: {best_kappa:.4f}")
                            
                            write_save_best_combo_results(best_predictions, best_accuracy, best_weighted_accuracy, best_loss, valid_labels, best_combination, excluded_temp, result_dir, run_number)
                            save_progress(run_number, best_metrics={'accuracy': best_accuracy, 'weighted_accuracy': best_weighted_accuracy, 'loss': best_loss, 'kappa': best_kappa}, best_combination=best_combination,best_predictions=best_predictions)
                            best_model.save(os.path.join(result_dir, f'best_gating_model_run{run_number}.h5'))
            if excluded_temp:
                excluded_models = excluded_temp.copy()
            print(f"Excluding model(s) {excluded_models} from future combinations")
            print(f"{max_tries_min} combinations with {num_models} models completed")
            # num_models -= 1
            if len(steps) == 1:
                break
            elif steps and len(steps) > 1:
                steps = steps[:-1]
                num_models = steps[-1]
            else:
                break
    # Final Results
    print("\nFinal Results:")
    print("==============")
    print(f"Best Cohen's Kappa: {best_kappa:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Weighted Accuracy: {best_weighted_accuracy:.4f}")
    print(f"Best Loss: {best_loss:.4f}")
    # if best_combination:
    #     print(f"Best Model Combination: {best_combination}")
    #     if excluded_models:
    #         print(f"Excluded Models: {excluded_models}")
    #         write_save_best_combo_results(best_predictions, best_accuracy, best_loss, val_labels, best_combination, excluded_models, result_dir, run_number)

    return best_predictions, valid_labels

def load_progress(run_number):
    """Load progress from JSON file with error handling"""
    import time
    progress_file = os.path.join(result_dir, f'training_progress_run{run_number}.json')
    predictions_file = os.path.join(result_dir, f'best_predictions_run{run_number}.npy')
    
    # Initialize default progress - note: no predictions here
    default_progress = {
        'completed_combinations': [],
        'best_metrics': {
            'accuracy': 0,
            'weighted_accuracy': 0,
            'loss': float('inf'),
            'kappa': 0
        },
        'best_combination': None
    }
    
    predictions = None
    max_retries = 6
    retry_delay = 0.4
    
    # Load progress dict if exists
    for attempt in range(max_retries):
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    loaded_progress = json.load(f)
                    default_progress.update(loaded_progress)
                break
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print(f"Error loading progress file: {e}")
            print("Using default progress values")
    
    # Try to load predictions file
    if os.path.exists(predictions_file):
        try:
            predictions = np.load(predictions_file)
        except Exception as e:
            print(f"Error loading predictions file: {e}")
    
    return default_progress, predictions

def save_progress(run_number, completed_combination=None, best_metrics=None, best_combination=None, best_predictions=None):
    """Save progress to JSON file with proper separation of numpy arrays"""
    import time
    progress_file = os.path.join(result_dir, f'training_progress_run{run_number}.json')
    predictions_file = os.path.join(result_dir, f'best_predictions_run{run_number}.npy')
    max_retries = 6
    retry_delay = 0.4
    
    for attempt in range(max_retries):
        try:
            # Load existing progress (but not predictions)
            progress, _ = load_progress(run_number)
            
            if completed_combination is not None:
                # Convert numpy types or tuples to list of integers
                if isinstance(completed_combination, (np.ndarray, tuple)):
                    completed_combination = [int(x) for x in completed_combination]
                    
                # Ensure the combination isn't already saved
                if completed_combination not in [tuple(x) for x in progress['completed_combinations']]:
                    progress['completed_combinations'].append(completed_combination)
            
            if best_metrics:
                # Ensure all metric values are Python native types
                progress['best_metrics'] = {
                    'accuracy': float(best_metrics['accuracy']),
                    'weighted_accuracy': float(best_metrics['weighted_accuracy']),
                    'loss': float(best_metrics['loss']),
                    'kappa': float(best_metrics['kappa'])
                }
            
            if best_combination is not None:
                # Convert numpy array or other sequence types to list of integers
                if isinstance(best_combination, (np.ndarray, tuple)):
                    best_combination = [int(x) for x in best_combination]
                elif isinstance(best_combination, list):
                    best_combination = [int(x) for x in best_combination]
                progress['best_combination'] = best_combination
            
            # Save progress dict (without predictions)
            try:
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
            except TypeError as e:
                print(f"Error during JSON serialization: {e}")
                print("Progress content:", progress)
                raise
                
            # Save predictions separately only if provided
            if best_predictions is not None:
                try:
                    np.save(predictions_file, best_predictions)
                except Exception as e:
                    print(f"Error saving predictions: {e}")
            break  
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print(f"Error saving progress: {e}")
            
def write_save_best_combo_results(best_predictions, best_accuracy, best_weighted_accuracy, best_loss, true_labels, selected_models, excluded_models, result_dir, run_number): 
    # Calculate final metrics 
    y_true = true_labels 
    y_pred = np.argmax(best_predictions, axis=1) 

    # Calculate metrics
    f1_class_0 = f1_score(y_true, y_pred, average=None, zero_division=0)[0] 
    f1_class_1 = f1_score(y_true, y_pred, average=None, zero_division=0)[1] 
    f1_class_2 = f1_score(y_true, y_pred, average=None, zero_division=0)[2] 
    f1_macro = f1_score(y_true, y_pred, average='macro') 
    f1_weighted = f1_score(y_true, y_pred, average='weighted') 
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic') 

    # Save results to file 
    results_file = os.path.join(result_dir, f'model_combination_results_{run_number}.txt') 
    with open(results_file, 'w') as f: 
        f.write("Model Training Results\n") 
        f.write("====================\n\n") 
         
        # Overall Metrics 
        f.write("Overall Metrics:\n") 
        f.write("--------------\n") 
        f.write(f"Final Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Final Weighted Accuracy: {best_weighted_accuracy:.4f}\n")
        f.write(f"Final Loss: {best_loss:.4f}\n") 
        f.write(f"Cohen's Kappa: {kappa:.4f}\n\n") 
         
        # F1 Scores 
        f.write("F1 Scores:\n") 
        f.write("----------\n") 
        f.write(f"Class 0 (Inflammatory): {f1_class_0:.4f}\n") 
        f.write(f"Class 1 (Proliferative): {f1_class_1:.4f}\n") 
        f.write(f"Class 2 (Remodeling): {f1_class_2:.4f}\n") 
        f.write(f"Macro Average F1: {f1_macro:.4f}\n") 
        f.write(f"Weighted Average F1: {f1_weighted:.4f}\n\n") 
         
        # Model Information 
        f.write("Model Information:\n") 
        f.write("-----------------\n") 
        f.write(f"Selected Models: {selected_models}\n\n")
        if excluded_models:
            f.write(f"Excluded Models: {excluded_models}\n\n")
         
        # Additional Details 
        f.write("\nDetailed Metrics:\n") 
        f.write("---------------\n") 
        f.write("Confusion Matrix:\n") 
        f.write(str(confusion_matrix(y_true, y_pred))) 
        f.write("\n\nClassification Report:\n") 
        f.write(str(classification_report(y_true, y_pred))) 

    print(f"\nDetailed results saved to {results_file}")
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn1 = Dense(ff_dim, activation="gelu")  # First Dense layer
        self.ffn2 = Dense(embed_dim)  # Second Dense layer
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn1(out1)  # Apply the first Dense layer
        ffn_output = self.ffn2(ffn_output)  # Apply the second Dense layer
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_hierarchical_gating_network(num_models, num_classes, embedding_dim=32):
    """
    Creates a hierarchical attention network for combining model predictions
    
    Args:
        num_models: Number of base models
        num_classes: Number of classification classes (3 for I,P,R)
        embedding_dim: Dimension of the embedding space
    """
    # Input shape: (batch_size, num_models, num_classes)
    inputs = Input(shape=(num_models, num_classes))
    
    # Phase-specific attention branch
    phase_specific_outputs = []
    
    for phase in range(num_classes):
        # Extract phase-specific probabilities
        phase_inputs = tf.keras.layers.Lambda(lambda x: x[:, :, phase])(inputs)
        phase_inputs = Reshape((num_models, 1))(phase_inputs)
        
        # Learn phase-specific attention
        attention = Dense(embedding_dim, activation='relu')(phase_inputs)
        attention = LayerNormalization()(attention)
        attention = Dense(1, activation='softmax')(attention)
        
        # Weight model predictions for this phase
        weighted_phase = Multiply()([phase_inputs, attention])
        weighted_phase = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(weighted_phase)
        
        phase_specific_outputs.append(weighted_phase)
    
    # Combine phase-specific outputs
    phase_concatenated = Concatenate(axis=-1)(phase_specific_outputs)
    
    # Add positional encoding for transformer
    pos_encoding = tf.cast(tf.range(num_classes), tf.float32)[tf.newaxis, :]
    pos_encoding = Dense(embedding_dim)(pos_encoding[:, :, tf.newaxis])
    
    # Project to embedding space
    phase_embedded = Dense(embedding_dim)(phase_concatenated)
    phase_embedded = Add()([phase_embedded, pos_encoding])
    
    # Transformer for inter-phase relationships
    transformer_output = TransformerBlock(
        embed_dim=embedding_dim,
        num_heads=2,
        ff_dim=embedding_dim * 2
    )(phase_embedded)
    
    # Final classification layer with residual connection
    transformer_pooled = GlobalAveragePooling1D()(transformer_output)
    residual = GlobalAveragePooling1D()(phase_embedded)
    
    combined = Concatenate()([transformer_pooled, residual])
    combined = Dense(embedding_dim, activation='relu')(combined)
    combined = LayerNormalization()(combined)
    combined = Dropout(0.1)(combined)
    
    # Output probabilities
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined) ##TRY THIS TO AVOID OVERFITTING
    # outputs = Dense(num_classes, activation='softmax')(combined)
    
    return Model(inputs=inputs, outputs=outputs)
def focal_loss_gating_network(gamma=2.0, alpha=None):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        if alpha is None:
            alpha_factor = 1
        else:
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            
        focusing_factor = y_true * K.pow(1. - y_pred, gamma) + \
                         (1. - y_true) * K.pow(y_pred, gamma)
                         
        return K.mean(alpha_factor * focusing_factor * K.binary_crossentropy(y_true, y_pred))
    return focal_loss_fixed
def train_hierarchical_gating_network(predictions_list, true_labels, n_splits=3, patience=20):
    """
    Train the hierarchical gating network using cross-validation
    
    Args:
        predictions_list: List of model predictions arrays
        true_labels: True class labels (one-hot encoded)
        n_splits: Number of cross-validation splits
        patience: Early stopping patience
    """
    print("\nTraining hierarchical gating network with cross-validation...")
    
    # Basic validation checks
    min_length = min(len(preds) for preds in predictions_list)
    if (true_labels.shape[1] != predictions_list[0].shape[1] or min_length < 6):
        print("Warning: Class mismatch or dataset too small. Using simple averaging.")
        combined_predictions = np.mean([preds[:min_length] for preds in predictions_list], axis=0)
        attention_weights = np.ones(len(predictions_list)) / len(predictions_list)
        print(f"Using equal weights for all models: {attention_weights}")
        return combined_predictions, attention_weights

    # Prepare data
    predictions_list = [preds[:min_length] for preds in predictions_list]
    true_labels = true_labels[:min_length]
    
    
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize arrays for predictions and metrics
    combined_predictions = np.zeros_like(true_labels)
    fold_attention_weights = []
    fold_metrics = []
    
    # Convert predictions to numpy arrays and stack
    stacked_predictions = np.stack(predictions_list, axis=1)
    
    # Cross-validation loop
    # for fold, (train_idx, val_idx) in enumerate(kf.split(stacked_predictions)):
    for fold, (train_idx, val_idx) in enumerate(skf.split(stacked_predictions, np.argmax(true_labels, axis=1))):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        X_train = stacked_predictions[train_idx]
        y_train = true_labels[train_idx]
        X_val = stacked_predictions[val_idx]
        y_val = true_labels[val_idx]
        
        # Create and compile model
        model = create_hierarchical_gating_network(
            num_models=len(predictions_list),
            num_classes=true_labels.shape[1],
            embedding_dim=32
        )
        class_weights = {
            0: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 0)),  # I
            1: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 1)),  # P
            2: len(true_labels) / (3 * np.sum(np.argmax(true_labels, axis=1) == 2))   # R
        }
        # alpha = list(class_weights.values())
        loss = focal_loss_gating_network(gamma=3.0, alpha=None)
        # loss = create_focal_ordinal_loss_with_params(ordinal_weight=1.5, gamma=3.0, alpha=[0.598, 0.315, 1.597])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=loss,
            metrics=['accuracy'],
        )
        
        # Train with early stopping
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            # class_weight=class_weights,
            epochs=500,
            batch_size=32,
            callbacks=[
                EarlyStopping(
                    monitor='loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-12
                )
            ],
            verbose=2
        )
        
        # Get predictions for validation fold
        combined_predictions[val_idx] = model.predict(X_val, verbose=0)
        
        # Store metrics
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        fold_metrics.append({'val_loss': val_loss, 'val_accuracy': val_acc})
        
        # Extract attention weights for analysis
        attention_model = Model(
            inputs=model.input,
            outputs=model.layers[1].output  # First attention layer
        )
        attention_weights = attention_model.predict(X_val, verbose=0)
        fold_attention_weights.append(np.mean(attention_weights, axis=0))
        
        print(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}")
    
    # Calculate and print overall metrics
    print("\nOverall cross-validation metrics:")
    mean_val_acc = np.mean([m['val_accuracy'] for m in fold_metrics])
    std_val_acc = np.std([m['val_accuracy'] for m in fold_metrics])
    print(f"Validation Accuracy: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    
    # Calculate average attention weights across folds
    overall_attention = np.mean(fold_attention_weights, axis=0)
    print("\nOverall model importance weights:")
    for i, weight in enumerate(overall_attention):
        print(f"Model {i + 1}: {np.mean(weight):.3f}")
    
    return combined_predictions, overall_attention

def calculate_and_save_metrics(true_labels, predictions, result_dir, configs):
    """Calculate metrics and save results."""
    if len(true_labels) != len(predictions):
        print(f"Warning: Length mismatch - true_labels: {len(true_labels)}, predictions: {len(predictions)}")
        # Truncate to shorter length
        min_len = min(len(true_labels), len(predictions))
        true_labels = true_labels[:min_len]
        predictions = predictions[:min_len]
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted', zero_division=0),
        'f1_classes': f1_score(true_labels, predictions, average=None, labels=[0, 1, 2], zero_division=0),
        'kappa': cohen_kappa_score(true_labels, predictions, weights='quadratic')
    }
    
    # Print results
    print("\nFinal Results:")
    print(classification_report(true_labels, predictions,
                              target_names=CLASS_LABELS,
                              labels=[0, 1, 2]))
    print(f"\nCohen's Kappa: {metrics['kappa']:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix with Specialized Modalities')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(result_dir, 'specialized_confusion_matrix.png'))
    plt.close()
    
    # Save to CSV
    results = {
        # 'Modality Configuration': f"Specialized (I:{'+'.join(configs['I']['modalities'])}, P:{'+'.join(configs['P']['modalities'])}, R:{'+'.join(configs['R']['modalities'])}, I2:{'+'.join(configs['I2']['modalities'])}, P2:{'+'.join(configs['P2']['modalities'])}, R2:{'+'.join(configs['R2']['modalities'])}, I3:{'+'.join(configs['I3']['modalities'])}, P3:{'+'.join(configs['P3']['modalities'])}, R3:{'+'.join(configs['R3']['modalities'])}, I4:{'+'.join(configs['I4']['modalities'])}, P4:{'+'.join(configs['P4']['modalities'])}, R4:{'+'.join(configs['R4']['modalities'])})",
        'Modality Configuration': "Specialized",
        'Accuracy': metrics['accuracy'],
        'Macro F1-score': metrics['f1_macro'],
        'Weighted F1-score': metrics['f1_weighted'],
        'I F1-score': metrics['f1_classes'][0],
        'P F1-score': metrics['f1_classes'][1],
        'R F1-score': metrics['f1_classes'][2],
        "Cohen's Kappa": metrics['kappa'],
        # 'I gamma': configs['I']['loss_params']['gamma'],
        # 'P gamma': configs['P']['loss_params']['gamma'],
        # 'R gamma': configs['R']['loss_params']['gamma'],
        # 'I ordinal weight': configs['I']['loss_params']['ordinal_weight'],
        # 'P ordinal weight': configs['P']['loss_params']['ordinal_weight'],
        # 'R ordinal weight': configs['R']['loss_params']['ordinal_weight']
    }
    
    csv_filename = os.path.join(result_dir, 'specialized_results_V66_augment_RGBGenAug.csv')
    fieldnames = list(results.keys())
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results)
    
    return metrics
def create_focal_ordinal_loss2(ordinal_weight, gamma, alpha):
    """Create a focal ordinal loss function with fixed parameters"""
    print("\nLoss Function Configuration for depth_rgb:")
    print(f"Ordinal weight: {ordinal_weight}")
    print(f"Gamma: {gamma}")
    print(f"Alpha: {alpha}")
    def loss_fn(y_true, y_pred):
        if not hasattr(loss_fn, 'params_verified'):
            print("\nVerifying loss parameters during first call:")
            print(f"Current gamma: {gamma}")
            print(f"Current alpha: {alpha}")
            loss_fn.params_verified = True
        # Clip prediction values to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Convert alpha to tensor with proper shape
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        
        # Focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.expand_dims(alpha_tensor, 0) * tf.math.pow(1 - y_pred, gamma)
        focal_loss = focal_weight * cross_entropy
        
        # Sum over classes
        focal_loss = tf.reduce_sum(focal_loss, axis=-1)
        
        # Ordinal penalty
        true_class = tf.argmax(y_true, axis=-1)
        pred_class = tf.argmax(y_pred, axis=-1)
        ordinal_penalty = tf.square(tf.cast(true_class, tf.float32) - tf.cast(pred_class, tf.float32))
        
        # Combine losses
        total_loss = focal_loss + ordinal_weight * ordinal_penalty
        return total_loss
    
    return loss_fn

def create_current_get_focal_ordinal_loss(ordinal_weight, gamma, alpha):
    """Create a wrapper function that matches the expected signature"""
    def current_get_focal_ordinal_loss(**kwargs):
        return create_focal_ordinal_loss2(ordinal_weight, gamma, alpha)
    return current_get_focal_ordinal_loss

def perform_grid_search(data_percentage=100, train_patient_percentage=0.8, n_runs=3):
    """
    Perform grid search over loss function parameters.
    """
    # Define parameter grids
    param_grid = {
        'ordinal_weight': [1.0],
        'gamma': [2.0, 3.0],
        'alpha': [
            [0.598, 0.315, 1.597],  # Your current weights
            [1, 0.5, 2],  # Alternative weights
            [1.5, 0.3, 1.5]  # Another alternative
        ]
    }
    
    # Create results CSV
    results_file = os.path.join(result_dir, 'loss_parameter_search_results.csv')
    fieldnames = ['ordinal_weight', 'gamma', 'alpha', 
                  'accuracy', 'f1_macro', 'f1_weighted',
                  'f1_I', 'f1_P', 'f1_R', 'kappa']
    
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    best_score = 0
    best_params = None
    
    # Store original function
    original_get_focal_ordinal_loss = globals()['get_focal_ordinal_loss']
    
    # Iterate over parameter combinations
    for ordinal_weight in param_grid['ordinal_weight']:
        for gamma in param_grid['gamma']:
            for alpha in param_grid['alpha']:
                print(f"\nTesting parameters:")
                print(f"Ordinal Weight: {ordinal_weight}")
                print(f"Gamma: {gamma}")
                print(f"Alpha: {alpha}")
                
                try:
                    # Create and set the current loss function
                    current_get_focal_ordinal_loss = create_current_get_focal_ordinal_loss(
                        ordinal_weight, gamma, alpha
                    )
                    globals()['get_focal_ordinal_loss'] = current_get_focal_ordinal_loss
                    
                    # Run your existing main function
                    metrics = main_with_specialized_evaluation(
                        data_percentage=data_percentage,
                        train_patient_percentage=train_patient_percentage,
                        n_runs=n_runs
                    )
                    
                    # Prepare results
                    result = {
                        'ordinal_weight': ordinal_weight,
                        'gamma': gamma,
                        'alpha': str(alpha),
                        'accuracy': metrics.get('accuracy', 0),
                        'f1_macro': metrics.get('f1_macro', 0),
                        'f1_weighted': metrics.get('f1_weighted', 0),
                        'f1_I': metrics.get('f1_classes', [0, 0, 0])[0],
                        'f1_P': metrics.get('f1_classes', [0, 0, 0])[1],
                        'f1_R': metrics.get('f1_classes', [0, 0, 0])[2],
                        'kappa': metrics.get('kappa', 0)
                    }
                    
                    # Append results
                    with open(results_file, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result)
                    
                    # Update best parameters
                    current_score = result['f1_weighted']
                    if current_score > best_score:
                        best_score = current_score
                        best_params = {
                            'ordinal_weight': ordinal_weight,
                            'gamma': gamma,
                            'alpha': alpha
                        }
                    
                    print(f"\nCurrent Results:")
                    print(f"F1 Weighted: {result['f1_weighted']:.4f}")
                    print(f"Kappa: {result['kappa']:.4f}")
                    
                except Exception as e:
                    print(f"Error during parameter combination: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                finally:
                    # Restore original function
                    globals()['get_focal_ordinal_loss'] = original_get_focal_ordinal_loss
                    
                    # Clear memory
                    tf.keras.backend.clear_session()
                    gc.collect()
    
    print("\nGrid Search Complete!")
    if best_params:
        print("\nBest Parameters:")
        print(f"Ordinal Weight: {best_params['ordinal_weight']}")
        print(f"Gamma: {best_params['gamma']}")
        print(f"Alpha: {best_params['alpha']}")
        print(f"Best F1 Weighted: {best_score:.4f}")
    
    return best_params, results_file
def main_search(data_percentage, train_patient_percentage=0.8, n_runs=3):
    csv_filename = os.path.join(result_dir, 'modality_combination_results_V64_4.csv')
    fieldnames = ['Modalities', 'Accuracy (Mean)', 'Accuracy (Std)', 
                  'Macro Avg F1-score (Mean)', 'Macro Avg F1-score (Std)', 
                  'Weighted Avg F1-score (Mean)', 'Weighted Avg F1-score (Std)', 
                  'I F1-score (Mean)', 'P F1-score (Mean)', 'R F1-score (Mean)', 
                  "Cohen's Kappa (Mean)", "Cohen's Kappa (Std)"]

    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    modalities = ['metadata', 'depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map']
    excluded_combinations =[]
    # excluded_combinations =[('depth_rgb',), ('depth_map',), ('thermal_rgb',), ('thermal_map',),]
    # excluded_combinations = [('metadata',), ('depth_rgb',), ('depth_map',), ('thermal_rgb',), ('thermal_map',), ('metadata','depth_rgb'), 
    #                          ('metadata','depth_map'), ('metadata','thermal_rgb'), ('metadata','thermal_map'), ('depth_rgb','depth_map'), ('depth_rgb','thermal_rgb'),]
    # excluded_combinations = [('metadata',), ('depth_rgb',), ('depth_map',), ('thermal_rgb',), ('thermal_map',), 
    #                          ('metadata','thermal_map'),('metadata','depth_rgb'),]
    included_combinations = [('metadata',), ('depth_rgb',), ('depth_map',), ('thermal_map',),
                             ('metadata','depth_rgb'), ('metadata','depth_map'), ('metadata','thermal_map'), ('depth_rgb','depth_map'), ('depth_rgb','thermal_map'),
                             ('metadata','thermal_rgb', 'thermal_map'), ('metadata','depth_rgb', 'depth_map'), ('depth_rgb','thermal_map', 'depth_map'),
                             ('metadata','depth_map', 'thermal_rgb', 'thermal_map'), 
                             ('depth_rgb','depth_map', 'thermal_rgb', 'thermal_map'),
                             ('metadata', 'depth_rgb','depth_map', 'thermal_rgb', 'thermal_map'),
                             ]

    # Generate all possible combinations
    all_combinations = []
    for r in range(1, len(modalities) + 1):
        all_combinations.extend(itertools.combinations(modalities, r))

    # Filter out the excluded combinations
    # combinations_to_process = [comb for comb in all_combinations if comb not in excluded_combinations]
    combinations_to_process = [comb for comb in all_combinations if comb in included_combinations]

    for combination in combinations_to_process:
    # for combination in [('metadata','depth_rgb','depth_map','thermal_map')]: #Cedar2
    # for combination in [1]:
        selected_modalities = list(combination)
        # selected_modalities = ['metadata', 'depth_rgb', 'thermal_map']
        print(f"\nTesting modalities: {', '.join(selected_modalities)}")
        # Load and prepare the dataset
        data = prepare_dataset(depth_bb_file, thermal_bb_file, csv_file, selected_modalities)
        from MisclassificationFunctions import filter_frequent_misclassifications
        data = filter_frequent_misclassifications(data, result_dir)
        if data_percentage < 100:
            data = data.sample(frac=data_percentage / 100, random_state=42).reset_index(drop=True)
        print(f"Using {data_percentage}% of the data: {len(data)} samples")
        # Perform cross-validation with manual patient split
        run_data = data.copy(deep=True)
        cv_results, confusion_matrices, histories = cross_validation_manual_split(run_data, selected_modalities, train_patient_percentage, n_runs)
          
        # Calculate average metrics and their standard deviations with error handling
        avg_accuracy = np.mean([m['accuracy'] for m in cv_results])
        std_accuracy = np.std([m['accuracy'] for m in cv_results])
        avg_f1_macro = np.mean([m['f1_macro'] for m in cv_results])
        std_f1_macro = np.std([m['f1_macro'] for m in cv_results])
        avg_f1_weighted = np.mean([m['f1_weighted'] for m in cv_results])
        std_f1_weighted = np.std([m['f1_weighted'] for m in cv_results])
        avg_kappa = np.mean([m['kappa'] for m in cv_results])
        std_kappa = np.std([m['kappa'] for m in cv_results])

        # Calculate average F1-scores for each class
        f1_classes_list = [m['f1_classes'] for m in cv_results]
        avg_f1_classes = np.mean(f1_classes_list, axis=0)

        # Prepare results for CSV
        result = {
            'Modalities': '+'.join(selected_modalities),
            'Accuracy (Mean)': avg_accuracy,
            'Accuracy (Std)': std_accuracy,
            'Macro Avg F1-score (Mean)': avg_f1_macro,
            'Macro Avg F1-score (Std)': std_f1_macro,
            'Weighted Avg F1-score (Mean)': avg_f1_weighted,
            'Weighted Avg F1-score (Std)': std_f1_weighted,
            'I F1-score (Mean)': avg_f1_classes[0],
            'P F1-score (Mean)': avg_f1_classes[1],
            'R F1-score (Mean)': avg_f1_classes[2],
            "Cohen's Kappa (Mean)": avg_kappa,
            "Cohen's Kappa (Std)": std_kappa
        }

        # Append the result to the CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)

        print(f"Results for {', '.join(selected_modalities)} appended to {csv_filename}")
        
        # Clean up after each modality combination
        try:
            tf.keras.backend.clear_session()
            gc.collect()
            clear_gpu_memory()
            reset_keras()
        except Exception as e:
            print(f"Error clearing memory stats: {str(e)}")
        try:
            del data, run_data, cv_results, confusion_matrices, histories, result
        except Exception as e:
            print(f"Error deleting variables: {str(e)}")

    print(f"\nAll results saved to {csv_filename}")
    
def main(mode='search', data_percentage=100, train_patient_percentage=0.8, n_runs=3):
    """
    Combined main function that can run either modality search or specialized evaluation.
    
    Args:
        mode (str): Either 'search' or 'specialized' to determine which analysis to run
        data_percentage (float): Percentage of data to use
        train_patient_percentage (float): Percentage of patients to use for training
        n_runs (int): Number of runs for cross-validation
    """
    # Clear any existing cache files
    import glob
    cache_patterns = [
        os.path.join(result_dir, 'tf_cache_train*'),
        os.path.join(result_dir, 'tf_cache_valid*'),
        'tf_cache_train*',
        'tf_cache_valid*'
    ]
    
    for pattern in cache_patterns:
        try:
            cache_files = glob.glob(pattern)
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                except Exception as e:
                    print(f"Warning: Could not remove cache file {cache_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error while processing pattern {pattern}: {str(e)}")

    if mode.lower() == 'search':
        main_search(data_percentage, train_patient_percentage, n_runs)
    elif mode.lower() == 'specialized':
        main_with_specialized_evaluation(data_percentage, train_patient_percentage, n_runs)
    else:
        raise ValueError("Mode must be either 'search' or 'specialized'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate multimodal models for wound healing phase classification.")
    parser.add_argument("--mode", type=str, choices=['search', 'specialized'], default='search',
                      help="Mode of operation: 'search' for modality combination search or 'specialized' for specialized evaluation")
    parser.add_argument("--data_percentage", type=float, default=100.0,
                      help="Percentage of data to use (default: 100.0)")
    parser.add_argument("--train_patient_percentage", type=float, default=0.8,
                      help="Percentage of patients to use for training (default: 0.8)")
    parser.add_argument("--n_runs", type=int, default=3,
                      help="Number of runs for cross-validation (default: 3)")
    args = parser.parse_args()

    # Clear memory before starting
    clear_cache_files()
    clear_gpu_memory()
    reset_keras()
    clear_cuda_memory()

    # Run the selected mode
    # main(args.mode, args.data_percentage, args.train_patient_percentage, args.n_runs)
    main('specialized', 100, 0.70, 5)
    # best_params, results_file = perform_grid_search(100, 0.8, 2)
    # Clear memory after completion
    clear_gpu_memory()
    reset_keras()
    clear_cuda_memory()
