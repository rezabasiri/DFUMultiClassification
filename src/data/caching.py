import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.data.generative_augmentation_v1_3 import create_enhanced_augmentation_fn
from src.data.preprocessing import load_and_preprocess_image
from src.utils.config import get_project_paths, get_data_paths
from src.utils.verbosity import vprint, get_verbosity

# Get paths from centralized config
directory, result_dir, root = get_project_paths()
data_paths = get_data_paths(root)

image_folder = data_paths['image_folder']
depth_folder = data_paths['depth_folder']
thermal_folder = data_paths['thermal_folder']
thermal_rgb_folder = data_paths['thermal_rgb_folder']

# Image processing parameters
image_size = 128
   
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
        processed_image = tf.ensure_shape(processed_image, (image_size, image_size, 3))
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
            tf.cast(row['Patient#'], tf.int32),
            tf.cast(row['Appt#'], tf.int32),
            tf.cast(row['DFU#'], tf.int32)
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
    )
    
    # Calculate how many samples we need
    n_samples = len(best_matching_df)
    steps = int(np.ceil(n_samples / batch_size))  # Keras 3 requires int for steps
    k = steps * batch_size  # Total number of samples needed
    
    # Cache the dataset
    cache_filename = 'tf_cache_train' if is_training else 'tf_cache_valid'
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        dataset = dataset.cache(os.path.join(cache_dir, cache_filename))
    else:
        dataset = dataset.cache(cache_filename)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(best_matching_df), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Apply augmentation after batching
    if is_training:
        if augmentation_fn:
            # Use provided augmentation function (includes generative augmentations)
            dataset = dataset.map(
                augmentation_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        # else:                                         #TODO: Add back default augmentations
        #     # Fall back to regular augmentation
        #     dataset = dataset.map(
        #         create_augmentation_fn(prob=0.25),
        #         num_parallel_calls=tf.data.AUTOTUNE
        #     )

    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, steps
def prepare_cached_datasets(data1, selected_modalities, train_patient_percentage=0.8,
                          batch_size=32, cache_dir=None, gen_manager=None, aug_config=None, run=0):
    """
    Prepare cached datasets with proper metadata handling based on selected modalities.
    """
    # Create a deep copy of the data
    data = data1.copy(deep=True)
    data = data.reset_index(drop=True)
    
    # Split patients into train and validation sets
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
    def apply_sampling_to_df(df):
        X = df.drop('Healing Phase Abs', axis=1)
        y = df['Healing Phase Abs']
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        
        resampled_df = pd.concat([
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.Series(y_resampled, name='Healing Phase Abs')
        ], axis=1)
        return resampled_df
    if 'metadata' in selected_modalities:
        # Calculate class weights for Random Forest models
        unique_cases = train_data[['Patient#', 'Appt#', 'DFU#', 'Healing Phase Abs']].drop_duplicates().copy()
        print(f"\nUnique cases: {len(unique_cases)} (before oversampling)")
        
        # Create binary labels on unique cases
        unique_cases['label_bin1'] = (unique_cases['Healing Phase Abs'] > 0).astype(int)
        unique_cases['label_bin2'] = (unique_cases['Healing Phase Abs'] > 1).astype(int)
        
        # Print true class distributions
        vprint("\nTrue binary label distributions (unique cases):", level=2)
        if get_verbosity() == 2:
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
    
    train_data = apply_sampling_to_df(train_data)
    
    def preprocess_split(split_data, is_training=True, class_weight_dict_binary1=None,
                            class_weight_dict_binary2=None, rf_model1=None, rf_model2=None,
                            imputer=None, scaler=None):
            """Preprocess data with proper handling of metadata and image columns"""
            # Create a copy of the data
            split_data = split_data.copy()

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
                
                # Feature engineering on metadata columns only
                metadata_df['BMI'] = metadata_df['Weight (Kg)'] / ((metadata_df['Height (cm)'] / 100) ** 2)
                metadata_df['Age above 60'] = (metadata_df['Age'] > 60).astype(int)
                metadata_df['Age Bin'] = pd.cut(metadata_df['Age'], 
                                            bins=range(0, int(metadata_df['Age'].max()) + 20, 20), 
                                            right=False, 
                                            labels=range(len(range(0, int(metadata_df['Age'].max()) + 20, 20)) - 1))
                metadata_df['Weight Bin'] = pd.cut(metadata_df['Weight (Kg)'], 
                                                bins=range(0, int(metadata_df['Weight (Kg)'].max()) + 20, 20), 
                                                right=False, 
                                                labels=range(len(range(0, int(metadata_df['Weight (Kg)'].max()) + 20, 20)) - 1))
                metadata_df['Height Bin'] = pd.cut(metadata_df['Height (cm)'], 
                                                bins=range(0, int(metadata_df['Height (cm)'].max()) + 10, 10), 
                                                right=False, 
                                                labels=range(len(range(0, int(metadata_df['Height (cm)'].max()) + 10, 10)) - 1))
                
                # Handle categorical columns
                categorical_columns = ['Sex (F:0, M:1)', 'Side (Left:0, Right:1)', 'Foot Aspect', 'Odor', 'Type of Pain Grouped']
                for col in categorical_columns:
                    if col in metadata_df.columns:
                        metadata_df[col] = pd.Categorical(metadata_df[col]).codes
                
                # Other categorical mappings
                categorical_mappings = {
                    'Location Grouped (Hallux:1,Toes,Middle,Heel,Ankle:5)': {'ankle': 4, 'Heel': 3, 'middle': 2, 'toes': 1, 'Hallux': 0},
                    'Dressing Grouped': {'NoDressing': 0, 'BandAid': 1, 'BasicDressing': 1, 'AbsorbantDressing': 2, 'Antiseptic': 3, 'AdvanceMethod': 4, 'other': 4},
                    'Exudate Appearance (Serous:1,Haemoserous,Bloody,Thick:4)': {'Serous': 0, 'Haemoserous': 1, 'Bloody': 2, 'Thick': 3}
                }
                
                for col, mapping in categorical_mappings.items():
                    if col in metadata_df.columns:
                        metadata_df[col] = metadata_df[col].map(mapping)

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
                # Impute missing values
                columns_to_impute = [col for col in metadata_df.columns if col not in ['depth_rgb', 'depth_map', 'thermal_rgb', 'thermal_map',
                                                                        'depth_xmin', 'depth_ymin', 'depth_xmax', 'depth_ymax',
                                                                        'thermal_xmin', 'thermal_ymin', 'thermal_xmax', 'thermal_ymax']+['Healing Phase Abs']]
                # numeric_columns = split_data.select_dtypes(include=[np.number]).columns

                # Proper train/valid separation for imputation
                if is_training:
                    imputer = KNNImputer(n_neighbors=5)
                    metadata_df[columns_to_impute] = imputer.fit_transform(metadata_df[columns_to_impute])
                else:
                    # Use fitted imputer from training
                    metadata_df[columns_to_impute] = imputer.transform(metadata_df[columns_to_impute])

                for column in integer_columns:
                    if column in metadata_df.columns:
                        metadata_df[column] = metadata_df[column].astype(int)
                    else:
                        continue

                # Normalize features with StandardScaler
                if is_training:
                    scaler = StandardScaler()
                    metadata_df[columns_to_impute] = scaler.fit_transform(metadata_df[columns_to_impute])
                else:
                    # Use fitted scaler from training
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
                            n_estimators=800,
                            random_state=42,
                            class_weight=class_weight_dict_binary1,
                            n_jobs=-1
                        )
                        rf_model2 = RandomForestClassifier(
                            n_estimators=800,
                            random_state=42,
                            class_weight=class_weight_dict_binary2,
                            n_jobs=-1
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
                    pred1 = rf_model1.predict(dataset1)
                    pred2 = rf_model2.predict(dataset1)
                    # Get probabilities for positive class (class 1)
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

            return split_data, rf_model1, rf_model2, imputer, scaler

    # Preprocess both splits
    train_data, rf_model1, rf_model2, imputer, scaler = preprocess_split(train_data, is_training=True, class_weight_dict_binary1=class_weight_dict_binary1, class_weight_dict_binary2=class_weight_dict_binary2)
    valid_data, _, _, _, _ = preprocess_split(valid_data, is_training=False, rf_model1=rf_model1, rf_model2=rf_model2, imputer=imputer, scaler=scaler)
        
    # Create cached datasets
    train_dataset, steps_per_epoch = create_cached_dataset(
        train_data,
        selected_modalities,
        batch_size,
        is_training=True,
        cache_dir=result_dir,
        augmentation_fn=create_enhanced_augmentation_fn(gen_manager, aug_config) if gen_manager else None
    )

    valid_dataset, validation_steps = create_cached_dataset(
        valid_data,
        selected_modalities,
        batch_size,
        is_training=False,
        cache_dir=result_dir,
        augmentation_fn=None
    )
    return train_dataset, valid_dataset, steps_per_epoch, validation_steps