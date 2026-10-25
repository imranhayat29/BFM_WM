"""
Data handling functionality for wall models
"""
import numpy as np
import torch
import pickle as pkl
from typing import Dict, Tuple, List, Optional, Union
import os
import sys
import pandas as pd
import re

# NOTE: Here we define the column mapping for different input scaling modes.
#
# Number > 100: use upstream and downstream data
# 
COLUMN_MAP = {
    0: ['u1_y_over_nu'],
    1: ['u1_y_over_nu', 'up_y_over_nu'],
    2: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu'],
    3: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u3_y_over_nu'],
    4: ['u1_y_over_nu', 'u2_y_over_nu'], 
    5: ['u1_y_over_nu', 'u2_y_over_nu', 'u3_y_over_nu'], 
    6: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'upn_y_over_nu'], 
    # Upstream Only
    101: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u1_y_over_nu_upstream'], 
    102: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u1_y_over_nu_upstream', 'u2_y_over_nu_upstream'], 
    # Downstream Only
    103: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u1_y_over_nu_downstream'], 
    104: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u1_y_over_nu_downstream', 'u2_y_over_nu_downstream'], 
    # Both Upstream and Downstream
    105: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u1_y_over_nu_upstream', 'u1_y_over_nu_downstream'], 
    106: ['u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u1_y_over_nu_upstream', 'u1_y_over_nu_downstream', 'u2_y_over_nu_upstream', 'u2_y_over_nu_downstream' ], 
}

# Add parent directory to path to import wall_model_cases
sys.path.insert(0, os.path.abspath('..'))
from wall_model_cases import INPUT_TURB_FILES, TURB_CASES, DATASET_PLOT_TITLE, STATION

class WallModelDataHandler:
    """
    Handles data loading, preprocessing, and partitioning for wall models
    """
    
    def __init__(self, config: Dict, YMAX_FIX: float = 1, YMIN_FIX: float = 0):
        """Initialize with configuration"""
        self.config = config
        self.device = torch.device(f"cuda:{config.get('general', {}).get('GpuNum', -1)}"
                                  if torch.cuda.is_available() else 'cpu')
        
        # Data containers
        self.input = None
        self.output = None
        self.flow_type = None
        self.input_dim = None
        
        # Preprocessing
        self.input_mean = None
        self.input_std = None
        
        # Data splits
        self.train_index = None
        self.valid_index = None
        self.input_train = None
        self.output_train = None
        self.input_valid = None
        self.output_valid = None

        # NOTE: YN -> Fix upy_max for consistent plotting
        self.YMAX_FIX = YMAX_FIX
        self.YMIN_FIX = YMIN_FIX

        assert self.YMAX_FIX >= self.YMIN_FIX, "YMAX_FIX must be greater than or equal to YMIN_FIX"
        
    def read_data(self, data_sources: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read data from specified sources
        
        Args:
            data_sources: Optional dictionary specifying data sources, if None uses config
        
        Returns:
            Tuple of (input, output, flow_type)
        """
        # If data_sources not provided, use config
        if data_sources is None:
            data_sources = self.config
        
        return self._load_from_raw_sources(data_sources)
    
    def _load_from_raw_sources(self, data_sources: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from raw sources based on configuration
        
        Args:
            data_sources: Dictionary with dataset configuration
            
        Returns:
            Tuple of (input, output, flow_type)
        """
        # Get data configuration
        data_config = data_sources.get('data', {})
        # Get list of datasets to use
        datasets = []
        
        # Iterate through data_config to find datasets
        for key, value in data_config.items():
            if value == 1:
                # If value is 1, add the key to datasets
                datasets.append(key)

        # # Check for TBL data with specific angles
        tbl_angles = data_config.get('TBL', [])
        if isinstance(tbl_angles, list) and tbl_angles:
            for angle in tbl_angles:
                datasets.append(f'TBL_{angle}')

        # Load each dataset
        all_inputs = []
        all_outputs = []
        all_flow_types = []
        
        # Get input scaling from configuration
        input_scaling = data_sources.get('model', {}).get('inputs', {}).get('InputScaling', 1)
        print(f"Using input scaling mode: {input_scaling}")
        
        for dataset in datasets:
            is_station_data = False
            case = dataset # Default case name
            station_num = None
            station_x = None
            ##################################################################################################
            # NOTE: This is to read in only data at stations
            if 'station' in dataset:
                match = re.search(r'(.+)_station_(\d+)$', dataset) # Get the case name and station number
                if match:
                    case = match.group(1) # Get the case name
                    station_num = int(match.group(2)) # Get the station number
                    try:
                        # Look up the target x-coordinate for this station
                        station_x = STATION[case][station_num]
                        is_station_data = True
                        print(f"Dataset '{dataset}' identified as station data: case='{case}', station_num={station_num}, target_x={station_x}")
                    except KeyError:
                        print(f"Warning: Station definition not found for case '{case}', station {station_num} in STATION dict. Skipping station filtering for '{dataset}'.")
                else:
                    print(f"Warning: Dataset key '{dataset}' contains 'station' but doesn't match expected format 'case_station_num'. Skipping station filtering.")
            ##################################################################################################
            if case in INPUT_TURB_FILES:
                # Construct potential HDF5 file path from INPUT_TURB_FILES (.pkl path)
                h5_file_path = INPUT_TURB_FILES[case]

                # --- READ HDF5 FILE ---
                if h5_file_path and os.path.exists(h5_file_path):
                    print(f"Reading HDF5 file: {h5_file_path}")
                    inputs_df = pd.read_hdf(h5_file_path, key='inputs')
                    outputs_series = pd.read_hdf(h5_file_path, key='output').iloc[:, 0] # Ensure it's a Series
                    flow_type_df = pd.read_hdf(h5_file_path, key='flow_type')
                    unnormalized_inputs_df = pd.read_hdf(h5_file_path, key='unnormalized_inputs')
                    # print(f"  Loaded keys: inputs, output, flow_type, unnormalized_inputs")

                    # Keep data as DataFrames for filtering and selection
                    all_data_inputs_df = inputs_df
                    outputs_for_processing = outputs_series
                    flow_type_for_processing = flow_type_df
                    unnormalized_for_filter = unnormalized_inputs_df
                else:
                    print(f"Warning: HDF5 file not found for dataset {case}. Expected path: {h5_file_path}. Skipping dataset.")
                    continue # Skip this dataset if HDF5 file doesn't exist

                # --- STATION X-LOCATION FILTERING (using DataFrames) ---
                if is_station_data:
                    print(f"Applying station filtering for station {station_num} (target x={station_x})...")
                    # *** IMPORTANT: Adjust 'x' if the column name for x-coordinates is different ***
                    x_coord_column = 'x'

                    if x_coord_column not in flow_type_for_processing.columns:
                        print(f"  Error: Column '{x_coord_column}' not found in flow_type DataFrame for dataset '{case}'. Cannot perform station filtering. Skipping dataset.")
                        continue # Skip this dataset as station filtering is not possible

                    if flow_type_for_processing.empty:
                        print(f"  Warning: flow_type DataFrame is empty for dataset '{case}'. Skipping station filtering.")
                    else:
                        x_coords = flow_type_for_processing[x_coord_column].values # Get x-coordinates as numpy array

                        # Find the index of the row with the x-coordinate closest to station_x
                        # Note: argmin() finds the index in the numpy array `x_coords`
                        closest_numpy_idx = np.abs(x_coords - station_x).argmin()
                        closest_x_in_data = x_coords[closest_numpy_idx]

                        # Validate if the closest point found is acceptably close
                        if np.abs(closest_x_in_data - station_x) > 1e-3:
                            print(f"  Error: No data points found sufficiently close to station {station_num} (target x={station_x}) in case {case}. Closest x found is {closest_x_in_data}. Skipping dataset '{dataset}'.")
                            # Depending on requirements, you might raise ValueError here instead of continuing
                            # raise ValueError(f"No data points found for station {station_num}...")
                            continue
                        else:
                            # Create a boolean mask for rows where the x-coordinate is very close
                            # to the *actual closest value found in the data*, handling float precision.
                            station_mask = np.abs(flow_type_for_processing[x_coord_column] - closest_x_in_data) < 1e-6

                            num_before = len(all_data_inputs_df)
                            # Apply the mask to filter all relevant DataFrames/Series
                            all_data_inputs_df = all_data_inputs_df[station_mask]
                            outputs_for_processing = outputs_for_processing[station_mask]
                            flow_type_for_processing = flow_type_for_processing[station_mask]
                            unnormalized_for_filter = unnormalized_for_filter[station_mask]
                            num_after = len(all_data_inputs_df)

                            if num_after == 0:
                                print(f"  Warning: Station filtering resulted in 0 data points for dataset '{dataset}' at x ≈ {closest_x_in_data:.6f}. Skipping remaining processing for this dataset.")
                                continue # Skip further processing if no data remains

                            print(f"  Filtered for station {station_num}: {num_before} -> {num_after} points at x ≈ {closest_x_in_data:.6f}")

                # --- Filtering (using DataFrames) ---
                ymax = data_config.get('upy', 0.15)
                # BFM_WM_v2: default downy 0.0 keeps points down to the wall (upstream default: 0.005)
                ymin = data_config.get('downy', 0.000)
                if ymax < 1.0:
                    print(f"Filtering {case} with upy_max={ymax}...")

                    # NOTE: Add stronger check for columns existence
                    if 'y' not in unnormalized_for_filter.columns:
                        raise ValueError(f"Column 'y' required for upy filtering not found in unnormalized inputs for dataset {case}.")
                    if 'delta' not in flow_type_for_processing.columns:
                        raise ValueError(f"Column 'delta' required for upy filtering not found in flow_type for dataset {case}.")

                    if 'y' in unnormalized_for_filter.columns and 'delta' in flow_type_for_processing.columns:
                        y_vals = unnormalized_for_filter['y'].values
                        delta_vals = flow_type_for_processing['delta'].astype(float).values

                        mask1 = y_vals <= ymax * delta_vals
                        mask2 = y_vals >= ymin * delta_vals
                        mask = mask1 & mask2

                        all_data_inputs_df = all_data_inputs_df[mask]
                        outputs_for_processing = outputs_for_processing[mask]
                        flow_type_for_processing = flow_type_for_processing[mask]

                        print(f"  Filtered dataset {case}: {len(mask)} -> {all_data_inputs_df.shape[0]} points")
                    else:
                        print(f"  Warning: Cannot filter by 'upy'. Missing 'y' in unnormalized or 'delta' in flow_type.")

                # --- Column Selection (using DataFrame column names) ---
                input_scaling = data_sources.get('model', {}).get('inputs', {}).get('InputScaling', 1)
                selected_columns = COLUMN_MAP.get(input_scaling, COLUMN_MAP[1])

                available_cols = all_data_inputs_df.columns
                valid_selection = all(col in available_cols for col in selected_columns)

                if valid_selection:
                    inputs_selected_df = all_data_inputs_df[selected_columns]
                else:
                    print(f"Warning: Dataset {case} (HDF5 source) is missing required columns for input_scaling={input_scaling}. Required: {selected_columns}. Available: {list(available_cols)}. Skipping dataset.")
                    continue

                # --- Secondary filtering (if needed) ---
                near_wall = data_config.get('near_wall_threshold', 0.0)
                # It seems that we only need to filter based on u1_y_over_nu
                if near_wall > 0.0:
                    # This branch only trains points close to the wall
                    if 'u1_y_over_nu' not in inputs_selected_df.columns:
                        raise ValueError(f"Column 'u1_y_over_nu' required for near_wall filtering not found in dataset {case}.")
                    else:
                        # Only concern data that has low u1_y_over_nu and u2_y_over_nu values
                        mask1 = inputs_selected_df['u1_y_over_nu'] <= near_wall
                        # BFM_WM_v2: no u1 > 0 mask, so reversed-flow (negative u1) points are kept
                        # (upstream: mask = mask1 & (u1_y_over_nu > 0))
                        mask = mask1

                        outputs_for_processing = outputs_for_processing[mask]
                        flow_type_for_processing = flow_type_for_processing[mask]
                        inputs_selected_df = inputs_selected_df[mask]

                        # Use log scale for near wall filtering
                        for col in selected_columns:
                            inputs_selected_df[col] = np.sign(inputs_selected_df[col]) * np.log1p(np.abs(inputs_selected_df[col]))

                        print(f"  Applied near_wall filtering at {near_wall}: {len(mask)} -> {len(inputs_selected_df)} points")
                else:
                    # This branch excludes points very close to the wall. Set near_wall == 0 to disable this filtering.
                    # If near_wall <= 0, exclude points with low u1_y_over_nu values (e.g., negative or zero)
                    mask1 = np.abs(inputs_selected_df['u1_y_over_nu']) > np.abs(near_wall)
                    mask  = mask1

                    outputs_for_processing = outputs_for_processing[mask]
                    flow_type_for_processing = flow_type_for_processing[mask]
                    inputs_selected_df = inputs_selected_df[mask]

                    print(f"  Applied near_wall filtering at {near_wall}: {len(mask)} -> {len(inputs_selected_df)} points")

                # --- Convert final selections to NumPy for concatenation ---
                inputs_np = inputs_selected_df.values
                outputs_np = outputs_for_processing.values
                flow_type_np = flow_type_for_processing.values

                all_inputs.append(inputs_np)
                all_outputs.append(outputs_np)
                all_flow_types.append(np.array([dataset] * len(outputs_np)))  # Store dataset name for each output

                print(f"  Processed {len(inputs_np)} samples from {dataset} with {inputs_np.shape[1]} input dimensions (Scaling Mode: {input_scaling})") 
                print(50*'-')

        # Combine all datasets
        if all_inputs:
            combined_inputs = np.vstack(all_inputs)
            combined_outputs = np.concatenate(all_outputs)
            combined_flow_types = np.array(np.concatenate(all_flow_types), dtype=str)
            
            # Store in class attributes
            self.input = combined_inputs
            self.output = combined_outputs
            self.flow_type = combined_flow_types
            self.input_dim = combined_inputs.shape[1]
            
            print(50 * '-')
            print(f"Combined dataset has {len(combined_inputs)} samples with {self.input_dim} input dimensions")
            
            return combined_inputs, combined_outputs
        else:
            raise ValueError("No data was loaded. Check dataset configuration and file paths.")
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data (standardization, etc.)
        
        Returns:
            Tuple of (preprocessed_input, output)
        """
        if self.input is None or self.output is None:
            raise ValueError("Data not loaded. Call read_data first.")
        
        # Standardize inputs
        self.input_mean = np.mean(self.input, axis=0)
        self.input_std = np.std(self.input, axis=0)
        
        # Replace zero standard deviations to avoid division by zero
        self.input_std[self.input_std == 0] = 1.0
        
        # Standardize
        preprocessed_input = (self.input - self.input_mean) / self.input_std
        
        return preprocessed_input, self.output
    
    def split_data(self, train_ratio: float = 0.8, random_split: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets
        
        Args:
            train_ratio: Ratio of training data
            random_split: Whether to split randomly or sequentially
        
        Returns:
            Tuple of (input_train, output_train, input_valid, output_valid)
        """
        if self.input is None or self.output is None:
            raise ValueError("Data not loaded. Call read_data first.")
        
        n_samples = len(self.input)
        n_train = int(train_ratio * n_samples)
        
        if random_split:
            # Random split
            indices = np.random.permutation(n_samples)
            self.train_index = indices[:n_train]
            self.valid_index = indices[n_train:]
        else:
            # Sequential split
            self.train_index = np.arange(n_train)
            self.valid_index = np.arange(n_train, n_samples)
        
        # Get train and validation data
        self.input_train = self.input[self.train_index]
        self.output_train = self.output[self.train_index]
        self.input_valid = self.input[self.valid_index]
        self.output_valid = self.output[self.valid_index]
        
        print(50 * '-')
        print(f"Split data into {len(self.input_train)} training and {len(self.input_valid)} validation samples")
        print(50 * '-')
        
        return self.input_train, self.output_train, self.input_valid, self.output_valid
    
    def convert_to_tensors(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert data to PyTorch tensors
        
        Args:
            device: Device to place tensors on
        
        Returns:
            Tuple of (input_train, output_train, input_valid, output_valid) as tensors
        """
        if self.input_train is None or self.output_train is None:
            raise ValueError("Data not split. Call split_data first.")
        
        if device is None:
            device = self.device
        
        # Convert to tensors
        input_train_tensor = torch.from_numpy(self.input_train).float().to(device)
        output_train_tensor = torch.from_numpy(self.output_train).float().to(device)
        input_valid_tensor = torch.from_numpy(self.input_valid).float().to(device)
        output_valid_tensor = torch.from_numpy(self.output_valid).float().to(device)
        
        # Update class attributes
        self.input_train = input_train_tensor
        self.output_train = output_train_tensor
        self.input_valid = input_valid_tensor
        self.output_valid = output_valid_tensor
        
        return input_train_tensor, output_train_tensor, input_valid_tensor, output_valid_tensor
    
    def load_external_dataset(self, dataset_key: str, config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a specific external dataset for testing
        
        Args:
            dataset_key: Key identifying the dataset to load
            config: Optional configuration to use (if None, uses the current config)
            
        Returns:
            Tuple of (inputs, outputs, unnormalized_inputs, flow_type)
        """
        if config is None:
            config = self.config
            
        # Get input scaling from configuration
        input_scaling = config.get('model', {}).get('inputs', {}).get('InputScaling', 1)
        
        # Try direct path first
        file_path = f'./data/{dataset_key}_data.h5'
        
        if not os.path.exists(file_path):
            # Try alternative paths
            if dataset_key in INPUT_TURB_FILES:
                file_path = INPUT_TURB_FILES[dataset_key].replace('.pkl', '.h5')
            elif dataset_key.isdigit() or (dataset_key.startswith('-') and dataset_key[1:].isdigit()):
                file_path = INPUT_TURB_FILES.get(f'TBL_{dataset_key}', '').replace('.pkl', '.h5')
            elif dataset_key.isdigit():
                file_path = INPUT_TURB_FILES.get(f'naca_4412_{dataset_key}', '').replace('.pkl', '.h5')
            else:
                for key, path in INPUT_TURB_FILES.items():
                    if dataset_key in key:
                        file_path = path.replace('.pkl', '.h5')
                        break
        
        if not os.path.exists(file_path):
            raise ValueError(f"Dataset {dataset_key} not found or file doesn't exist. Tried path: {file_path}")
        
        # Read each component from HDF5
        inputs_df = pd.read_hdf(file_path, key='inputs')
        outputs = pd.read_hdf(file_path, key='output').values
        unnormalized_inputs = pd.read_hdf(file_path, key='unnormalized_inputs').values
        flow_type = pd.read_hdf(file_path, key='flow_type').values
        
        # Apply optional filtering based on config
        ymax = config.get('data', {}).get('upy', 0.15)  # Upper y value for input
        # BFM_WM_v2: default downy 0.0 keeps points down to the wall (upstream default: 0.005)
        ymin = config.get('data', {}).get('downy', 0.000)  # Lower y value for input
        
        # Filter based on upy if needed
        if ymax < 1.0:
            # First column of unnormalized_inputs contains y values relative to delta
            y = unnormalized_inputs[:, 0]
            delta = np.array([float(flow_type[i, 3]) for i in range(len(flow_type))])

            # Filter out points where y exceeds upy_max * delta
            # NOTE: YN -> If UPY_MAX_FIX is set and less than 1.0, use it to apply consistent filtering
            if self.YMAX_FIX < 1.0:
                mask1 = y <= self.YMAX_FIX * delta
            else:
                mask1 = y <= ymax * delta

            if self.YMIN_FIX > 0.0:
                mask2 = y >= self.YMIN_FIX * delta
            else:
                mask2 = y >= ymin * delta

            mask = mask1 & mask2
            
            # Apply mask to all data arrays
            inputs_df = inputs_df[mask]
            outputs = outputs[mask]
            unnormalized_inputs = unnormalized_inputs[mask]
            flow_type = flow_type[mask]
            
            print(50 * '-')
            # BFM_WM_v2: the message also prints the applied upy (as in the BFM_WM fork)
            print(f"Filtered external dataset {dataset_key}: {len(mask)} -> {len(inputs_df)} points based on upy_max={ymax}")
        
        # Select input columns based on input_scaling
        selected_columns = COLUMN_MAP.get(input_scaling, COLUMN_MAP[1])

        # --- Secondary filtering (if needed) ---
        near_wall = config.get('data', {}).get('near_wall_threshold', -1.0)
        # It seems that we only need to filter based on u1_y_over_nu
        if near_wall > 0.0:
            if 'u1_y_over_nu' not in inputs_df.columns:
                raise ValueError(f"Column 'u1_y_over_nu' required for near_wall filtering not found in dataset {case}.")
            else:
                # Only concern data that has low u1_y_over_nu and u2_y_over_nu values
                mask1 = inputs_df['u1_y_over_nu'] <= near_wall
                # BFM_WM_v2: no u1 > 0 mask, so reversed-flow (negative u1) points are kept
                # (upstream: mask = mask1 & (u1_y_over_nu > 0))
                mask = mask1

                inputs_df = inputs_df[mask]
                outputs = outputs[mask]
                unnormalized_inputs = unnormalized_inputs[mask]
                flow_type = flow_type[mask]

                # Use log scale for near wall filtering
                for col in selected_columns:
                    inputs_df[col] = np.sign(inputs_df[col]) * np.log1p(np.abs(inputs_df[col]))

                print(f"  Applied near_wall filtering at {near_wall}: {len(mask)} -> {len(inputs_df)} points")
        else:
            print("No near_wall target filtering applied.")

        
        # Verify all required columns exist
        if not all(col in inputs_df.columns for col in selected_columns):
            print(f"Warning: Not all required columns {selected_columns} found in dataset. Available columns: {inputs_df.columns}")
            # Return None if columns are missing
            return None, None, None, None
        else:
            inputs = inputs_df[selected_columns].values
        
        print(f"Loaded external dataset {dataset_key} with input scaling mode {input_scaling}, input dims {inputs.shape[1]}")
        return inputs, outputs, unnormalized_inputs, flow_type
    
    def _prepare_weights_custom(self) -> torch.Tensor:
        """
        Prepare customized weights based on flow types
        
        Returns:
            Tensor of weights corresponding to each sample
        """
        # Get custom weight power from config
        custom_weight_power = self.config.get('model', {}).get('weights', {}).get('custom', 0)
        
        if custom_weight_power <= 0 or self.flow_type is None:
            return None
            
        # Convert to numpy for easier manipulation
        flow_types_np = self.flow_type
        if isinstance(flow_types_np, torch.Tensor):
            flow_types_np = flow_types_np.cpu().numpy()
            
        if len(flow_types_np.shape) > 1:
            flow_types_np = flow_types_np[:, 0]  # Extract first column if multidimensional
            
        # Count occurrences of each flow type
        unique_flow_types = np.unique(flow_types_np)
        count_per_flow_type = {}
        weight_per_flow_type = {}
        min_weight = float('inf')
        
        # Calculate base weights (inverse frequency)
        for flow_type in unique_flow_types:
            count = np.sum(flow_types_np == flow_type)
            count_per_flow_type[flow_type] = count
            # Base weight is inverse frequency
            weight = len(flow_types_np) / count
            weight_per_flow_type[flow_type] = weight
            min_weight = min(min_weight, weight)
        
        # Normalize weights by minimum weight and apply power
        for flow_type in unique_flow_types:
            normalized_weight = weight_per_flow_type[flow_type] / min_weight
            # Apply power based on config setting
            weight_per_flow_type[flow_type] = normalized_weight ** int(custom_weight_power)
        
        # Log information about flow types and weights
        for flow_type in unique_flow_types:
            print(f"Flow type: {flow_type}, count: {count_per_flow_type[flow_type]}, "
                  f"weight: {weight_per_flow_type[flow_type]}")
        
        # Create weight tensor matching original data order
        weights = torch.tensor([weight_per_flow_type[ft] for ft in flow_types_np], 
                              dtype=torch.float32)
        
        # If a device is defined, move tensor to that device
        if hasattr(self, 'device'):
            weights = weights.to(self.device)
        
        return weights
    
    @classmethod
    def load_preprocessed_data(cls, path: str, config: Dict) -> 'WallModelDataHandler':
        """
        Load preprocessed data from file
        
        Args:
            path: Path to load the data from
            config: Configuration dictionary
        
        Returns:
            WallModelDataHandler instance with loaded data
        """
        instance = cls(config)
        
        with open(path, 'rb') as f:
            data = pkl.load(f)
            instance.input = data['inputs']
            instance.output = data['outputs']
            instance.flow_type = data['flow_type']
            instance.input_mean = data.get('input_mean')
            instance.input_std = data.get('input_std')
        
        instance.input_dim = instance.input.shape[1]
        
        print(f"Loaded preprocessed data from {path} with {len(instance.input)} samples")
        return instance
