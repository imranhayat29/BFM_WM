import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Callable
import wandb
import glob  # Import the glob module
from src.loss import weighted_mse_loss, EWC_loss

class WallModelTrainer:
    """
    Handles model training functionality for wall models
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """
        Initialize trainer with model and configuration

        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device

        # Training history
        self.train_loss_history = []
        self.valid_loss_history = []
        self.best_valid_loss = float('inf')

        # Set up optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Sample weights
        self.weights = None
        
        # Set up wandb if configured
        self.use_wandb = self.config.get('general', {}).get('UseWandb', False)
        if self.use_wandb:
            self._init_wandb()

    def _setup_optimizer(self) -> None:
        """Set up the optimizer based on configuration"""
        optimizer_type = self.config.get('training', {}).get('optimizer', {}).get('Type', 'Adam')
        lr = self.config.get('training', {}).get('optimizer', {}).get('LearningRate', 1e-3)
        weight_decay = self.config.get('training', {}).get('optimizer', {}).get('WeightDecay', 0)

        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = self.config.get('training', {}).get('optimizer', {}).get('Momentum', 0.9)
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _setup_scheduler(self) -> None:
        """Set up the learning rate scheduler based on configuration"""
        scheduler_type = self.config.get('training', {}).get('scheduler', {}).get('Type', 'plateau')

        if scheduler_type.lower() == 'plateau':
            patience = self.config.get('training', {}).get('scheduler', {}).get('Patience', 50)
            factor = self.config.get('training', {}).get('scheduler', {}).get('Factor', 0.95)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience
            )
        elif scheduler_type.lower() == 'step':
            step_size = self.config.get('training', {}).get('scheduler', {}).get('StepSize', 30)
            gamma = self.config.get('training', {}).get('scheduler', {}).get('Gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        else:
            # Default: no scheduler
            self.scheduler = None

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases for experiment tracking"""
        wandb_project = self.config.get('general', {}).get('WandbProject', 'wall_model')
        wandb_entity = self.config.get('general', {}).get('WandbEntity', None)
        wandb_name = self.config.get('general', {}).get('WandbName', None)

        # Create a meaningful run name if not provided
        if wandb_name is None:
            # Base on model configuration
            ch_flag  = self._data_flag_check('CH')
            syn_flag = self._data_flag_check('SYN')
            tbl_flag = self._data_flag_check('TBL')
            input_scaling = self.config.get('model', {}).get('inputs', {}).get('InputScaling', 0)

            wandb_name = f"WM_CH{ch_flag}_SYN{syn_flag}_TBL{tbl_flag}_InputScaling{input_scaling}"

        # Initialize wandb
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name, config=self.config)

        # Watch the model
        wandb.watch(self.model)

    def _print_config_summary(self) -> None:
        """Print a summary of all configurations, indicating which are using default values"""
        default_config = {
            'general': {
                'Verbose': 1,
                'UseWandb': False,
                'Save': True,
                'SaveDir': './models',
                'GpuNum': 0,
                'WandbProject': 'wall_model',
                'WandbEntity': None,
                'WandbName': None
            },
            'data': {
                'CH': 0,
                'SYN': 0,
                'TBL': 0,
                'gaussian': 0,
                'bub': 0,
                'partition': {
                    'TrainRatio': 0.8,
                    'RandomSplitWM': True,
                },
                'upy': 0.2
            },
            'model': {
                'InputDim': 2,
                'HiddenLayers': [32, 32],
                'OutputDim': 1,
                'Activation': 'relu',
                'inputs': {
                    'InputScaling': 1
                },
                'outputs': {
                    'OutputScaling': 1
                },
                'LDS': {
                    'lds': 0,
                    'reweight': 'none',
                    'kernel': 'gaussian',
                    'ks': 5,
                    'sigma': 2,
                    'num_bins': 100
                },
                'FDS': {
                    'fds': 0
                },
                'weights': {
                    'custom': 0
                }
            },
            'training': {
                'LogTransform': 0,
                'LossFunction ': "MSE",
                'optimizer': {
                    'Type': 'Adam',
                    'LearningRate': 1e-3,
                    'WeightDecay': 0,
                    'Momentum': 0.9
                },
                'scheduler': {
                    'Type': 'plateau',
                    'Patience': 50,
                    'Factor': 0.95,
                    'Epochs': 100,
                    'PrintInterval': 10,
                    'SaveInterval': 100,
                    'StepSize': 30,
                    'Gamma': 0.1
                }
            }
        }
        
        print("\n" + "="*80)
        print(" "*30 + "CONFIGURATION SUMMARY")
        print("="*80)
        
        def print_config_section(config_section, default_section, prefix=""):
            for key, default_value in default_section.items():
                full_key = f"{prefix}{key}"
                
                # Get actual value, default to the default value if not present
                actual_value = config_section.get(key, default_value)
                
                # Check if the value is a dictionary (nested config)
                if isinstance(default_value, dict):
                    print(f"\n{full_key}:")
                    print_config_section(config_section.get(key, {}), default_value, prefix=f"  {full_key}.")
                else:
                    # Mark unaltered values
                    is_default = (actual_value == default_value)
                    default_mark = " [default]" if is_default else ""
                    
                    # Format the values for pretty printing
                    if isinstance(actual_value, list):
                        value_str = str(actual_value)
                    elif isinstance(actual_value, float):
                        value_str = f"{actual_value:.6g}"
                    else:
                        value_str = str(actual_value)
                    
                    print(f"  {full_key}: {value_str}{default_mark}")
        
        # Print each main section
        for section_name, default_section in default_config.items():
            print(f"\n[{section_name.upper()}]")
            print_config_section(self.config.get(section_name, {}), default_section)
            
        print("\n" + "="*80 + "\n")

    def train(self,
              input_train: torch.Tensor,
              output_train: torch.Tensor,
              input_valid: torch.Tensor,
              output_valid: torch.Tensor,
              FIM: torch.Tensor = None,
              data_handler=None,
              loss_fn: Optional[Callable] = None,
              epochs: Optional[int] = None,
              save_dir: Optional[str] = None,
              save_best: bool = True,
              model_name_prefix: Optional[str] = None) -> nn.Module:
        """
        Train the model

        Args:
            input_train: Training inputs
            output_train: Training targets
            input_valid: Validation inputs
            output_valid: Validation targets
            FIM: Fisher Information Matrix for EWC (if applicable)
            data_handler: Optional data handler instance for custom weighting
            loss_fn: Loss function (defaults to MSE if None)
            epochs: Number of epochs (if None, uses config)
            save_dir: Directory to save models (if None, uses config or disables saving)
            save_best: Whether to save only the best model
            model_name_prefix: Prefix for saved model names

        Returns:
            Trained model
        """
        # Print configuration summary before training
        self._print_config_summary()
        
        # Set up training parameters
        if epochs is None:
            epochs = self.config.get('training', {}).get('scheduler', {}).get('Epochs', 100)

        print_interval = self.config.get('training', {}).get('scheduler', {}).get('PrintInterval', 10)
        save_interval = self.config.get('training', {}).get('scheduler', {}).get('SaveInterval', 100)

        # Store training data for reference
        self.output_train = output_train
        
        # Store data sizes for reference
        self.n_train = len(input_train)
        self.n_valid = len(input_valid)
        
        # Prepare custom weights if custom weighting is enabled and data handler is provided
        self.weights = None
        self.weights_train = None
        self.weights_valid = None

        # Standardize inputs if scaling is enabled
        # self.input_mean = torch.mean(input_train, dim=0)
        # self.input_std = torch.std(input_train, dim=0)
        # input_train = (input_train - self.input_mean) / self.input_std
        # input_valid = (input_valid - self.input_mean) / self.input_std
        
        if self.config.get('model', {}).get('weights', {}).get('custom', 0) > 0:
            # Get the full weights
            self.weights = data_handler._prepare_weights_custom()
            
            if self.weights is not None:
                # Split weights into train and validation using data_handler indices
                self.weights_train = self.weights[data_handler.train_index]
                self.weights_valid = self.weights[data_handler.valid_index]
                print(50 * '=')
                print(f"Using custom weights with power {self.config.get('model', {}).get('weights', {}).get('custom')}")
                print(f"Split weights: {len(self.weights_train)} training, {len(self.weights_valid)} validation")
                print(50 * '=')

        # Set up loss function
        if loss_fn is None:
            # Use weighted MSE loss if weights are available
            if self.weights_train is not None:
                # Create separate loss functions for train and validation datasets
                loss_fn_train = lambda inputs, targets: weighted_mse_loss(inputs, targets, self.weights_train)
                loss_fn_valid = lambda inputs, targets: weighted_mse_loss(inputs, targets, self.weights_valid)
                loss_fn = loss_fn_train  # Default to training loss function
            else:
                loss_fn_train = nn.MSELoss()
                loss_fn_valid = nn.MSELoss()
                loss_fn = loss_fn_train

        # If specific EWC in the config, override the loss function
        if self.config.get('training', {}).get('LossFunction', '') == 'EWC':
            use_ecw = True
            loss_fn_ewc = EWC_loss
            self.old_params = {
                name: param.clone().detach() # Clone and detach to get a copy
                for name, param in self.model.named_parameters()
            }
            lambda_ewc = self.config.get('training', {}).get('lambda_ewc', 0.1)
            print(f"Using EWC loss with lambda: {lambda_ewc}")
        else:
            use_ecw = False

        # Determine saving options
        save_models = False
        save_dir = self.config.get('general', {}).get('SaveDir', './models') # Get SaveDir early
        if self.config.get('general', {}).get('Save', False):
            save_models = True
            os.makedirs(save_dir, exist_ok=True)

            # Create model name prefix if not provided
            if model_name_prefix is None:
                model_name_prefix = self._generate_model_name_prefix()

            save_dir = os.path.join(save_dir, model_name_prefix)

        # Main training loop
        start_time = time.time()
        epochs_since_improvement = 0

        # For relative loss
        output_proxy_train = torch.ones_like(output_train).to(self.device)
        output_proxy_valid = torch.ones_like(output_valid).to(self.device)

        for epoch in range(epochs):
            # Training step
            self.model.train()

            # Forward pass
            outputs = self.model(input_train).squeeze()

            # Compute loss (often using relative error)
            if self.weights_train is not None:
                # For weighted loss, use the training-specific loss function
                loss = loss_fn_train(outputs / output_train, output_proxy_train)
            else:
                loss = loss_fn(outputs / output_train, output_proxy_train)

            # Add EWC loss if applicable
            if use_ecw:
                loss += loss_fn_ewc(self.model, FIM, self.old_params, lambda_ewc=lambda_ewc)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                valid_outputs = self.model(input_valid).squeeze()
                # Use validation-specific loss function for validation
                if self.weights_valid is not None:
                    valid_loss = loss_fn_valid(valid_outputs / output_valid, output_proxy_valid)
                else:
                    valid_loss = loss_fn(valid_outputs / output_valid, output_proxy_valid)

                # Add EWC loss if applicable
                if use_ecw:
                    valid_loss += loss_fn_ewc(self.model, FIM, self.old_params, lambda_ewc=lambda_ewc)

            # Record losses
            train_loss = loss.item()
            valid_loss = valid_loss.item()

            self.train_loss_history.append(train_loss)
            self.valid_loss_history.append(valid_loss)

            # Update scheduler if available
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()

            # Print progress
            if (epoch + 1) % print_interval == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}')

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    # 'input_mean': self.input_mean.cpu().numpy(),
                    # 'input_std': self.input_std.cpu().numpy(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Save model if improved
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                epochs_since_improvement = 0

                if save_models and save_best:
                    # Save best model
                    model_path = os.path.join(
                        save_dir,
                        f"{model_name_prefix}_ep{epoch + 1}_tl{train_loss:.8f}_vl{valid_loss:.8f}.pth"
                    )
                    self._save_model(model_path)
            else:
                epochs_since_improvement += 1

            # Save model periodically if configured
            if save_models and not save_best and (epoch + 1) % save_interval == 0:
                model_path = os.path.join(
                    save_dir,
                    f"{model_name_prefix}_ep{epoch + 1}_tl{train_loss:.8f}_vl{valid_loss:.8f}.pth"
                )
                self._save_model(model_path)

        # Save final model if requested
        if save_models:
            final_model_path = os.path.join(
                save_dir,
                f"{model_name_prefix}_final_ep{epochs}_tl{train_loss:.8f}_vl{valid_loss:.8f}.pth"
            )
            self._save_model(final_model_path)

        # Training complete
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {self.best_valid_loss:.8f}")

        # Finalize wandb logging
        if self.use_wandb:
            wandb.log({
                'best_valid_loss': self.best_valid_loss,
                'training_time': training_time
            })
            wandb.finish()

        return self.model

    def _data_flag_check(self, case_str: str) -> int:
        """Check if a flag is set and return a string accordingly"""
        case_specifier =  self.config.get('data', {}).get(case_str, 0)

        if type(case_specifier) == list and len(case_specifier) > 0:
            return 1
        elif type(case_specifier) == int and case_specifier > 0:
            return 1
        else:
            return 0

    def _generate_model_name_prefix(self) -> str:
        """Generate a descriptive model name prefix based on configuration"""
        # Data sources used
        ch_flag = self._data_flag_check('CH')
        syn_flag = self._data_flag_check('SYN')
        gauss_flag = self._data_flag_check('gaussian')
        tbl_flag = self._data_flag_check('TBL')
        bub_flag = self._data_flag_check('bub')

        # Preprocessing flags
        fds_flag = 1 if self.config.get('model', {}).get('FDS', {}).get('fds', 0) > 0 else 0
        lds_flag = 1 if self.config.get('model', {}).get('LDS', {}).get('lds', 0) > 0 else 0
        custom_w_flag = 1 if self.config.get('model', {}).get('weights', {}).get('custom', 0) > 0 else 0

        # EWC flag
        ewc_flag = 1 if self.config.get('training', {}).get('LossFunction', '') == 'EWC' else 0
        lambda_ewc = self.config.get('training', {}).get('lambda_ewc', 0.1)

        # Input options
        input_scaling = self.config.get('model', {}).get('inputs', {}).get('InputScaling', 0)

        # Count training and validation samples if available
        n_train = ""
        n_valid = ""
        if hasattr(self, 'n_train') and hasattr(self, 'n_valid'):
            n_train = f"tn{self.n_train}"
            n_valid = f"vn{self.n_valid}"

        # Create name
        prefix = f"NN_wm_CH{ch_flag}_G{gauss_flag}_S{syn_flag}"

        # Add TBL and BUB flags if used
        if tbl_flag:
            prefix += f"_TBL{tbl_flag}"
        if bub_flag:
            prefix += f"_BUB{bub_flag}"

        # Add sample counts if available
        if n_train and n_valid:
            prefix += f"_{n_train}_{n_valid}"

        # Add preprocessing flags
        prefix += f"_fds{fds_flag}_lds{lds_flag}"

        # Add custom weights flag if used
        if custom_w_flag:
            prefix += f"_customw{custom_w_flag}"

        # Add input scaling
        prefix += f"_inputs{input_scaling}"

        # Add EWC flag if used
        if ewc_flag:
            prefix += f"_EWC{ewc_flag}_lambda{lambda_ewc}"

        return prefix

    @staticmethod
    def _index_tensor(index) -> Optional[torch.Tensor]:
        """BFM_WM_v2: split indices as an int64 CPU tensor (None stays None) for the checkpoint."""
        if index is None:
            return None
        return torch.as_tensor(np.asarray(index), dtype=torch.int64).cpu()

    def _save_model(self, path: str) -> None:
        """
        Save the model without data to reduce file size
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print('Create model folder: ', os.path.dirname(path))

        print('Save model to ', path)
        print('Current lowest validation loss:', self.best_valid_loss)

        # For nonfinal models, we only need a few checkpoints for restart
        # Therefore, we remove old models before saving
        if not path.endswith('final.pth'):  # Check if it's not the final model
            self._keep_max_files(os.path.dirname(path), max_files=5)

        # Save only essential model information (not the entire object)
        save_dict = {
            # Model architecture and parameters
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),

            # Essential metrics for evaluation
            'train_loss_history': self.train_loss_history,
            'valid_loss_history': self.valid_loss_history,
            'best_valid_loss': self.best_valid_loss,

            # Whether to do log transformation
            'log_transform': self.config.get('training', {}).get('LogTransform', 0),

            # Whether to focus solely on near-wall region
            'near_wall': self.config.get('near_wall_threshold', -1.0),

            # Normalization parameters (essential for inference)
            # 'input_mean': self.input_mean,
            # 'input_std': self.input_std,

            # Configuration and model definition params
            'config': self.config,
            'input_dim': getattr(self, 'input_dim', None),
            'input_scaling': self.config.get('model', {}).get('inputs', {}).get('InputScaling', None),
            'output_scaling': self.config.get('model', {}).get('outputs', {}).get('OutputScaling', None),
            'Nl': self.config.get('model', {}).get('architecture', {}).get('NumLayers', None),
            'Nn': self.config.get('model', {}).get('architecture', {}).get('NumNeurons', None),
            'Ni': getattr(self, 'input_dim', None), # NOTE: Is this correct?

            # BFM_WM_v2: train/valid split indices (into the concatenated training data), set by
            # WallModel.train(); None when unavailable. Upstream checkpoints do not contain these keys.
            # Stored as torch int64 tensors so that torch.load(weights_only=True) can still read the file.
            'train_index': self._index_tensor(getattr(self, 'train_index', None)),
            'valid_index': self._index_tensor(getattr(self, 'valid_index', None)),
        }

        # Add LDS parameters if used
        if self.config.get('model', {}).get('LDS', {}).get('lds', 0):
            save_dict['lds'] = self.config.get('model', {}).get('LDS', {}).get('lds', 0)
            save_dict['reweight'] = self.config.get('model', {}).get('LDS', {}).get('reweight', 'none')
            save_dict['lds_kernel'] = self.config.get('model', {}).get('LDS', {}).get('kernel', 'gaussian')
            save_dict['lds_ks'] = self.config.get('model', {}).get('LDS', {}).get('ks', 5)
            save_dict['lds_sigma'] = self.config.get('model', {}).get('LDS', {}).get('sigma', 2)
            save_dict['lds_num_bins'] = self.config.get('model', {}).get('LDS', {}).get('num_bins', 100)

        torch.save(save_dict, path)
        print(f"Model saved successfully to {path} (compact version)")

    def load_checkpoint(self, path: str) -> int:
        """
        Load training checkpoint to resume training

        Args:
            path: Path to the checkpoint file

        Returns:
            Epoch number after loading checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load history
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.valid_loss_history = checkpoint.get('valid_loss_history', [])
        self.best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))

        # Return the next epoch number
        return len(self.train_loss_history)

    def _keep_max_files(self, folder_path, max_files=5):
        # Get the list of all files in the folder
        files = glob.glob(os.path.join(folder_path, '*'))  # Use '*' to match all files

        # Check if the number of files exceeds the limit
        if len(files) > max_files:
            # Sort files by modification time (oldest first)
            files_sorted = sorted(files, key=os.path.getmtime)

            # Delete the oldest files if count exceeds max_files
            files_to_delete = len(files_sorted) - max_files
            for file in files_sorted[:files_to_delete]:
                os.remove(file)
                
