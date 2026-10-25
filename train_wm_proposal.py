import parsing_wm
from src.wall_model import WallModel
import sys
import random
import numpy as np
import torch

# Get the config file name from command line arguments
config_file = sys.argv[1]
config = parsing_wm.parse_toml_config('./inputfiles/' + config_file)
gpu_num = config.get('general', {}).get('GpuNum', -1)
if gpu_num >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_num}')
else:
    # BFM_WM_v2: raise a proper exception (upstream raised a string, which is itself a TypeError)
    raise RuntimeError("device is incorrect")

# BFM_WM_v2: optional [general] Seed makes the random train/valid split and the weight initialisation
# reproducible (upstream: unseeded). The split indices are stored in the saved checkpoints.
seed = config.get('general', {}).get('Seed', None)
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Restore from a checkpoint if provided
checkpoint = sys.argv[2] if len(sys.argv) > 2 else None
if checkpoint:
    # NOTE: Here for continual training, we assume that we are working on a new dataset using EWC
    # So by default, we also need to read in FIM and using it to add the EWC loss
    checkpoint_path = './models/' + checkpoint
    training_instance = WallModel.load(checkpoint_path, device=device)
    training_instance.override_config(config, checkpoint)
else:
    # Training from scratch
    training_instance = WallModel(config)

# Load and train
training_instance.load_data()
training_instance.train()

# Save testing results
r2_train, r2_valid = training_instance.test(save_path='./results')
