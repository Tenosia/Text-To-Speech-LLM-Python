import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional

def continue_training(checkpoint_path: str, model: DDP, optimizer: optim.Optimizer) -> int:
    """
    Load the latest checkpoints and optimizers for resuming training.
    
    Args:
        checkpoint_path: Directory containing checkpoint files
        model: DDP-wrapped model to load state into
        optimizer: Optimizer to load state into
        
    Returns:
        Starting epoch number (0 if no checkpoint found, or next epoch after loaded checkpoint)
    """
    if not os.path.exists(checkpoint_path):
        print(f'Checkpoint directory {checkpoint_path} does not exist. Starting from epoch 0.')
        return 0
    
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f'Checkpoint path {checkpoint_path} is not a directory.')
    
    model_dict = {}
    optimizer_dict = {}
    
    # glob all the checkpoints in the directory
    try:
        files = os.listdir(checkpoint_path)
    except PermissionError:
        raise PermissionError(f'Permission denied accessing checkpoint directory: {checkpoint_path}')
    
    for file in files:
        if file.endswith(".pt") and '_' in file:
            try:
                name, epoch_str = file.rsplit('_', 1)
                epoch = int(epoch_str.split('.')[0])
                
                if name.startswith("checkpoint"):
                    model_dict[epoch] = file
                elif name.startswith("optimizer"):
                    optimizer_dict[epoch] = file
            except (ValueError, IndexError) as e:
                print(f'Warning: Skipping invalid checkpoint file {file}: {e}')
                continue
    
    # get the largest epoch
    common_epochs = set(model_dict.keys()) & set(optimizer_dict.keys())
    if common_epochs:
        max_epoch = max(common_epochs)
        model_path = os.path.join(checkpoint_path, model_dict[max_epoch])
        optimizer_path = os.path.join(checkpoint_path, optimizer_dict[max_epoch])
        
        try:
            # load model and optimizer
            model_state = torch.load(model_path, map_location='cpu', weights_only=True)
            optimizer_state = torch.load(optimizer_path, map_location='cpu', weights_only=True)
            
            model.module.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            
            print(f'Resumed model and optimizer from epoch {max_epoch}')
            return max_epoch + 1
        except Exception as e:
            print(f'Error loading checkpoint from epoch {max_epoch}: {e}')
            print('Starting from epoch 0')
            return 0
    
    else:
        # load pretrained checkpoint (model only)
        if model_dict:
            max_epoch = max(model_dict.keys())
            model_path = os.path.join(checkpoint_path, model_dict[max_epoch])
            try:
                model_state = torch.load(model_path, map_location='cpu', weights_only=True)
                model.module.load_state_dict(model_state)
                print(f'Loaded pretrained model from epoch {max_epoch}. Starting training from epoch 0.')
            except Exception as e:
                print(f'Error loading pretrained checkpoint: {e}')
                print('Starting from epoch 0')
            
        return 0