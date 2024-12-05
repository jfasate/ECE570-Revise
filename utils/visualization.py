import matplotlib.pyplot as plt
import numpy as np
import torch
from model.module.dgconv import DGConv2d, _aggregate, _kronecker_product

import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from common import Config

def plot_layer_statistics(models, save_path=None):
    """
    Creates visualization of DGConv layer statistics for multiple model replicates.
    
    Args:
        models: List of trained models containing DGConv2d layers
        save_path: Optional path to save the generated plot
    """
    all_layer_stats = []
    layer_names = None
    
    # Collect statistics from each model
    for model in models:
        layer_stats = []
        current_names = []
        
        for name, module in model.named_modules():
            if isinstance(module, DGConv2d):
                gate = torch.stack((1 - module.gate, module.gate))
                U, _ = _aggregate(gate, module.D, module.I, module.K, sort=module.sort)
                
                if module.out_channels // module.in_channels >= 2:
                    U = torch.mm(module._I, U)
                elif module.in_channels // module.out_channels >= 2:
                    U = torch.mm(U, module._I)
                
                U = U[:module.out_channels, :module.in_channels]
                
                if module.groups > 1:
                    U = U * module.group_mask
                
                U = U.detach().cpu().numpy()
                
                sparsity = (np.abs(U) < 1e-4).sum() / U.size
                unique_patterns = len(np.unique(U.round(decimals=4), axis=0))
                groups = max(1, unique_patterns)
                
                layer_stats.append((sparsity, groups))
                current_names.append(name)
        
        all_layer_stats.append(layer_stats)
        if layer_names is None:
            layer_names = current_names
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Plot sparsity levels
    for i, stats in enumerate(all_layer_stats):
        ax1.plot(range(len(layer_names)), [s[0] for s in stats], 
                f'{colors[i]}-', label=f'replicate {i+1}', linewidth=2)
    
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45)
    ax1.set_ylabel('Sparsity')
    ax1.set_title('Sparsity Level in Different Layers')
    ax1.grid(True)
    ax1.legend()
    
    # Plot number of groups 
    for i, stats in enumerate(all_layer_stats):
        ax2.plot(range(len(layer_names)), [s[1] for s in stats],
                f'{colors[i]}-', label=f'replicate {i+1}', linewidth=2)
    
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45)
    ax2.set_ylabel('#Groups')
    ax2.set_title('Learned Conv Groups in Different Layers')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()