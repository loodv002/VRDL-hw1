import torch.nn as nn

def check_n_parameters(model: nn.Module):
    n_parameters = sum(p.numel() for p in model.parameters())
    
    print(f'Number of params: {n_parameters:,}')
    
    assert n_parameters <= 100 * 1024 * 1024