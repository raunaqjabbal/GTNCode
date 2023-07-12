import random as _random
import os as _os
import numpy as _np
import torch as _torch
from collections import defaultdict as _defaultdict
import matplotlib.pyplot as _plt                        # type: ignore
from torchvision.utils import make_grid as _make_grid   # type: ignore

def _seed_everything(seed=1234):
    ''' Sets seed for reproducibility.'''
    _random.seed(seed)
    _os.environ['PYTHONHASHSEED'] = str(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed(seed)
    _torch.backends.cudnn.deterministic = True

def _recursive_detach(target):
    ''' Recursively detaches objects.
    '''
    if isinstance(target, _torch.Tensor):
        target = target.detach()
        return target
    elif isinstance(target,dict):
        return {k:_recursive_detach(v) for k,v in target.items()}
    elif isinstance(target,list):
        return [_recursive_detach(v) for v in target]
    elif isinstance(target,tuple):
        return tuple(_recursive_detach(v) for v in target)
    else:
        return target

    
def _diffopt_state_dict(diffopt):
    ''' Makes optimizer stateful. Converts `diffopt` back to `inner_optim`
    '''
    param_mappings = {}
    start_index = 0
    with _torch.no_grad():
        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index) if id(p) not in param_mappings})
            
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed

        res = _defaultdict(dict)
        param_groups = [pack_group(g) for g in diffopt.param_groups]
        for group_idx, group in enumerate(diffopt.param_groups):
            for p_idx, p in enumerate(group['params']):
                res[p] = { k:v for k,v in diffopt.state[group_idx][p_idx].items() }
        
        param_groups = _recursive_detach(param_groups)
        packed_state = {(param_mappings[id(k)] if isinstance(k, _torch.Tensor) else k): _recursive_detach(v) for k, v in res.items()}
        return {'state':packed_state,'param_groups':param_groups}
    
def _imshow(images,num_classes=10, r=False):          
    ''' Displays a grid of images.
    '''  
    images = images.detach().cpu()
    images = (images + 1) / 2 
    image_grid = _make_grid(images, nrow= 2 * num_classes)
    _plt.figure(figsize=(20,12))
    _plt.imshow(image_grid.permute(1, 2, 0),cmap='gray')
    _plt.axis("off")
    _plt.show()
    
def _weights_init(m):
    ''' Initializes weights of the model.
    '''
    if isinstance(m, _torch.nn.Linear):
        _torch.nn.init.kaiming_uniform_(m.weight)
    if isinstance(m, _torch.nn.ConvTranspose2d):
        _torch.nn.init.kaiming_uniform_(m.weight)
    if isinstance(m, _torch.nn.Conv2d):
        _torch.nn.init.kaiming_uniform_(m.weight)

def _cycle(iterable):
    ''' Makes iterable cyclic in nature.
    '''
    while True:
        for x in iterable:
            yield x
            
def _divide_chunks(seq, n):
    ''' divides an array into sequences each of length `n`
    '''
    for i in range(0, len(seq), n): 
        yield seq[i:i + n]
  