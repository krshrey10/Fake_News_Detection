import random, os
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
