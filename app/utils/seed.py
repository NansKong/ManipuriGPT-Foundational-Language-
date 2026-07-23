import random
from app.utils.logger import logger

def set_seed(seed: int) -> None:
    """
    Sets seed for reproducibility across random, numpy, and torch.
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    logger.info(f"Seed set for standard 'random': {seed}")
    
    try:
        import numpy as np
        np.random.seed(seed)
        logger.info(f"Seed set for 'numpy': {seed}")
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Seed set for 'torch': {seed}")
    except ImportError:
        pass
