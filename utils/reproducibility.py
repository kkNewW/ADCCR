import os
import random

import numpy as np
import torch


def set_global_seed(seed, deterministic=False):
    """Seed Python, NumPy and PyTorch in every launched process."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
