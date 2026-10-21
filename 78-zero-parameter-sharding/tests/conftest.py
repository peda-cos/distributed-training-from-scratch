import warnings

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy:.*")

import torch

torch.set_num_threads(1)
