import torch
import random
import numpy as np
import cv2


def random_seed(seed_value):
    # reproducibility
    # python RNG
    random.seed(seed_value)

    # numpy RNG
    np.random.seed(seed_value)

    # pytorch RNGs
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cv2.setRNGSeed(seed_value)
