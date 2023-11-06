import random

import torch

LOWER_BOUND = -55
UPPER_BOUND = 155
def transforms_dla(img):
    lower_bound = LOWER_BOUND
    upper_bound = UPPER_BOUND
    lower_bound += random.randint(-5, 5)
    upper_bound += random.randint(-5, 5)
    img = torch.clamp(img, lower_bound, upper_bound)
    img = 2 * (img - lower_bound) / (upper_bound - lower_bound) - 1
    return img