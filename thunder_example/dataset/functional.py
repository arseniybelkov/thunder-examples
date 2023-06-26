import numpy as np


def rotate_totalsegm(x: np.ndarray) -> np.ndarray:
    return np.flip(np.rot90(x), (1, -1))


def normalize_unit(image: np.ndarray, max_=200, min_=-200) -> np.ndarray:
    return np.clip((np.float32(image) - min_) / (max_ - min_), 0, 1)
