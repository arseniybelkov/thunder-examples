import numpy as np
from connectome import Transform
from torch.utils.data import Dataset

from .functional import normalize_unit, rotate_totalsegm


class RotateTotalsegm(Transform):
    __inherit__ = True
    def image(image):
        return rotate_totalsegm(image)
    
    def liver(liver):
        return liver


class NormalizeCT(Transform):
    _max_: int = 150
    _min_: int = -50
    __inherit__ = True
    
    def image(image: np.ndarray, _max_: int, _min_: int) -> np.ndarray:
        return normalize_unit(image, _max_, _min_)


class ConToTorch(Dataset):
    def __init__(self, dataset, fields):
        self.loader = dataset._compile(fields)
        self.ids = dataset.ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.loader(self.ids[item])
