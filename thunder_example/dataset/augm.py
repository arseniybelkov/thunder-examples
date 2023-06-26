from typing import Tuple

import numpy as np
from connectome import Transform, impure
from imops import crop_to_shape


class RandomPatch(Transform):
    _patch_size: Tuple[int, int, int]
    __inherit__ = True
    
    @impure
    def _ratio() -> np.ndarray:
        return np.random.uniform(size=3)
    
    def image(image, _patch_size, _ratio) -> np.ndarray:
        return crop_to_shape(image, _patch_size, ratio=_ratio)
    
    def liver(liver, _patch_size, _ratio) -> np.ndarray:
        return crop_to_shape(liver, _patch_size, ratio=_ratio)
