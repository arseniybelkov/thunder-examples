from functools import wraps
from typing import Callable, Iterable

import numpy as np
from thunder.predict import Predictor
from toolz import compose


class DecoratedPredictor(Predictor):
    def __init__(self, *decorators: Callable):
        super().__init__()
        self.decorators = compose(*decorators)
        
    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return super().run(batches, self.decorators(predict_fn))



def add_channels_dims(n: int = 1):
    def decorator(predict):
        @wraps(predict)
        def wrapper(*xs, **kwargs):
            return predict(*[np.stack([x[:, None, ...] for _ in range(n)], 1) for x in xs], **kwargs).squeeze(1)
        return wrapper
    return decorator
