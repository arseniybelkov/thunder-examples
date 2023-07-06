from typing import Callable, Iterable

from thunder.predict import Predictor
from toolz import compose


class Decorated(Predictor):
    def __init__(self, *decorators: Callable):
        super().__init__()
        self.decorators = compose(*decorators)
        
    def run(self, batches: Iterable, predict_fn: Callable) -> Iterable:
        return super().run(batches, self.decorators(predict_fn))
