import numpy as np


class OfflineKafka:
    def set_agent(self, *args, **kwargs):
        pass

    def subscribe(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass


def discretize(value: np.typing.ArrayLike, resolution: np.typing.ArrayLike):
    return np.floor(value / resolution)


def make_hashable(x):
    try:
        return tuple(map(float, x))
    except TypeError:
        return float(x)
