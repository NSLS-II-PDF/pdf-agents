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


def make_wafer_grid_list(x_min, x_max, y_min, y_max, step):
    """
    Make the list of all of the possible 2d points that lie within a circle of the origin
    """
    x = np.arange(x_min, x_max, step)
    y = np.arange(y_min, y_max, step)
    xx, yy = np.meshgrid(x, y)
    center = np.array([x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2])
    distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    radius = min((x_max - x_min) / 2, (y_max - y_min) / 2)
    return np.array([xx[distance < radius], yy[distance < radius]]).T
