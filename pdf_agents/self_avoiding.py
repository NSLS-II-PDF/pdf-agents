import numpy as np
import scipy.ndimage as sndi
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple


def compute_prob(a, *, sigma=20, gamma=2, bonus=None):
    if (a != 0).all():
        dist = np.ones_like(a)
    else:
        dist = sndi.distance_transform_edt(a)
    if gamma != 1:
        dist **= gamma

    dist = 1-np.exp(-dist/ (2*sigma*sigma))

    if bonus is not None:
        dist += bonus

    dist[np.isnan(a)] = 0
    dist[dist<=0] = 0
    dist /= dist.sum()

    return dist


def pick_next_point(a, N=1, *, gamma=1, sigma=20, bonus=None):
    dist = compute_prob(a, gamma=gamma, sigma=sigma, bonus=bonus)
    cdf = np.cumsum(dist.ravel())
    r = np.random.rand(N)
    idx = np.searchsorted(cdf, r, side='right')
    return np.unravel_index(idx, a.shape)


def show_dan(a, extent=None, *, gamma=1, sigma=20, bonus=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained", sharex=True, sharey=True, figsize=(8, 4))
    ax1.imshow(a, origin="lower", extent=extent)
    d = compute_prob(a, gamma=gamma, sigma=sigma, bonus=bonus)
    cmap = mpl.colormaps['viridis']
    cmap.set_under('w')
    im = ax2.imshow(d, origin="lower", vmin=1e-25, extent=extent, interpolation_stage='rgba', cmap=cmap)

    fig.colorbar(im, extend='min')

    return fig, (ax1, ax2)


class WaferManager:
    def __init__(self, radius: float, resolution: float, center: Tuple[float, float]):
        """
        Parameters
        ----------
        radius : float
            radius of wafer.

            In same units as resolution and center.
        resolution : float
            scale to quantize the space on.

            In same units as radius and center
        center : Tuple[float, float]
            location of center of wafer in motor coordiates

            In same units as radius and resoultion
        """
        N = int(np.ceil(2 * radius / resolution))

        self._measured = np.ones((N, N))
        self._center = np.array(center)
        # 0, 0 in the mask is the lower outboard
        self._ll = self._center - radius
        self._resolution = resolution
        self._radius = radius
        self._mask = ~(np.hypot(*np.ogrid[-N//2:N//2, -N//2:N//2]) < N//2) 
        
        self._measured[self._mask] = np.nan
        self.gamma = 2
        self.sigma = 20
        self.extra = np.zeros_like(self._measured)

    def xy_to_index(self, xy):
        xy = np.atleast_1d(xy)
        rel_xy = xy - self._ll
        return tuple((rel_xy[::-1] // self._resolution).astype("int"))

    def add_measurement(self, xy):

        indx = self.xy_to_index(xy)
        self._measured[indx] = 0

    def debug_vis(self):
        center_x, center_y = self._center

        return show_dan(
            self._measured,
            extent=[
                center_x - self._radius,
                center_x + self._radius,
                center_y - self._radius,
                center_y + self._radius,
            ],
            gamma=self.gamma,
            sigma=self.sigma,
            bonus=self.extra
        )

    def pick_next_point(self, N=1):
        y, x = pick_next_point(self._measured, N=N, gamma=self.gamma, sigma=self.sigma, bonus=self.extra)
        return (np.stack([x, y]).T * self._resolution + self._ll).T


wm = WaferManager(40, 0.4, (-130, 50))
wm.extra[50:100, 50:100] = -100
wm.add_measurement((-110, 50))
fig, (ax1, ax2) = wm.debug_vis()
ax2.plot(*wm.pick_next_point(N=100), 'o')

for j in range(15):
    (x,), (y,) = wm.pick_next_point(N=1)
    wm.add_measurement((x, y))
    fig, (ax1, ax2) = wm.debug_vis()

    ax2.plot(*wm.pick_next_point(N=100), 'o')