import importlib
from abc import ABC
from logging import getLogger
from typing import Callable, Optional

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.spatial import distance_matrix

from .agents import PDFBaseAgent
from .sklearn import PassiveKmeansAgent

logger = getLogger("pdf_agents.scientific_value")


def scientific_value_function(X, Y, sd=None, multiplier=1.0, y_distance_function=None):
    """The value of two datasets, X and Y. Both X and Y must have the same
    number of rows. The returned result is a value of value for each of the
    data points.
    Parameters
    ----------
    X : numpy.ndarray
        The input data of shape N x d.
    Y : numpy.ndarray
        The output data of shape N x d'. Note that d and d' can be different
        and they also do not have to be 1.
    sd : float, optional
        Controls the length scale decay. We recommend this be set to ``None``
        to allow for automatic detection of the decay length scale(s).
    multiplier : float, optional
        Multiplies the automatically derived length scale if ``sd`` is
        ``None``.
    y_distance_function : callable, optional
        A callable function which takes the array ``Y`` as input and returns
        an N x N array in which the ith row and jth column is the distance
        measure between points i and j. Defaults to
        ``scipy.spatial.distance_matrix`` with its default kwargs (i.e. it is
        the L2 norm).
    Returns
    -------
    array_like
        The value for each data point.
    """

    X_dist = distance_matrix(X, X)

    if sd is None:
        distance = X_dist.copy()
        distance[distance == 0.0] = np.inf
        sd = distance.min(axis=1).reshape(1, -1) * multiplier

    # We can make this more pythonic but it makes sense in this case to keep
    # the default behavior explicit
    if y_distance_function is None:
        Y_dist = distance_matrix(Y, Y)
    else:
        Y_dist = y_distance_function(Y)

    v = Y_dist * np.exp(-(X_dist**2) / sd**2 / 2.0)

    return v.mean(axis=1)


class ScientificValueAgentBase(PDFBaseAgent, ABC):
    def __init__(
        self,
        *,
        bounds: torch.Tensor,
        device: torch.device = None,
        num_restarts: int = 10,
        raw_samples: int = 20,
        observable_distance_function: Optional[Callable] = None,
        ucb_beta=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.independent_cache = []
        self.observable_cache = []
        self.observable_distance_function = observable_distance_function

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.bounds = torch.tensor(bounds, device=self.device).view(2, -1)

        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.ucb_beta = ucb_beta

    def server_registrations(self) -> None:
        super().server_registrations()
        self._register_method("update_acquisition_function")

    def update_acquisition_function(self, acqf_name, **kwargs):
        module = importlib.import_module("botorch.acquisition")
        self.acqf_name = acqf_name
        self._partial_acqf = lambda gp: getattr(module, acqf_name)(gp, **kwargs)
        self.close_and_restart()

    def start(self, *args, **kwargs):
        _md = dict(acqf_name=self.acqf_name)
        self.metadata.update(_md)
        super().start(*args, **kwargs)

    def _value_function(self, X, Y):
        return scientific_value_function(X, Y, y_distance_function=self.observable_distance_function)

    def tell(self, x, y):
        return PassiveKmeansAgent().tell(x, y)

    def report(self):
        value = self._value_function(np.array(self.independent_cache), np.array(self.observable_cache))
        dict(latest_data=self.tell_cache[-1], cache_len=len(self.independent_cache), latest_value=value[-1])

    def ask(self, batch_size: int = 1):
        value = self._value_function(np.array(self.independent_cache), np.array(self.observable_cache))
        value = value.reshape(-1, 1)

        train_x = torch.tensor(self.independent_cache, dtype=torch.float, device=self.device)
        train_y = torch.tensor(value, dtype=torch.float, device=self.device)
        gp = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)
        fit_gpytorch_mll(mll)
        acq = (
            UpperConfidenceBound(gp, beta=self._beta).to(self.device)
            if batch_size == 1
            else qUpperConfidenceBound(gp, beta=self._beta).to(self.device)
        )
        candidates, acq_value = optimize_acqf(
            acq, bounds=self.bounds, q=batch_size, num_restarts=self.num_restarts, raw_samples=self.raw_samples
        )
        docs = [
            dict(
                candidate=candidate.detach().cpu().numpy(),
                acquisition_value=acq.detach().cpu().numpy(),
                latest_data=self.tell_cache[-1],
                cache_len=len(self.independent_cache),
                latest_value=value.squeeze()[-1],
            )
            for candidate, acq in zip(candidates, acq_value)
        ]
        return docs, torch.atleast_1d(candidates).detach().cpu().numpy().tolist()
