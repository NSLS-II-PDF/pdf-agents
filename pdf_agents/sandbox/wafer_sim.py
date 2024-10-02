import logging
import time as ttime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from bluesky_adaptive.agents.sklearn import ClusterAgentBase
from numpy.polynomial.polynomial import polyfit, polyval
from numpy.typing import ArrayLike
from scipy.stats import rv_discrete
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from pdf_agents.scientific_value import ScientificValueAgentBase, scientific_value_function
from pdf_agents.utils import discretize, make_hashable, make_wafer_grid_list

logger = logging.getLogger(__name__)


def load_groundtruth(path: Path):
    dataset = xr.open_dataset(path)
    return dataset


class WaferAgentBase(ABC):
    def __init__(self, *, dataset: Union[xr.Dataset, Path, str], data_array_string: str = "iq", **kwargs):
        if isinstance(dataset, (Path, str)):
            dataset = load_groundtruth(dataset)
        self.ground_truth = dataset
        self._data_array = data_array_string
        self._doc_cache = []

    @abstractmethod
    def tell(self, x, y) -> Dict[str, ArrayLike]:
        """
        Tell the agent about some new data
        Parameters
        ----------
        x :
            Independent variable for data observed
        y :
            Dependent variable for data observed

        Returns
        -------
        dict
            Dictionary to be unpacked or added to a document

        """
        ...

    @abstractmethod
    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        """
        Ask the agent for a new batch of points to measure.

        Parameters
        ----------
        batch_size : int
            Number of new points to measure

        Returns
        -------
        docs : Sequence[dict]
            Documents of key metadata from the ask approach for each point in next_points.
            Must be length of batch size.
        next_points : Sequence
            Sequence of independent variables of length batch size
        """
        ...

    def _experiment_step(self):
        """Single round of ask for next point to measure, and observation of that point"""
        _docs, _next_points = self.ask(1)
        self._doc_cache.extend([("ask", doc) for doc in _docs])
        for point in _next_points:
            observation = self.ground_truth[self._data_array].interp({"x": point[0], "y": point[1]}).data
            if np.isnan(observation).any():
                raise ValueError(f"x,y input ({point[0]}, {point[1]}) is outside sample bounds")
            _doc = self.tell(point, observation)
            self._doc_cache.append(("tell", _doc))

    def experiment(self, n_steps: int, init_points=10):
        if init_points > 0:
            grid = make_wafer_grid_list(-25, 25, -25, 25, step=0.1)
            random_indicies = np.random.choice(grid.shape[0], size=init_points, replace=False)
            points = grid[random_indicies, :]
            for point in points:
                observation = self.ground_truth[self._data_array].interp({"x": point[0], "y": point[1]}).data
                _doc = self.tell(point, observation)
                self._doc_cache.append(("tell", _doc))

        for _ in range(n_steps):
            self._experiment_step()

    def measurement_plan(self, *args, **kwargs):
        return NotImplementedError("This test agent does not support/require a measurement plan")

    def unpack_run(self, *args, **kwargs):
        return NotImplementedError("This test agent does not support/require databroker integrations")


def current_dist_gen(x, px):
    """from distribution defined by p(x), produce a discrete generator.
    This helper function will normalize px as required, and return the generator ready for use.

    use:

    my_gen = current_dist(gen(x,px))

    my_gen.rvs() = xi # random variate of given type

    where xi is a random discrete value, taken from the set x, with probability px.

    my_gen.rvs(size=10) = np.array([xi1, xi2, ..., xi10]) # a size=10 array from distribution.

    If you want to return the probability mass function:

    my_gen.pmf

    See more in scipy.stats.rv_discrete
    """
    px[px < 0] = 0  # ensure non-negativitiy
    return rv_discrete(name="my_gen", values=(x, px / sum(px)))


def pick_from_distribution(x, px, num_picks=1):
    my_gen = current_dist_gen(x, px)
    if num_picks != 1:
        return my_gen.rvs(size=num_picks)
    else:
        return my_gen.rvs()


class WaferClusterAgent(WaferAgentBase):
    def __init__(self, *, k_clusters: int, resolution: float, bounds: ArrayLike, **kwargs):
        estimator = KMeans(k_clusters, n_init="auto")
        self.independent_cache = []
        self.observable_cache = []
        self.model = estimator
        self.resolution = resolution
        self.bounds = np.array(bounds)
        self.knowledge_cache = set()
        self.grid = make_wafer_grid_list(*self.bounds.ravel(), step=self.resolution)
        super().__init__(**kwargs)

    def tell(self, x, y) -> Dict[str, ArrayLike]:
        doc = ClusterAgentBase.tell(self, x, y)
        self.knowledge_cache.add(make_hashable(discretize(doc["independent_variable"], self.resolution)))
        return doc

    def _construct_model(self):
        try:
            sorted_independents, sorted_observables = zip(
                *sorted(zip(self.independent_cache, self.observable_cache))
            )
        except ValueError:
            # Multidimensional case
            sorted_independents, sorted_observables = zip(
                *sorted(zip(self.independent_cache, self.observable_cache), key=lambda x: (x[0][0], x[0][1]))
            )

        sorted_independents = np.array(sorted_independents)
        sorted_observables = np.array(sorted_observables)
        self.model.fit(sorted_observables)
        return sorted_independents, sorted_observables

    def _sample_uncertainty_proxy(self, batch_size=1):
        """Some Dan Olds magic to cast the distance from a cluster as an uncertainty. Then sample there

        Parameters
        ----------
        batch_size : int, optional

        Returns
        -------
        samples : ArrayLike
        centers : ArrayLike
            Kmeans centers for logging
        """
        # Borrowing from Dan's jupyter fun
        # from measurements, perform k-means
        sorted_independents, sorted_observables = self._construct_model()
        # retreive centers
        centers = self.model.cluster_centers_

        if self.bounds.size == 2:
            # One dimensional case, Use the Dan Olds approach
            # calculate distances of all measurements from the centers
            distances = self.model.transform(sorted_observables)
            # determine golf-score of each point (minimum value)
            min_landscape = distances.min(axis=1)
            # Assume a 1d scan
            # generate 'uncertainty weights' - as a polynomial fit of the golf-score for each point
            _x = np.arange(*self.bounds, self.motor_resolution)
            if batch_size is None:
                batch_size = len(_x)
            uwx = polyval(_x, polyfit(sorted_independents, min_landscape, deg=5))
            # Chose from the polynomial fit
            return pick_from_distribution(_x, uwx, num_picks=batch_size), centers
        else:
            # assume a 2d scan, use a linear model to predict the uncertainty
            labels = self.model.predict(sorted_observables)
            proby_preds = (
                LogisticRegression(solver="newton-cg").fit(sorted_independents, labels).predict_proba(self.grid)
            )  # TODO: NOTE THIS CHANGE
            shannon = -np.sum(proby_preds * np.log(proby_preds), axis=-1)  # TODO: NOTE THIS CHANGE
            top_indicies = (
                np.argsort(shannon)[::-1] if batch_size is None else np.argsort(shannon)[-batch_size:]
            )  # TODO: NOTE THIS CHANGE
            return self.grid[top_indicies], centers

    def ask(self, batch_size=1):
        """Get's a relative position from the agent. Returns a document and hashes the suggestion for redundancy"""
        suggestions, centers = self._sample_uncertainty_proxy(None)
        kept_suggestions = []
        if not isinstance(suggestions, Iterable):
            suggestions = [suggestions]
        # Keep non redundant suggestions and add to knowledge cache
        for suggestion in suggestions:
            hashable_suggestion = make_hashable(discretize(suggestion, self.resolution))
            if hashable_suggestion in self.knowledge_cache:
                logger.warn(
                    f"Suggestion {suggestion} is ignored as already in the knowledge cache: {hashable_suggestion}"
                )
                continue
            else:
                self.knowledge_cache.add(hashable_suggestion)
                kept_suggestions.append(suggestion)
            if len(kept_suggestions) >= batch_size:
                break

        base_doc = dict(
            cluster_centers=centers,
            cache_len=(
                len(self.independent_cache)
                if isinstance(self.independent_cache, list)
                else self.independent_cache.shape[0]
            ),
            requested_batch_size=batch_size,
            redundant_points_discarded=batch_size - len(kept_suggestions),
        )
        docs = [dict(suggestion=suggestion, **base_doc) for suggestion in kept_suggestions]

        return docs, kept_suggestions


class SVAgent(WaferAgentBase):
    def __init__(
        self,
        *,
        bounds: torch.Tensor,
        device: torch.device = None,
        num_restarts: int = 10,
        raw_samples: int = 128,
        observable_distance_function=None,
        ucb_beta=1.0,
        **kwargs,
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
        self.bounds = torch.tensor(bounds, device=self.device, dtype=torch.float).view(2, -1)

        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.ucb_beta = ucb_beta

    def ask(self, batch_size: int):
        return ScientificValueAgentBase.ask(self, batch_size)

    def tell(self, x, y):
        return ScientificValueAgentBase.tell(self, x, y)

    def report(self):
        return ScientificValueAgentBase.report(self)

    def _value_function(self, X, Y):
        return ScientificValueAgentBase._value_function(self, X, Y)


def kmeans_main(*, dataset, data_array_string, k_clusters, resolution, bounds, init_points=20, n_steps=200):
    time = ttime.time()
    agent = WaferClusterAgent(
        dataset=dataset,
        data_array_string=data_array_string,
        k_clusters=k_clusters,
        resolution=resolution,
        bounds=bounds,
    )
    agent.experiment(n_steps, init_points=init_points)
    print(f"Time taken: {ttime.time() - time}")
    print("Experimnent Done")
    return agent


def kmeans_plotting(agent):
    # Add model
    sorted_independents, sorted_observables = agent._construct_model()
    centers = agent.model.cluster_centers_
    labels = agent.model.predict(sorted_observables)
    proby_preds = LogisticRegression(solver="newton-cg").fit(sorted_independents, labels).predict_proba(agent.grid)
    shannon = -np.sum(proby_preds * np.log(proby_preds), axis=-1)
    num_classes = proby_preds.shape[1]
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    axes = axes.ravel()

    sc = axes[-1].scatter(*sorted_independents.T, c=labels, cmap="tab10")
    plt.colorbar(sc, ax=axes[-1], ticks=range(np.max(labels) + 1))
    axes[-1].set_title("Observations and Classes")
    sc = axes[-2].scatter(*zip(*agent.independent_cache), c=range(len(agent.independent_cache)), cmap="viridis")
    plt.colorbar(sc, ax=axes[-2])
    axes[-2].set_title("Observations and Order")
    for i, center in enumerate(centers):
        axes[-3].plot(center + i)
    axes[-3].set_title("Cluster Centers")

    # Add circle
    radius = 30
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    axes[-1].plot(x, y, color="black")
    axes[-2].plot(x, y, color="black")

    for i in range(num_classes):
        sc = axes[i].scatter(agent.grid[:, 0], agent.grid[:, 1], c=proby_preds[:, i], cmap="viridis", s=1)
        axes[i].set_title(f"Class {i+1} Probability")
        plt.colorbar(sc, ax=axes[i])
    sc_entropy = axes[num_classes].scatter(agent.grid[:, 0], agent.grid[:, 1], c=shannon, cmap="inferno", s=1)
    axes[num_classes].set_title("Shannon Entropy")
    plt.colorbar(sc_entropy, ax=axes[num_classes])

    for ax in axes:
        ax.set_aspect("equal")
    axes[-3].set_aspect("auto")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    agent = kmeans_main(
        dataset=Path(
            "/Users/phillipmaffettone/Development/beamline-profiles/pdf-agents/pdf_agents/scratch/ds_AlLiFe_complex_21Sep2024_12-04-04.nc"
        ).expanduser(),
        data_array_string="iq",
        k_clusters=6,
        resolution=0.05,
        bounds=[[-29, 29], [-29, 29]],
        init_points=200,
        n_steps=20,
    )
    print("Experimnent Done")

    # Plotting
    fig = kmeans_plotting(agent)
    fig.show()
    print("Plotting Done")
