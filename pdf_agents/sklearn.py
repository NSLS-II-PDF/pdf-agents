import logging
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from bluesky_adaptive.agents.sklearn import ClusterAgentBase
from databroker.client import BlueskyRun
from numpy.polynomial.polynomial import polyfit, polyval
from numpy.typing import ArrayLike
from scipy.stats import rv_discrete
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .agents import PDFBaseAgent
from .utils import discretize, make_hashable, make_wafer_grid_list

logger = logging.getLogger(__name__)


class PassiveKmeansAgent(PDFBaseAgent, ClusterAgentBase):
    def __init__(self, k_clusters, *args, **kwargs):
        estimator = KMeans(k_clusters, n_init="auto")
        _default_kwargs = self.get_beamline_objects()
        _default_kwargs.update(kwargs)
        super().__init__(*args, estimator=estimator, **kwargs)

    def clear_caches(self):
        self.independent_cache = []
        self.dependent_cache = []

    def close_and_restart(self, *, clear_tell_cache=False, retell_all=False, reason=""):
        if clear_tell_cache:
            self.clear_caches()
        return super().close_and_restart(clear_tell_cache=clear_tell_cache, retell_all=retell_all, reason=reason)

    def server_registrations(self) -> None:
        self._register_method("clear_caches")
        return super().server_registrations()

    def tell(self, x, y):
        """Update tell using relative info"""
        x = x - self._motor_origins
        doc = super().tell(x, y)
        doc["absolute_position_offset"] = self._motor_origins
        return doc

    def report(self, **kwargs):
        arr = np.array(self.observable_cache)
        self.model.fit(arr)
        return dict(
            cluster_centers=self.model.cluster_centers_,
            cache_len=len(self.independent_cache),
            latest_data=self.tell_cache[-1],
        )

    @classmethod
    def hud_from_report(
        cls,
        run: BlueskyRun,
        report_idx=None,
        scaler: float = 1000.0,
        offset: float = 1.0,
        reorder_labels: bool = True,
    ):
        """Creates waterfall plot of spectra from a previously generated agent report.
        Waterfall plot of spectra will use 'scaler' to rescale each spectra prior to plotting.
        Waterfall plot will then use 'offset' to offset each spectra.

        Parameters
        ----------
        run : BlueskyRun
            Agent run to reference
        report_idx : int, optional
            Report index, by default most recent
        scaler : float, optional
            Rescaling of each spectra prior to plotting, by default 1000.0
        offset : float, optional
            Offset of plots to be tuned with scaler for waterfal, by default 1.0
        reorder_labels : bool, optional
            Optionally reorder the labelling so the first label appears first in the list, by default True

        Returns
        -------
        _type_
            _description_
        """
        _, data = cls.remodel_from_report(run, idx=report_idx)
        labels = data["clusters"]
        # distances = data["distances"]
        # cluster_centers = data["cluster_centers"]
        independent_vars = data["independent_vars"]
        observables = data["observables"]

        if reorder_labels:
            labels = cls.ordered_relabeling(labels)

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(2, 1, 1)
        for i in range(len(labels)):
            ax.scatter(independent_vars[i], labels[i], color=f"C{labels[i]}")
        ax.set_xlabel("measurement axis")
        ax.set_ylabel("K-means label")

        ax = fig.add_subplot(2, 1, 2)
        for i in range(len(observables)):
            plt.plot(
                np.arange(observables.shape[1]),
                scaler * observables[i] + i * offset,
                color=f"C{labels[i]}",
                alpha=0.1,
            )
        ax.set_xlabel("Dataset index")
        ax.set_ylabel("Intensity")
        fig.tight_layout()

        return fig

    @staticmethod
    def ordered_relabeling(x):
        """assume x is a list of labels,
        return same labeling structure, but with label names introduced sequentially.

        e.g. [4,4,1,1,2,1,3] -> [1,1,2,2,3,2,4]"""
        convert_dict = {}
        next_label = 0
        new_x = []
        for i in range(len(x)):
            if x[i] not in convert_dict.keys():
                convert_dict[x[i]] = next_label
                next_label += 1
            # x[i] = convert_dict[x[i]]

        for i in range(len(x)):
            new_x.append(convert_dict[x[i]])

        return new_x


class ActiveKmeansAgent(PassiveKmeansAgent):
    def __init__(self, *args, bounds: ArrayLike, **kwargs):
        super().__init__(*args, **kwargs)
        self._bounds = bounds
        self.knowledge_cache = set()  # Discretized knowledge cache of previously asked/told points

    @property
    def name(self):
        return "PDFActiveKMeans"

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value: ArrayLike):
        self._bounds = value

    def server_registrations(self) -> None:
        self._register_property("bounds")
        return super().server_registrations()

    def tell(self, x, y):
        """A tell that adds to the local discrete knowledge cache, as well as the standard caches"""
        doc = super().tell(x, y)
        self.knowledge_cache.add(make_hashable(discretize(doc["independent_variable"], self.motor_resolution)))
        doc["background"] = self.background
        return doc

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
            grid = make_wafer_grid_list(*self.bounds.ravel(), step=self.motor_resolution)
            labels = self.model.predict(sorted_observables)
            proby_preds = LogisticRegression().fit(sorted_independents, labels).predict_proba(grid)
            shannon = -np.sum(proby_preds * np.log(1 / proby_preds), axis=-1)
            top_indicies = np.argsort(shannon) if batch_size is None else np.argsort(shannon)[-batch_size:]
            return grid[top_indicies], centers

    def ask(self, batch_size=1):
        """Get's a relative position from the agent. Returns a document and hashes the suggestion for redundancy"""
        suggestions, centers = self._sample_uncertainty_proxy(None)
        kept_suggestions = []
        if not isinstance(suggestions, Iterable):
            suggestions = [suggestions]
        # Keep non redundant suggestions and add to knowledge cache
        for suggestion in suggestions:
            hashable_suggestion = make_hashable(discretize(suggestion, self.motor_resolution))
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
            latest_data=self.tell_cache[-1],
            requested_batch_size=batch_size,
            redundant_points_discarded=batch_size - len(kept_suggestions),
            absolute_position_offset=self._motor_origins,
        )
        docs = [dict(suggestion=suggestion, **base_doc) for suggestion in kept_suggestions]

        return docs, kept_suggestions

    def measurement_plan(self, relative_point: ArrayLike):
        """Send measurement plan absolute point from reltive position"""
        absolute_point = relative_point + self._motor_origins
        return super().measurement_plan(absolute_point)


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
