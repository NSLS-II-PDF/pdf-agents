import matplotlib.pyplot as plt
import numpy as np
from bluesky_adaptive.agents.sklearn import ClusterAgentBase
from databroker.client import BlueskyRun
from sklearn.cluster import KMeans

from .agents import PDFBaseAgent


class PassiveKmeansAgent(PDFBaseAgent, ClusterAgentBase):
    def __init__(self, k_clusters, *args, **kwargs):
        estimator = KMeans(k_clusters)
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
