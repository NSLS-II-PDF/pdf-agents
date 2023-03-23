import logging
from typing import Iterable, List, Literal, Sequence, Tuple

import numpy as np
from bluesky_adaptive.agents.base import MonarchSubjectAgent
from bmm_agents.base import BMMBaseAgent
from numpy.typing import ArrayLike

from .sklearn import ActiveKmeansAgent
from .utils import discretize, make_hashable

logger = logging.getLogger(__name__)


class KMeansMonarchSubject(MonarchSubjectAgent, ActiveKmeansAgent):
    sample_position_motors = ("xafs_x", "xafs_y")

    def __init__(
        self,
        *args,
        pdf_origin: Tuple[float, float],
        filename: str,
        exp_mode: Literal["fluorescence", "transmission"],
        exp_data_type: Literal["chi", "mu"],
        elements: Sequence[str],
        edges: Sequence[str],
        element_origins: Sequence[Tuple[float, float]],
        element_det_positions: Sequence[float],
        sample: str = "Unknown",
        preparation: str = "Unknown",
        exp_bounds: str = "-200 -30 -10 25 12k",
        exp_steps: str = "10 2 0.3 0.05k",
        exp_times: str = "0.5 0.5 0.5 0.5",
        **kwargs,
    ):
        """
        Working in relative coordinates by default in all docs.
        The offsets are provided as extra data to relate back to plans, which are issued in absolute coordinates.

        # TODO: UPDATE HERE ALL
        Parameters
        ----------
        elements : Sequence[str]
            _description_
        element_origins : Sequence[Tuple[float, float]]
            _description_
        element_det_positions : Sequence[float]
            _description_
        pdf_origin: Tuple[float, float]
            _description_

        Example
        -------
        >>> agent = KMeansMonarchSubject(elements=["Cu", "Ti"],
        >>>         element_origins=[(155.390, 83.96), (155.381, 82.169)],
        >>>         element_det_positions=(205, 20),
        >>>         pdf_origin = (69.2, 2.0),
        >>>         bounds=[-30, 30])
        """
        self._filename = filename
        self._edges = edges
        self._exp_mode = exp_mode
        self._abscissa = exp_data_type
        self._ordinate = "k" if exp_data_type == "chi" else "energy"
        self._elements = elements
        self._element_origins = np.array(element_origins)
        self._element_det_positions = np.array(element_det_positions)

        self._sample = sample
        self._preparation = preparation
        self._exp_bounds = exp_bounds
        self._exp_steps = exp_steps
        self._exp_times = exp_times
        self.pdf_origin = np.array(pdf_origin)
        self.subject_knowledge_cache = set()  # Discretized knowledge cache of previously asked/told points
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        "KMeansPDFMonarchBMMSubject"

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value: str):
        self._filename = value

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value: Sequence[str]):
        self._edges = value

    @property
    def exp_mode(self):
        return self._exp_mode

    @exp_mode.setter
    def exp_mode(self, value: Literal["fluorescence", "transmission"]):
        self._exp_mode = value
        self.close_and_restart(clear_tell_cache=True)

    @property
    def exp_data_type(self):
        return self._abscissa

    @exp_data_type.setter
    def exp_data_type(self, value: Literal["chi", "mu"]):
        self._abscissa = value
        self._ordinate = "k" if value == "chi" else "energy"
        self.close_and_restart(clear_tell_cache=True)

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, value: Sequence[str]):
        self._elements = value

    @property
    def element_origins(self):
        return self._element_origins

    @element_origins.setter
    def element_origins(self, value: Sequence[Tuple[float, float]]):
        self._elements_origins = np.array(value)

    @property
    def element_det_positions(self):
        return self._element_det_positions

    @element_det_positions.setter
    def element_det_positions(self, value: Sequence[float]):
        self._element_det_positions = np.array(value)

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, value: Tuple[float, float]):
        self._roi = value
        self.close_and_restart(clear_tell_cache=True)

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value: str):
        self._sample = value

    @property
    def preparation(self):
        return self._filename

    @preparation.setter
    def preparation(self, value: str):
        self._preparation = value

    @property
    def exp_bounds(self):
        return self._exp_bounds

    @exp_bounds.setter
    def exp_bounds(self, value: str):
        self._exp_bounds = value

    @property
    def exp_steps(self):
        return self._exp_steps

    @exp_steps.setter
    def exp_steps(self, value: str):
        self._exp_steps = value

    @property
    def exp_times(self):
        return self._exp_times

    @exp_times.setter
    def exp_times(self, value: str):
        self._exp_times = value

    def server_registrations(self) -> None:
        # This ensures relevant properties are in the rest API
        self._register_property("filename")
        self._register_property("elements")
        self._register_property("element_origins")
        self._register_property("element_det_positions")
        self._register_property("exp_data_type")
        self._register_property("exp_mode")
        self._register_property("roi")
        self._register_property("sample")
        self._register_property("preparation")
        self._register_property("exp_bounds")
        self._register_property("exp_steps")
        self._register_property("exp_times")
        return super().server_registrations()

    def subject_measurement_plan(self, relative_point: ArrayLike) -> Tuple[str, List, dict]:
        return BMMBaseAgent.measurement_plan(self, relative_point)

    def subject_ask(self, batch_size=1) -> Tuple[Sequence[dict[str, ArrayLike]], Sequence[ArrayLike]]:
        """Copy default ask with minor modifications for BMM and subject cache"""
        suggestions, centers = self._sample_uncertainty_proxy(batch_size)
        kept_suggestions = []
        if not isinstance(suggestions, Iterable):
            suggestions = [suggestions]
        # Keep non redundant suggestions and add to knowledge cache
        for suggestion in suggestions:
            if suggestion in self.subject_knowledge_cache:
                logger.info(f"Suggestion {suggestion} is ignored as already in the subject knowledge cache")
                continue
            else:
                self.subject_knowledge_cache.add(make_hashable(discretize(suggestion, self.motor_resolution)))
                kept_suggestions.append(suggestion)
        _default_doc = dict(
            elements=self.elements,
            element_origins=self.element_origins,
            element_det_positions=self.element_det_positions,
            cluster_centers=centers,
            cache_len=(
                len(self.independent_cache)
                if isinstance(self.independent_cache, list)
                else self.independent_cache.shape[0]
            ),
            latest_data=self.tell_cache[-1],
            requested_batch_size=batch_size,
            redundant_points_discarded=batch_size - len(kept_suggestions),
        )
        docs = [dict(suggestion=suggestion, **_default_doc) for suggestion in kept_suggestions]
        return docs, suggestions

    def tell(self, x, y):
        """Update tell using relative info"""
        x = x - self.pdf_origin[0]
        doc = super().tell(x, y)
        doc["absolute_position_offset"] = self.pdf_origin[0]
        return doc

    def ask(self, batch_size=1) -> Tuple[Sequence[dict[str, ArrayLike]], Sequence[ArrayLike]]:
        """Update ask with relative info"""
        docs, suggestions = super().ask(batch_size=batch_size)
        for doc in docs:
            doc["absolute_position_offset"] = self.pdf_origin[0]
        return docs, suggestions

    def measurement_plan(self, relative_point: ArrayLike) -> Tuple[str, List, dict]:
        """Send measurement plan absolute point from reltive position"""
        absolute_point = relative_point + self.pdf_origin[0]
        return super().measurement_plan(absolute_point)
