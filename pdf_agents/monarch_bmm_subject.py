from typing import List, Sequence, Tuple

import numpy as np
from bluesky_adaptive.agents.base import MonarchSubjectAgent
from matplotlib import ArrayLike

from .sklearn import ActiveKmeansAgent


class KMeansMonarchSubject(MonarchSubjectAgent, ActiveKmeansAgent):
    bmm_measurement_plan_name = "agent_move_and_measure"
    bmm_sample_position_motors = ("xafs_x", "xafs_y")

    def __init__(
        self,
        *args,
        elements: Sequence[str],
        element_origins: Sequence[Tuple[float, float]],
        element_det_positions: Sequence[float],
        pdf_origin=Tuple[float, float],
        **kwargs,
    ):
        """
        Working in relative coordinates by default in all docs.
        The offsets are provided as extra data to relate back to plans, which are issued in absolute coordinates.

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
        # TODO: Make these parameters
        self.elements = elements
        self.element_origins = np.array(element_origins)
        self.element_det_positions = np.array(element_det_positions)
        self.pdf_origin = np.array(pdf_origin)

    def subject_measurement_plan(self, relative_point: ArrayLike) -> Tuple[str, List, dict]:
        """Transform relative position into absolute position for plans"""
        args = [
            self.bmm_sample_position_motors[0],
            *(self.element_origins[:, 0] + relative_point),
            self.bmm_sample_position_motors[1],
            *self.element_origins[:, 1],
        ]

        kwargs = dict(
            filename="MultimodalMadnessSpring23",
            nscans=1,
            start="next",
            mode="fluorescence",
            edge="K",
            sample="CuTi",
            preparation="film sputtered on silica",
            bounds="-200 -30 -10 25 12k",
            steps="10 2 0.3 0.05k",
            times="0.5 0.5 0.5 0.5",
            snapshots=False,
            md={"relative_position": relative_point},
        )
        kwargs.update(
            {
                f"{element}_det_position": det_position
                for element, det_position in zip(self.elements, self.element_det_positions)
            }
        )

        return self.bmm_measurement_plan_name, args, kwargs

    def subject_ask(self, batch_size=1):
        """Use default ask with minor modifications for BMM"""
        doc, suggestions = super().ask(batch_size=batch_size)
        doc["elements"] = self.elements
        doc["element_origins"] = self.element_origins
        doc["element_det_positions"] = self.element_det_positions
        return doc, suggestions

    def tell(self, x, y):
        """Update tell using relative info"""
        x = x - self.pdf_origin[0]
        doc = super().tell(x, y)
        doc["absolute_position_offset"] = self.pdf_origin[0]
        return doc

    def ask(self, batch_size=1):
        """Update ask with relative info"""
        doc, suggestions = super().ask(batch_size=batch_size)
        doc["absolute_position_offset"] = self.pdf_origin[0]
        return doc, suggestions

    def measurement_plan(self, relative_point: ArrayLike) -> Tuple[str, List, dict]:
        """Send measurement plan absolute point"""
        absolute_point = relative_point + self.pdf_origin[0]
        return super().measurement_plan(absolute_point)
