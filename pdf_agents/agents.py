import ast
import uuid
from abc import ABC
from logging import getLogger
from typing import Dict, List, Optional, Sequence, Tuple, Union

import nslsii.kafka_utils
import numpy as np
import redis
import tiled
from bluesky_adaptive.agents.base import Agent, AgentConsumer
from bluesky_adaptive.agents.simple import SequentialAgentBase
from bluesky_kafka import Publisher
from bluesky_queueserver_api.http import REManagerAPI
from numpy.typing import ArrayLike

from .utils import OfflineKafka

logger = getLogger(__name__)


class PDFBaseAgent(Agent, ABC):
    def __init__(
        self,
        *args,
        motor_names: List[str] = ["xstage", "ystage"],
        motor_origins: List[float] = [0.0, 0.0],
        motor_resolution: float = 0.2,  # mm
        data_key: str = "chi_I",
        roi_key: str = "chi_Q",
        roi: Optional[Tuple] = None,
        norm_region: Optional[Tuple] = None,
        offline=False,
        **kwargs,
    ):
        if offline:
            self._rkvs = {"PDF:desired_exposure_time": 1.0, "PDF:xpdacq:sample_number": 1}
            for key, val in self._rkvs.items():
                self._rkvs[key] = str(val).encode("utf-8")
        else:
            self._rkvs = redis.Redis(host="info.pdf.nsls2.bnl.gov", port=6379, db=0)  # redis key value store
        self._motor_names = motor_names
        self._motor_resolution = motor_resolution
        self._motor_origins = np.array(motor_origins)
        self._data_key = data_key
        self._roi_key = roi_key
        self._roi = roi
        self._norm_region = norm_region
        self._ordinate = None
        # Attributes pulled in from Redis
        self._exposure = float(self._rkvs.get("PDF:desired_exposure_time").decode("utf-8"))
        self._sample_number = int(self._rkvs.get("PDF:xpdacq:sample_number").decode("utf-8"))
        try:
            self._background = np.array(
                [
                    ast.literal_eval(self._rkvs.get("PDF:bgd:x").decode("utf-8")),
                    ast.literal_eval(self._rkvs.get("PDF:bgd:y").decode("utf-8")),
                ]
            )
        except AttributeError:
            # None available in redis
            self._background = np.zeros((2,))
        if offline:
            _default_kwargs = self.get_offline_objects()
        else:
            _default_kwargs = self.get_beamline_objects()
        _default_kwargs.update(kwargs)
        md = dict(
            motor_names=self.motor_names,
            motor_resolution=self.motor_resolution,
            data_key=self.data_key,
            roi_key=self.roi_key,
            roi=self.roi,
        )
        super().__init__(*args, metadata=md, **_default_kwargs)

    def measurement_plan(self, point: ArrayLike) -> Tuple[str, List, Dict]:
        """Default measurement plan is an agent modified simple count  of pe1c,
        that uses redis to fill in key values like exposure time and sample number.

        Parameters
        ----------
        point : ArrayLike
            Next point to measure using a given plan, given in absolute coordinates.

        Returns
        -------
        plan_name : str
        plan_args : List
            List of arguments to pass to plan from a point to measure.
        plan_kwargs : dict
            Dictionary of keyword arguments to pass the plan, from a point to measure.
        """
        return "agent_move_and_measure_hanukkah23", [], {"x": point[0], "y": point[1], "exposure": 5}

    def unpack_run(self, run) -> Tuple[Union[float, ArrayLike], Union[float, ArrayLike]]:
        """Subtracts background and returns motor positions and data"""
        y = run.primary.data[self.data_key].read().flatten()
        if self.background is not None:
            y = y - self.background[1]

        if self.norm_region is not None:
            ordinate = np.array(run.primary.data[self.roi_key]).flatten()
            idx_min = (
                np.where(ordinate < self.norm_region[0])[0][-1]
                if len(np.where(ordinate < self.norm_region[0])[0])
                else None
            )
            idx_max = (
                np.where(ordinate > self.norm_region[1])[0][-1]
                if len(np.where(ordinate > self.norm_region[1])[0])
                else None
            )
            bkg_idx_min = (
                np.where(self.background[0] < self.norm_region[0])[0][-1]
                if len(np.where(self.background[0] < self.norm_region[0])[0])
                else None
            )
            bkg_idx_max = (
                np.where(self.background[0] > self.norm_region[1])[0][-1]
                if len(np.where(self.background[0] > self.norm_region[1])[0])
                else None
            )
            scale_factor = np.sum(self.background[1][bkg_idx_min:bkg_idx_max]) / np.sum(y[idx_min:idx_max])
        else:
            scale_factor = 1

        y = y * scale_factor

        if self.roi is not None:
            ordinate = np.array(run.primary.data[self.roi_key]).flatten()
            idx_min = np.where(ordinate < self.roi[0])[0][-1] if len(np.where(ordinate < self.roi[0])[0]) else None
            idx_max = np.where(ordinate > self.roi[1])[0][-1] if len(np.where(ordinate > self.roi[1])[0]) else None

        y = y[idx_min:idx_max]
        self._ordinate = ordinate[idx_min:idx_max]  # Update self oridnate. Should be constant unless roi changes.

        try:
            x = np.array(
                [
                    run.start["more_info"][motor_name][f"OT_stage_2_{motor_name[0].upper()}"]["value"]
                    for motor_name in self.motor_names
                ]
            )
        except KeyError:
            x = np.array([run.start[motor_name][motor_name]["value"] for motor_name in self.motor_names])
        return x, y

    def server_registrations(self) -> None:
        self._register_property("motor_resolution")
        self._register_property("motor_names")
        self._register_property("exposure_time")
        self._register_property("sample_number")
        self._register_property("data_key")
        self._register_property("roi_key")
        self._register_property("roi")
        self._register_property("background")
        self._register_property("norm_region")
        return super().server_registrations()

    @property
    def motor_names(self):
        """Name of motor to be used as the independent variable in the experiment"""
        return self._motor_names

    @motor_names.setter
    def motor_names(self, value: str):
        self._motor_names = value

    @property
    def motor_resolution(self):
        """Minimum resolution for measurement in milimeters, i.e. (beam width)/2"""
        return self._motor_resolution

    @motor_resolution.setter
    def motor_resolution(self, value: float):
        self._motor_resolution = value

    # @property
    # def exposure_time(self):
    #     """Exposure time of scans in seconds"""
    #     value = float(self._rkvs.get("PDF:desired_exposure_time").decode("utf-8"))
    #     if value != self._exposure:
    #         logger.warning(
    #             f"Mismatch between agent exposure time ({self._exposure}) and redis value {value}. "
    #             "Updating to redis value."
    #         )
    #         self._exposure = value
    #     return self._exposure

    # @exposure_time.setter
    # def exposure_time(self, value: float):
    #     self._exposure = value
    #     self._rkvs.set("PDF:desired_exposure_time", value)

    # @property
    # def sample_number(self):
    #     """XPDAQ Sample Number"""
    #     value = int(self._rkvs.get("PDF:xpdacq:sample_number").decode("utf-8"))
    #     if value != self._sample_number:
    #         logger.warning(
    #             f"Mismatch between agent sample_number ({self._sample_number}) and redis value {value}. "
    #             "Updating to redis value."
    #         )
    #         self._sample_number = value
    #     return self._sample_number

    # @sample_number.setter
    # def sample_number(self, value: int):
    #     self._sample_number = value
    #     self._rkvs.set("PDF:xpdacq:sample_number", value)

    @property
    def background(self):
        try:
            self._background = np.array(
                [
                    ast.literal_eval(self._rkvs.get("PDF:bgd:x").decode("utf-8")),
                    ast.literal_eval(self._rkvs.get("PDF:bgd:y").decode("utf-8")),
                ]
            )
        except AttributeError:
            self._background = np.zeros((2,))
        return self._background

    # @background.setter
    # def background(self, arr):
    #     self._rkvs.set("PDF:bgd:x", str(list(arr[0, :])))
    #     self._rkvs.set("PDF:bgd:y", str(list(arr[1, :])))
    #     self._background = np.array(arr)
    #     if self._background.shape[0] != 2:
    #         raise ValueError("Background array should have shape [2, N]")

    @property
    def data_key(self):
        return self._data_key

    @data_key.setter
    def data_key(self, value: str):
        self._data_key = value
        self.close_and_restart(clear_tell_cache=True)

    @property
    def roi_key(self):
        return self._roi_key

    @roi_key.setter
    def roi_key(self, value: str):
        self._roi_key = value
        self.close_and_restart(clear_tell_cache=True)

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, value: Tuple[float, float]):
        self._roi = value
        self.close_and_restart(clear_tell_cache=True)

    @property
    def norm_region(self):
        return self._norm_region

    @norm_region.setter
    def norm_region(self, value: Tuple[float, float]):
        self._norm_region = value
        self.close_and_restart(clear_tell_cache=True)

    @staticmethod
    def get_beamline_objects() -> dict:
        beamline_tla = "pdf"
        kafka_config = nslsii.kafka_utils._read_bluesky_kafka_config_file(
            config_file_path="/etc/bluesky/kafka.yml"
        )
        qs = REManagerAPI(http_server_uri=f"https://qserver.nsls2.bnl.gov/{beamline_tla}")
        qs.set_authorization_key(api_key="yyyyy")

        kafka_consumer = AgentConsumer(
            topics=[
                f"{beamline_tla}.bluesky.pdfstream.documents",
            ],
            consumer_config=kafka_config["runengine_producer_config"],
            bootstrap_servers=",".join(kafka_config["bootstrap_servers"]),
            group_id=f"echo-{beamline_tla}-{str(uuid.uuid4())[:8]}",
        )

        kafka_producer = Publisher(
            topic=f"{beamline_tla}.mmm.bluesky.adjudicators",
            bootstrap_servers=",".join(kafka_config["bootstrap_servers"]),
            key="{beamline_tla}.key",
            producer_config=kafka_config["runengine_producer_config"],
        )

        return dict(
            kafka_consumer=kafka_consumer,
            kafka_producer=kafka_producer,
            tiled_data_node=tiled.client.from_uri(
                "https://tiled.nsls2.bnl.gov/api/v1/metadata/pdf/bluesky_sandbox"
            ),
            tiled_agent_node=tiled.client.from_uri(
                "https://tiled.nsls2.bnl.gov/api/v1/metadata/pdf/bluesky_sandbox"
            ),
            qserver=qs,
        )

    @staticmethod
    def get_offline_objects() -> dict:
        """Objects to spin up agent with access to only tiled if available."""
        beamline_tla = "pdf"
        offline_kafka = OfflineKafka()
        try:
            node = tiled.client.from_profile(f"{beamline_tla}_bluesky_sandbox")
        except tiled.profiles.ProfileNotFound:
            node = None

        return dict(
            kafka_consumer=offline_kafka,
            kafka_producer=offline_kafka,
            tiled_data_node=node,
            tiled_agent_node=node,
            qserver=None,
        )

    def trigger_condition(self, uid) -> bool:
        return True


class PDFSequentialAgent(PDFBaseAgent, SequentialAgentBase):
    def __init__(
        self,
        *,
        sequence: Sequence[Union[float, ArrayLike]],
        relative_bounds: Tuple[Union[float, ArrayLike]] = None,
        **kwargs,
    ) -> None:
        super().__init__(sequence=sequence, relative_bounds=relative_bounds, **kwargs)

    def tell(self, x, y) -> Dict[str, ArrayLike]:
        doc = super().tell(x, y)
        doc["background"] = self.background
        return doc


class PDFReporterMixin:
    """Mixin for sending reports to Kafka as well as Tiled.
    This wraps every report in a single run because downstream agents operate per-run instead of per-event.
    This behavior is similar to the pdfstream service.

    Parameters
    ----------
    report_producer : Publisher
        Bluesky Kafka publisher to produce document stream of agent reports.

    Examples
    --------
    >>> class PassiveKmeansAgentReporter(PDFReporterMixin, PassiveKmeansAgent)
    >>> agent = PassiveKmeansAgentReporter(report_producer=Publisher(...), k_clusters=3)
    """

    def __init__(self, *args, report_producer: Publisher, **kwargs):
        self._report_producer = report_producer
        super().__init__(*args, **kwargs)

    def generate_report(self, **kwargs):
        doc = self.report(**kwargs)
        uid = self._write_event("report", doc)
        self._report_producer("report", doc)
        logger.info(f"Generated report. Tiled: {uid}\n Kafka: {doc.get('uid', 'No UID')}")
        self.close_and_restart(clear_tell_cache=False, retell_all=False, reason="Per-Run Subscribers")

    @classmethod
    def get_beamline_objects(cls) -> dict:
        ret = super().get_beamline_objects()
        beamline_tla = "pdf"
        kafka_config = nslsii.kafka_utils._read_bluesky_kafka_config_file(
            config_file_path="/etc/bluesky/kafka.yml"
        )
        ret["report_producer"] = Publisher(
            topic=f"{beamline_tla}.mmm.bluesky.agents",
            bootstrap_servers=",".join(kafka_config["bootstrap_servers"]),
            key="{beamline_tla}.key",
            producer_config=kafka_config["runengine_producer_config"],
        )
