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
from bluesky_queueserver_api.zmq import REManagerAPI
from numpy.typing import ArrayLike

from .utils import OfflineKafka

logger = getLogger(__name__)


class PDFBaseAgent(Agent, ABC):
    def __init__(
        self,
        *args,
        motor_name: str = "Grid_X",
        motor_resolution: float = 0.0002,
        data_key: str = "chi_I",
        roi_key: str = "chi_Q",
        roi: Optional[Tuple] = None,
        offline=False,
        **kwargs,
    ):
        self._rkvs = redis.Redis(host="info.pdf.nsls2.bnl.gov")  # redis key value store
        self._motor_name = motor_name
        self._motor_resolution = motor_resolution
        self._data_key = data_key
        self._roi_key = roi_key
        self._roi = roi
        # Attributes pulled in from Redis
        self._exposure = float(self._rkvs.get("PDF:desired_exposure_time").decode("utf-8"))
        self._sample_number = int(self._rkvs.get("PDF:xpdacq:sample_number").decode("utf-8"))
        if offline:
            _default_kwargs = self.get_offline_objects()
        else:
            _default_kwargs = self.get_beamline_objects()
        _default_kwargs.update(kwargs)
        super().__init__(*args, **_default_kwargs)

    def measurement_plan(self, point: ArrayLike) -> Tuple[str, List, Dict]:
        """Default measurement plan is an agent modified simple count  of pe1c, for a 30 sec exposure.
        agent_sample_count(motor, position: float, exposure: float, *, sample_number: int, md=None):

        Parameters
        ----------
        point : ArrayLike
            Next point to measure using a given plan

        Returns
        -------
        plan_name : str
        plan_args : List
            List of arguments to pass to plan from a point to measure.
        plan_kwargs : dict
            Dictionary of keyword arguments to pass the plan, from a point to measure.
        """
        return (
            "agent_sample_count",
            [self.motor_name, point, self.exposure_time],
            dict(sample_number=self.sample_number),
        )

    def unpack_run(self, run) -> Tuple[Union[float, ArrayLike], Union[float, ArrayLike]]:
        y = run.primary.data[self.data_key].read().flatten()
        if self.roi is not None:
            ordinate = np.array(run.primary.data[self.roi_key]).flatten()
            idx_min = np.where(ordinate < self.roi[0])[0][-1] if len(np.where(ordinate < self.roi[0])[0]) else None
            idx_max = np.where(ordinate > self.roi[1])[0][-1] if len(np.where(ordinate > self.roi[1])[0]) else None
            y = y[idx_min:idx_max]
        return run.start[self.motor_name][self.motor_name]["value"], y

    def server_registrations(self) -> None:
        self._register_property("motor_resolution")
        self._register_property("motor_name")
        self._register_property("exposure_time")
        self._register_property("sample_number")
        self._register_property("data_key")
        self._register_property("roi_key")
        self._register_property("roi")
        return super().server_registrations()

    @property
    def motor_name(self):
        """Name of motor to be used as the independent variable in the experiment"""
        return self._motor_name

    @motor_name.setter
    def motor_name(self, value: str):
        self._motor_name = value

    @property
    def motor_resolution(self):
        """Minimum resolution for measurement in milimeters, i.e. (beam width)/2"""
        return self._motor_resolution

    @motor_resolution.setter
    def motor_resolution(self, value: float):
        self._motor_resolution = value

    @property
    def exposure_time(self):
        """Exposure time of scans in seconds"""
        value = float(self._rkvs.get("PDF:desired_exposure_time").decode("utf-8"))
        if value != self._exposure:
            logger.warning(
                f"Mismatch between agent exposure time ({self._exposure}) and redis value {value}. "
                "Updating to redis value."
            )
            self._exposure = value
        return self._exposure

    @exposure_time.setter
    def exposure_time(self, value: float):
        self._exposure = value
        self._rkvs.set("PDF:desired_exposure_time", value)

    @property
    def sample_number(self):
        """XPDAQ Sample Number"""
        value = int(self._rkvs.get("PDF:xpdacq:sample_number").decode("utf-8"))
        if value != self._sample_number:
            logger.warning(
                f"Mismatch between agent sample_number ({self._sample_number}) and redis value {value}. "
                "Updating to redis value."
            )
            self._sample_number = value
        return self._sample_number

    @sample_number.setter
    def sample_number(self, value: int):
        self._sample_number = value
        self._rkvs.set("PDF:xpdacq:sample_number", value)

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
        return self._roi_key

    @roi.setter
    def roi(self, value: Tuple[float, float]):
        self._roi = value
        self.close_and_restart(clear_tell_cache=True)

    @staticmethod
    def get_beamline_objects() -> dict:
        beamline_tla = "pdf"
        kafka_config = nslsii.kafka_utils._read_bluesky_kafka_config_file(
            config_file_path="/etc/bluesky/kafka.yml"
        )
        qs = REManagerAPI(http_server_uri=f"https://qserver.nsls2.bnl.gov/{beamline_tla}")
        qs.set_authorization_key(api_key=None)

        kafka_consumer = AgentConsumer(
            topics=[
                f"{beamline_tla}.bluesky.pdfstream.documents",
            ],
            consumer_config=kafka_config["runengine_producer_config"],
            bootstrap_servers=kafka_config["bootstrap_servers"],
            group_id=f"echo-{beamline_tla}-{str(uuid.uuid4())[:8]}",
        )

        kafka_producer = Publisher(
            topic=f"{beamline_tla}.bluesky.adjudicators",
            bootstrap_servers=kafka_config["bootstrap_servers"],
            key="{beamline_tla}.key",
            producer_config=kafka_config["runengine_producer_config"],
        )

        return dict(
            kafka_consumer=kafka_consumer,
            kafka_producer=kafka_producer,
            tiled_data_node=tiled.client.from_profile(f"{beamline_tla}_bluesky_sandbox"),
            tiled_agent_node=tiled.client.from_profile(f"{beamline_tla}_bluesky_sandbox"),
            qserver=qs,
        )

    @staticmethod
    def get_offline_objects() -> dict:
        """Objects to spin up agent with access to only tiled if available."""
        beamline_tla = "pdf"
        offline_kafka = OfflineKafka()
        try:
            node = tiled.client.from_profile(f"{beamline_tla}_bluesky_sandbox")
        except tiled.profile.ProfileNotFound:
            node = None

        return dict(
            kafka_consumer=offline_kafka,
            kafka_producer=offline_kafka,
            tiled_data_node=node,
            tiled_agent_node=node,
            qserver=None,
        )


class PDFSequentialAgent(PDFBaseAgent, SequentialAgentBase):
    def __init__(
        self,
        *,
        sequence: Sequence[Union[float, ArrayLike]],
        relative_bounds: Tuple[Union[float, ArrayLike]] = None,
        **kwargs,
    ) -> None:
        super().__init__(sequence=sequence, relative_bounds=relative_bounds, **kwargs)
