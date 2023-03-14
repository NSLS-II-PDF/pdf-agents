import uuid
from abc import ABC
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


class PDFBaseAgent(Agent, ABC):
    def __init__(
        self,
        *args,
        motor_name: str = "Grid_X",
        motor_resolution: float = 0.0002,
        exposure: float = 30.0,
        data_key: str = "chi_I",
        roi_key: str = "chi_Q",
        roi: Optional[Tuple] = None,
        **kwargs,
    ):
        self._redis = redis.Redis(host="info.pdf.nsls2.bnl.gov")
        self._motor_name = motor_name
        self._motor_resolution = motor_resolution
        self._exposure = exposure
        self._data_key = data_key
        self._roi_key = roi_key
        self._roi = roi
        super().__init__(*args, **kwargs)

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
        # TODO: get sample number from redis
        return "agent_sample_count", [self.motor_name, point, self.exposure_time], dict(sample_number=0)

    def unpack_run(self, run) -> Tuple[Union[float, ArrayLike], Union[float, ArrayLike]]:
        y = run.primary.data[self.data_key].read().flatten()
        if self.roi is not None:
            q = np.array(run.primary.data[self.data_key]).flatten()
            idx_min = np.where(q < self.roi[0])[0][-1] if len(np.where(q < self.roi[0])[0]) else None
            idx_max = np.where(q > self.roi[1])[0][-1] if len(np.where(q > self.roi[1])[0]) else None
            y = y[idx_min:idx_max]
        return run.start[self.motor_name][self.motor_name]["value"], y

    def server_registrations(self) -> None:
        self._register_property("motor_resolution")
        self._register_property("motor_name")
        self._register_property("exposure_time")
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
        return self._exposure

    @exposure_time.setter
    def exposure_time(self, value: float):
        self._exposure = value

    @property
    def data_key(self):
        return self._data_key

    @data_key.setter
    def data_key(self, value: str):
        self._data_key = value
        self.close_and_restart(clear_tell_cache=True)
        # TODO: Ensure a clear caches method is built in here for subclass

    @property
    def roi_key(self):
        return self._roi_key

    @roi_key.setter
    def roi_key(self, value: str):
        self._roi_key = value
        self.close_and_restart(clear_tell_cache=True)
        # TODO: Ensure a clear caches method is built in here for subclass

    @property
    def roi(self):
        return self._roi_key

    @roi.setter
    def roi(self, value: Tuple[float, float]):
        self._roi = value
        self.close_and_restart(clear_tell_cache=True)
        # TODO: Ensure a clear caches method is built in here for subclass

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
            key="cms.key",
            producer_config=kafka_config["runengine_producer_config"],
        )

        return dict(
            kafka_consumer=kafka_consumer,
            kafka_producer=kafka_producer,
            tiled_data_node=tiled.client.from_profile(f"{beamline_tla}_bluesky_sandbox"),
            tiled_agent_node=tiled.client.from_profile(f"{beamline_tla}_bluesky_sandbox"),
            qserver=qs,
        )


class PDFSequentialAgent(PDFBaseAgent, SequentialAgentBase):
    def __init__(
        self,
        *,
        sequence: Sequence[Union[float, ArrayLike]],
        relative_bounds: Tuple[Union[float, ArrayLike]] = None,
        **kwargs,
    ) -> None:
        _default_kwargs = self.get_beamline_objects()
        _default_kwargs.update(kwargs)
        super().__init__(sequence=sequence, relative_bounds=relative_bounds, **_default_kwargs)
