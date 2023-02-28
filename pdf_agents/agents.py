import uuid
from typing import Dict, List, Sequence, Tuple, Union

import nslsii.kafka_utils
import numpy as np
import tiled
from bluesky_adaptive.agents.base import AgentConsumer
from bluesky_adaptive.agents.simple import SequentialAgentBase
from bluesky_kafka import Publisher
from bluesky_queueserver_api.zmq import REManagerAPI
from numpy.typing import ArrayLike


class PDFBaseAgent:
    def __init__(self, motor="Grid_X", exposure=30):
        self._motor = motor
        self._exposure = exposure

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
        return "agent_sample_count", [self._motor, point, self._exposure], dict(sample_number=0)

    def unpack_run(self, run) -> Tuple[Union[float, ArrayLike], Union[float, ArrayLike]]:
        y = np.array(run.primary.data["chi_I"][0])
        return run.start["Grid_X"]["Grid_X"]["value"], y

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
