import time as ttime
import uuid

import nslsii
import numpy as np
import tiled.client.node  # noqa: F401
from bluesky_adaptive.agents.base import AgentConsumer
from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator
from bmm_agents.base import BMMBaseAgent

from pdf_agents.monarch_bmm_subject import KMeansMonarchSubject

bmm_objects = BMMBaseAgent.get_beamline_objects()


# Custom Kafka Consumer Needed since we are subscribing downstream from GSAS
beamline_tla = "pdf"
kafka_config = nslsii.kafka_utils._read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")
kafka_consumer = AgentConsumer(
    topics=[
        f"{beamline_tla}.mmm.bluesky.agents",
    ],
    consumer_config=kafka_config["runengine_producer_config"],
    bootstrap_servers=",".join(kafka_config["bootstrap_servers"]),
    group_id=f"echo-{beamline_tla}-{str(uuid.uuid4())[:8]}",
)


class Agent(KMeansMonarchSubject):
    def __init__(self, *args, **kwargs):
        self.last_time = ttime.time()
        self._auto_asking = False
        super().__init__(self, *args, **kwargs)

    @property
    def name(self):
        return "GSAS-Based-KMeans-PDF-Monarch-BMM-Subject"

    def trigger_condition(self, uid) -> bool:
        return self.exp_catalog.metadata["start"]["agent_name"].startswith("GSAS-Refinement-Agent")

    def subject_ask_condition(self):
        if not self._auto_asking:
            return False
        elif ttime.time() - self.last_time > 60 * 90:  # 1.5 hours
            self.last_time = ttime.time()
            return True

    @property
    def auto_asking(self):
        return self._auto_asking

    def enable_auto_subject_asking(self):
        self.last_time = ttime.time()
        self._auto_asking = True

    def disable_auto_subject_asking(self):
        self._auto_asking = False

    def server_registrations(self) -> None:
        self._register_method("enable_auto_subject_asking")
        self._register_method("disable_auto_subject_asking")
        return super().server_registrations()


agent = Agent(
    # Monarch-Subject args
    filename="Pt-Zr-Multimodal",
    exp_mode="fluorescence",
    exp_data_type="mu",
    elements=["Pt", "Ni"],
    edges=["L3", "K"],
    element_origins=[[186.307, 89.276], [186.384, 89.305]],
    element_det_positions=[185, 160],
    sample="AlPtNi wafer pretend-binary PtNi",
    preparation="AlPtNi codeposited on a silica wafer",
    exp_bounds="-200 -30 -10 25 13k",
    exp_steps="10 2 0.5 0.05k",
    exp_times="1 1 1 1",
    subject_qserver=bmm_objects["qserver"],
    subject_kafka_producer=bmm_objects["kafka_producer"],
    subject_endstation_key="bmm",
    pdf_origin=[-154.7682, 48.9615],  # This is apparently an unused value, redundant with motor_origins
    # K means Args
    bounds=np.array([(-32, 32), (-32, 32)]),
    k_clusters=4,
    # PDF Args
    motor_names=["xstage", "ystage"],
    motor_origins=[-154.7682, 48.9615],
    # BS Adaptive Args
    kafka_consumer=kafka_consumer,
    ask_on_tell=False,
    report_on_tell=False,
)


@startup_decorator
def startup():
    agent.start()


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("Tell Cache", agent, "tell_cache")
register_variable("Agent Name", agent, "instance_name")
