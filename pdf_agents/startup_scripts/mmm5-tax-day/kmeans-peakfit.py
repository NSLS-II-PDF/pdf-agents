"""Kmeans Agent that consumes Peak Fit Agent Output"""

import uuid

import nslsii
import numpy as np
import tiled.client.node  # noqa: F401
from bluesky_adaptive.agents.base import AgentConsumer
from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator

from pdf_agents.sklearn import ActiveKmeansAgent

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


class Agent(ActiveKmeansAgent):
    @property
    def name(self):
        return "PeakFit-Based-Active-Kmeans"

    def trigger_condition(self, uid) -> bool:
        return self.exp_catalog.metadata["start"]["agent_name"].startswith("Peak-Fit-Agent")


agent = Agent(
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
