"""SVA agent, that finds regions of scientific value, according to Peak Fit Agent Output"""

import uuid

import nslsii
import numpy as np
import tiled.client.node  # noqa: F401
from bluesky_adaptive.agents.base import AgentConsumer
from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator

from pdf_agents.scientific_value import ScientificValueAgentBase

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


class Agent(ScientificValueAgentBase):
    @property
    def name(self):
        return "Peakfit-Based-SVA"

    def trigger_condition(self, uid) -> bool:
        return self.exp_catalog[uid].metadata["start"]["agent_name"].startswith("Peak-Fit-Agent")

    def unpack_run(self, run):
        data = run.report.data
        x = data["raw_independent_variable"].read().flatten()
        y = np.concatenate(
            [
                data[key].read().flatten()
                for key in [
                    "peak_amplitudes",
                    "peak_positions",
                    "peak_fwhms",
                ]
            ]
        )
        return x, y


agent = Agent(
    # SVA Args
    bounds=np.array([(-30, 30), (-30, 30)]),
    # PDF Args
    motor_names=["xstage", "ystage"],
    motor_origins=[-128.85, 49.91],
    # BS Adaptive Args
    kafka_consumer=kafka_consumer,
    ask_on_tell=True,
    report_on_tell=True,
)


@startup_decorator
def startup():
    agent.start()


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("Tell Cache", agent, "tell_cache")
register_variable("Agent Name", agent, "instance_name")
