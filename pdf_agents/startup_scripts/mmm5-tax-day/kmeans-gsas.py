"""Kmeans Agent that consumes GSAS Refinement Agent Output"""

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
        return "GSAS-Based-Active-Kmeans"

    def trigger_condition(self, uid) -> bool:
        return self.exp_catalog[uid].metadata["start"]["agent_name"].startswith("GSAS-Refinement-Agent")

    def unpack_run(self, run):
        data = run.report.data
        x = data["raw_independent_variable"].read().flatten()
        y = np.concatenate(
            [
                data[key].read().flatten()
                for key in [
                    "gsas_as",
                    "gsas_bs",
                    "gsas_cs",
                    "gsas_alphas",
                    "gsas_betas",
                    "gsas_gammas",
                    "gsas_volumes",
                    "gsas_rwps",
                ]
            ]
        )
        return x, y


agent = Agent(
    # K means Args
    bounds=np.array([(-30, 30), (-30, 30)]),
    k_clusters=4,
    # PDF Args
    motor_names=["xstage", "ystage"],
    motor_origins=[-128.85, 49.91],
    # BS Adaptive Args
    kafka_consumer=kafka_consumer,
    ask_on_tell=False,
    report_on_tell=False,
)


@startup_decorator
def startup():
    agent.start()
    path = "/src/pdf-agents/pdf_agents/startup_scripts/mmm5-tax-day/fri-gsas-out-uids.txt"
    with open(path, "r") as f:
        uids = []
        for line in f:
            uid = line.strip().strip(",").strip("'")
            if agent.trigger_condition(uid):
                uids.append(uid)
    agent.tell_agent_by_uid(uids)


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("Tell Cache", agent, "tell_cache")
register_variable("Agent Name", agent, "instance_name")
