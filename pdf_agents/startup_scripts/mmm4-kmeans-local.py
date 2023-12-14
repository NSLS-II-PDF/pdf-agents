import numpy as np
import tiled.client.node
from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator
from bluesky_queueserver_api.zmq import REManagerAPI

from pdf_agents.sklearn import ActiveKmeansAgent

qserver = REManagerAPI(zmq_control_addr="tcp://xf28id1-srv1:60615", zmq_info_addr="tcp://xf28id1-srv1:60625")
agent = ActiveKmeansAgent(
    bounds=np.array([(-32, 32), (-32, 32)]),
    ask_on_tell=False,
    report_on_tell=True,
    k_clusters=4,
    motor_names=["xstage", "ystage"],
    motor_origins=[-154.7682, 48.9615],
    qserver=qserver,
)


@startup_decorator
def startup():
    agent.start()


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("Tell Cache", agent, "tell_cache")
register_variable("Agent Name", agent, "instance_name")
