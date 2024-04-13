import time as ttime

import numpy as np
from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator
from bmm_agents.base import BMMBaseAgent

from pdf_agents.monarch_bmm_subject import KMeansMonarchSubject

bmm_objects = BMMBaseAgent.get_beamline_objects()


class TimedAgent(KMeansMonarchSubject):
    def __init__(self, *args, **kwargs):
        self.last_time = ttime.time()
        super().__init__(self, *args, **kwargs)

    def subject_ask_condition(self):
        if ttime.time() - self.last_time > 60 * 90:  # 1.5 hours
            self.last_time = ttime.time()
            return True
        else:
            return False

    @property
    def name(self):
        return "TimedKMeansPDFMonarchBMMSubject"


agent = TimedAgent(
    filename="Pt-Zr-Multimodal",
    exp_mode="fluorescence",
    exp_data_type="mu",
    elements=["Pt", "Zr"],
    edges=["L3", "K"],
    element_origins=[[152.341, 119.534], [152.568, 119.632]],
    element_det_positions=[170, 120],
    sample="PtZr wafer 2",
    preparation="PtZr codeposited on a silica wafer",
    exp_bounds="-200 -30 -10 25 13k",
    exp_steps="10 2 0.5 0.05k",
    exp_times="1 1 1 1",
    subject_qserver=bmm_objects["qserver"],
    subject_kafka_producer=bmm_objects["kafka_producer"],
    subject_endstation_key="bmm",
    pdf_origin=(17.574, 4.075),
    bounds=(-29, 29),
    ask_on_tell=False,
    report_on_tell=True,
    k_clusters=6,
)


@startup_decorator
def startup():
    agent.start()
    path = "/nsls2/data/pdf/shared/config/source/pdf-agents/pdf_agents/startup_scripts/historical_uids.txt"
    with open(path, "r") as f:
        uids = []
        for line in f:
            uids.append(line.strip().strip(",").strip("'"))

    agent.tell_agent_by_uid(np.random.choice(uids, 15, replace=False))
    agent.ask_on_tell = True
    agent.add_suggestions_to_queue(3)
    agent.add_suggestions_to_subject_queue(1)


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("tell cache", agent, "tell_cache")
register_variable("agent name", agent, "instance_name")
