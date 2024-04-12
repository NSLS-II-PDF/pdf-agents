"""Passive PeakFit Agent that Publishes Reports"""

import tiled.client.node  # noqa: F401
from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator

from pdf_agents.peakfit import PeakFitAgent

xrois = [(2.75, 2.95), (3.6, 3.8), (4.6, 5.0)]  # IMPORTANT: these bounds are for x = Q, not x = tth
fit_func = "gaussian"
pos_percent_lim = 2
maxcycles = 1000

report_producer = PeakFitAgent.get_default_producer()


agent = PeakFitAgent(
    # fit_roi Args
    xrois=xrois,
    fit_func=fit_func,
    pos_percent_lim=pos_percent_lim,
    maxcycles=maxcycles,
    # Required report producer
    report_producer=report_producer,
    # BS Adaptive Args
    ask_on_tell=False,
    report_on_tell=True,
)


@startup_decorator
def startup():
    agent.start()


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("Agent Name", agent, "instance_name")
