"""Passive GSAS Agent that Publishes Reports"""

import tiled.client.node  # noqa: F401
from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator

from pdf_agents.gsas import RefinementAgent

param_dict_phase1 = {
    "set": {
        "Background": {"type": "chebyschev-1", "no. coeffs": 6, "refine": True},
        "Limits": [1, 12],
        "Scale": True,
        "Cell": True,
        "Size": {
            "type": "isotropic",
            "refine": True,
            "value": 0.2,
        },  # crystallite size in microns
        "Mustrain": {"type": "isotropic", "refine": True, "value": 0.001},  # RMS strain
    }
}

agent = RefinementAgent(
    # GSAS Args
    cif_paths=["/src/pdf-agents/assets/fcc.cif"],
    refinement_params=[param_dict_phase1],
    inst_param_path="/src/pdf-agents/assets/LaB6_IPF.instprm",
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
