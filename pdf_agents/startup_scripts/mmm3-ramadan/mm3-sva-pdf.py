"""SVA agent, that finds regions fo scientific value, and tiggers PDF to supplement XRD."""

from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator

from pdf_agents.scientific_value import ScientificValueAgentBase


class PDFSVA(ScientificValueAgentBase):
    @property
    def name(self):
        return "ScientificValuePDF"

    def measurement_plan(self, point):
        return "agent_redisAware_PDFcount", [point], {}


agent = PDFSVA(bounds=[-12.0, 46.0], ask_on_tell=True, report_on_tell=True)


@startup_decorator
def startup():
    agent.start()
    path = "/nsls2/data/pdf/shared/config/source/pdf-agents/pdf_agents/startup_scripts/historical_uids.txt"
    with open(path, "r") as f:
        uids = []
        for line in f:
            uids.append(line.strip().strip(",").strip("'"))

    agent.tell_agent_by_uid(uids)


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("tell cache", agent, "tell_cache")
register_variable("agent name", agent, "instance_name")
