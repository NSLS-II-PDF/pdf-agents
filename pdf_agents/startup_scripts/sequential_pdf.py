from bluesky_adaptive.server import shutdown_decorator, startup_decorator

from pdf_agents.agents import PDFSequentialAgent

agent = PDFSequentialAgent([0.0, 0.1, 0.3], ask_on_tell=False, report_on_tell=True, direct_to_queue=False)


@startup_decorator
def startup():
    agent.start()


@shutdown_decorator
def shutdown_agent():
    return agent.stop()
