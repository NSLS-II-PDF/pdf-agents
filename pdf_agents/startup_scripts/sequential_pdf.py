from bluesky_adaptive.server import shutdown_decorator, startup_decorator

from pdf_agents.agents import PDFSequentialAgent

agent = PDFSequentialAgent(
    sequence=[40.0, 45.0, 48.0], ask_on_tell=True, report_on_tell=True, direct_to_queue=True
)


@startup_decorator
def startup():
    agent.start()


@shutdown_decorator
def shutdown_agent():
    return agent.stop()
