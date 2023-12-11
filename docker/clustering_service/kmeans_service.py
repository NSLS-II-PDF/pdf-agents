import os

from bluesky_adaptive.server import register_variable, shutdown_decorator, startup_decorator

from pdf_agents.sklearn import PassiveKmeansAgent


class Agent(PassiveKmeansAgent):
    """Copy of Kmeans agent for PDF Beamline that exposes and on/off switch to stop listening, and
    a full reset method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = False

    @property
    def name(self):
        return "KmeansAgentService"

    @property
    def running(self):
        return self._running

    def activate(self):
        self._running = True

    def pause(self):
        self._running = False

    def exp_start_clean_run(self):
        """Convenience method to start a fresh agent that is accumulating documents live."""
        self.pause()
        self.close_and_restart()
        self.activate()

    def exp_report_from_historical_data(self, uids):
        """Convenience method to spawn a fresh agent that writes one report."""
        self.pause()
        self.disable_continuous_reporting()
        self.close_and_restart()
        self.tell_agent_by_uid(uids)
        self.report()

    def exp_start_run_with_data(self, uids):
        """Convenience method to spawn a fresh agent with data that accumualtes documents live."""
        self.pause()
        self.disable_continuous_reporting()
        self.close_and_restart()
        self.tell_agent_by_uid(uids)
        self.enable_continuous_reporting()
        self.activate()

    def trigger_condition(self, uid) -> bool:
        return self.running

    def server_registrations(self) -> None:
        self._register_method("close_and_restart")
        self._register_property("running")
        self._register_method("activate")
        self._register_method("pause")
        self._register_method("exp_start_clean_run")
        self._register_method("exp_report_from_historical_data")
        self._register_method("exp_start_run_with_data")
        return super().server_registrations()


offline_mode = str(os.getenv("OFFLINE_MODE", "False")).lower() in ["true", "1", "yes"]
print(offline_mode)
agent = Agent(k_clusters=3, report_on_tell=True, ask_on_tell=False, direct_to_queue=False, offline=offline_mode)


@startup_decorator
def startup():
    agent.start()


@shutdown_decorator
def shutdown_agent():
    return agent.stop()


register_variable("UID Cache", agent, "tell_cache")
register_variable("Agent Name", agent, "instance_name")
