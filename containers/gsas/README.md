# GSAS Image

Fedora based image that contains gsas, python3, bluesky, and bluesky-adaptive.
Primarily for gsas agent.

We had to use the conda install because the pure Fedora and Ubuntu image ran into issues with the GSASII install from pip and package managers.


```bash
cd /path/to/pdf-agents
curl https://subversion.xray.aps.anl.gov/admin_pyGSAS/downloads/gsas2full-Latest-Linux-x86_64.sh > ./containers/gsas/gsas2full-Latest-Linux-x86_64.sh
podman build --platform linux/amd64 -t gsas:conda -f containers/gsas/Containerfile-conda .
podman run -v ./pdf_agents:/src/pdf-agents/pdf_agents:ro -it --rm gsas:conda conda run -n GSASII  --no-capture-output ipython
```

## Poking around with the agent
```python
import tiled.client.node # Workaround for API issue
from pdf_agents.gsas import RefinementAgent
offline_obj = RefinementAgent.get_offline_objects()
agent = RefinementAgent(cif_paths = [], refinement_params = [], inst_param_path = "", report_producer=offline_obj["kafka_producer"], offline=True)
```
