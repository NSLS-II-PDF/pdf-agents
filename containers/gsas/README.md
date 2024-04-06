# GSAS Image

Fedora based image that contains gsas, python3, bluesky, and bluesky-adaptive.
Primarily for gsas agent.

We had to use the conda install because the pure Fedora and Ubuntu image ran into issues with the GSASII install from pip and package managers.


```bash
cd /path/to/pdf-agents
curl https://subversion.xray.aps.anl.gov/admin_pyGSAS/downloads/gsas2full-Latest-Linux-x86_64.sh > ./containers/gsas/gsas2full-Latest-Linux-x86_64.sh
podman build --platform linux/amd64 -t gsas:conda -f containers/gsas/Containerfile-conda .
podman run -it --rm gsas:conda conda run -n GSASII --no-capture-output ipython
```
