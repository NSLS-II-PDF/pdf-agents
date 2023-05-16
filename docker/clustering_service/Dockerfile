# docker build -t kmeans_service:latest .
FROM  fedora
# Lock in a python version because of some backward compatability issues
RUN dnf update -y \
    && dnf install -y \
    python3.9 \
    g++ \
    gcc \
    git \
    && dnf clean all

RUN python3.9 -m ensurepip --upgrade
RUN pip3 install 'bluesky-adaptive[all]'
RUN pip3 install uvicorn fastapi caproto nslsii
RUN pip3 install git+https://github.com/NSLS-II-PDF/pdf-agents.git@main
RUN pip3 install git+https://github.com/bluesky/databroker.git@v2.0.0b13#egg=databroker

COPY kmeans_service.py /src/kmeans_service.py

ENV BS_AGENT_STARTUP_SCRIPT_PATH=/src/kmeans_service.py

CMD uvicorn bluesky_adaptive.server:app --host 127.0.0.1 --port 60610