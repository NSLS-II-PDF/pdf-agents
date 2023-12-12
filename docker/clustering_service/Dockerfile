# docker build -t kmeans_service:latest .
FROM  python:3.10

RUN pip3 install bluesky-adaptive
RUN pip3 install uvicorn fastapi caproto nslsii
RUN pip3 install git+https://github.com/NSLS-II-PDF/pdf-agents.git@enh-container-service
RUN pip3 install git+https://github.com/bluesky/databroker.git@v2.0.0b13#egg=databroker
RUN pip3 install scikit-learn

COPY kmeans_service.py /src/kmeans_service.py

ENV BS_AGENT_STARTUP_SCRIPT_PATH=/src/kmeans_service.py
ENV OFFLINE_MODE="TRUE"

CMD uvicorn bluesky_adaptive.server:app --host 0.0.0.0 --port 60610
