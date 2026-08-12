FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/swebench

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/swebench
COPY . /opt/swebench
RUN python -m pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "swebench.harness.run_evaluation"]
