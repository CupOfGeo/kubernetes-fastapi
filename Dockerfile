#FROM python:3.7
# FROM python:3.9.4-slim-buster
# FROM gcr.io/deeplearning-platform-release/base-cu113:latest

# dosent work no pip no python
FROM nvidia/cuda:11.0-base

COPY requirements.txt /usr/src/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir -r /usr/src/requirements.txt

EXPOSE 8080

COPY ./service /service


# Use the ping endpoint as a healthcheck,
# so Docker knows if the API is still running ok or needs to be restarted
HEALTHCHECK --interval=21s --timeout=3s --start-period=10s CMD curl --fail http://localhost:8080/ping || exit 1

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
