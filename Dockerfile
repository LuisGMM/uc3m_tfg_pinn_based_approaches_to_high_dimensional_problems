
FROM python:3.11


ENV PYTHONPATH="${PYTHONPATH}:/tfg"

COPY ./requirements /requirements
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade -r /requirements/dev.txt \
    && apt-get clean \
    && rm -rf /requirements \
    && rm -rf /root/.cache/pip/* \
    && :

# RUN mkdir /tfg
# COPY ./src/tfg /tfg
WORKDIR /tfg

