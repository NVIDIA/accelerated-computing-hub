FROM rapidsai/rapidsai:21.06-cuda11.0-runtime-ubuntu18.04-py3.8

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

COPY . ${HOME}
COPY binder/noop.sh /opt/docker/bin/entrypoint_source
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER}
WORKDIR ${HOME}