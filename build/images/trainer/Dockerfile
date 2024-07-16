ARG BASE_IMAGE=ubuntu

FROM golang:1.21.1 as builder

ARG GOPROXY
ARG GOTAGS
ARG GOGCFLAGS

WORKDIR /go/src/d7y.io/dragonfly/v2

RUN apt-get update \
    && apt-get install -y bash make gcc libc-dev git

RUN apt-get install -y curl \
    && curl -o libtensorflow.tar.gz https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz \
    && tar -C /usr/local -xzf libtensorflow.tar.gz  \ 
    && rm libtensorflow.tar.gz

COPY . /go/src/d7y.io/dragonfly/v2

RUN make build-trainer && make install-trainer

FROM ${BASE_IMAGE} as health

ENV GRPC_HEALTH_PROBE_VERSION v0.4.24

RUN apt-get update \
    && apt-get install -y wget

RUN if [ "$(uname -m)" = "ppc64le" ]; then \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-ppc64le; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-arm64; \
    else \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64; \
    fi && \
    chmod +x /bin/grpc_health_probe

FROM ${BASE_IMAGE}

ENV PATH=/opt/dragonfly/bin:$PATH
ENV TF_ENABLE_ONEDNN_OPTS=0
RUN echo "hosts: files dns" > /etc/nsswitch.conf

COPY --from=builder /opt/dragonfly/bin/trainer /opt/dragonfly/bin/trainer
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=health /bin/grpc_health_probe /bin/grpc_health_probe
COPY --from=health /var/lib/apt/lists /var/lib/apt/lists
COPY ../trainer/training/models /var/lib/dragonfly/models

RUN ldconfig /usr/local/lib

RUN apt-get install -y vim bash-completion \
    && rm /bin/sh \
    && ln -s /bin/bash /bin/sh \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'PS1="\[\e[1;32m\]\u@\h \[\e[1;34m\]\w\[\e[0m\]\$ "' >> ~/.bashrc \
    && . ~/.bashrc

EXPOSE 9090

ENTRYPOINT ["/opt/dragonfly/bin/trainer"]
