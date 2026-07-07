# Fable SYCL build (Intel Arc / Battlemage, e.g. Arc Pro B70).
#
# Mirrors the build recipe used for the llama-sycl-fable production images
# (see FABLE-CHANGES.md in fable-bench):
#   - monolithic build (no GGML_BACKEND_DL) with icx/icpx
#   - -O3, SYCL F16, oneDNN enabled (GGML_SYCL_DNN=ON) for the XMX
#     flash-attention prefill path
#   - JIT (no GGML_SYCL_DEVICE_ARCH AOT) so the
#     -ze-intel-greater-than-4GB-buffer-required link flag stays active,
#     which multi-GB KV buffers at 240k context need
#
# Targets:
#   build    - compile stage
#   artifact - binaries + shared libs only (for `docker buildx --output type=local`)
#   server   - runtime image with llama-server (default)
#   full     - runtime image with all tools + python conversion scripts

ARG ONEAPI_VERSION=2025.3.3-0-devel-ubuntu24.04
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

## Web UI build

ARG NODE_VERSION=24

FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

## Build stage

FROM docker.io/intel/deep-learning-essentials:$ONEAPI_VERSION AS build

ARG LEVEL_ZERO_VERSION=1.28.2
ARG LEVEL_ZERO_UBUNTU_VERSION=u24.04
RUN apt-get update && \
    apt-get install -y git libssl-dev wget ca-certificates ninja-build build-essential && \
    cd /tmp && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero.deb && \
    wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero-devel_${LEVEL_ZERO_VERSION}%2B${LEVEL_ZERO_UBUNTU_VERSION}_amd64.deb" -O level-zero-devel.deb && \
    apt-get -o Dpkg::Options::="--force-overwrite" install -y ./level-zero.deb ./level-zero-devel.deb && \
    rm -f /tmp/level-zero.deb /tmp/level-zero-devel.deb

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG' \
        -DGGML_NATIVE=OFF \
        -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON \
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
        -DLLAMA_BUILD_TESTS=OFF && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Artifact export (binaries + shared libs, no runtime image)

FROM scratch AS artifact

COPY --from=build /app/full/llama* /bin/
COPY --from=build /app/lib/ /lib/

## Runtime base

FROM docker.io/intel/deep-learning-essentials:$ONEAPI_VERSION AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/fnrcum/llama.cpp
ARG IMAGE_SOURCE=https://github.com/fnrcum/llama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="llama.cpp fable-sycl" \
      org.opencontainers.image.description="llama.cpp with fable SYCL optimizations (Ornith/Gemma fattn, turboquant/rotorquant KV)" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

ARG IGC_VERSION=v2.34.4
ARG IGC_VERSION_FULL=2_2.34.4+21428
ARG COMPUTE_RUNTIME_VERSION=26.18.38308.1
ARG COMPUTE_RUNTIME_VERSION_FULL=26.18.38308.1-0
ARG IGDGMM_VERSION=22.10.0
RUN mkdir /tmp/neo/ && cd /tmp/neo/ \
  && wget https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/intel-graphics-compiler/releases/download/$IGC_VERSION/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-ocloc-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-opencl-icd-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libze-intel-gpu1-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
  && wget https://github.com/intel/compute-runtime/releases/download/$COMPUTE_RUNTIME_VERSION/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
  && dpkg --install *.deb

RUN apt-get update \
    && apt-get install -y libgomp1 curl \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

# IMPORTANT: append, never overwrite - icx runtime libs (libsvml etc.) live under
# /opt/intel/oneapi and are found via the inherited LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH}

## Full

FROM base AS full

COPY --from=build /app/lib/ /app
COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        python3-pip \
        python3-venv && \
    python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["/app/tools.sh"]

## Server (default production target)

FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
