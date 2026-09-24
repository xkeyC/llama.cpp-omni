# llama-omni-server (MiniCPM-o 4.5) with CUDA, built from this checkout.
#
#   cd tools/omni/docker && docker compose up -d --build
#
# Models are not in the image: they live in a volume (see download-models.sh).

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# GPU architectures to build for: 75 Turing, 80 A100, 86 RTX 30xx, 89 RTX 40xx,
# 90 H100, 120 RTX 50xx. Fewer builds faster, e.g. CUDA_DOCKER_ARCH=86 for a 3060.
ARG CUDA_DOCKER_ARCH="75;80;86;89;90;120"

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc-14 g++-14 build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /src
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release --target llama-omni-server -j"$(nproc)"
RUN set -eux; \
    mkdir -p /out/bin /out/lib; \
    cp build/bin/llama-omni-server /out/bin/; \
    find build -name '*.so*' -exec cp -P {} /out/lib/ \;

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Voice clone (tools/omni/voice/make_voice_bundle.py): CPU-only ONNX models.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir numpy onnxruntime librosa soundfile kaldi-native-fbank
ENV PATH=/opt/venv/bin:${PATH}

WORKDIR /app
COPY --from=build /out/bin/ /app/
COPY --from=build /out/lib/ /app/lib/
# The server reads its default reference audio relative to the working dir.
COPY tools/omni/assets /app/tools/omni/assets
COPY tools/omni/voice /app/tools/omni/voice
COPY tools/omni/docker/entrypoint.sh tools/omni/docker/download-models.sh /app/
ENV LD_LIBRARY_PATH=/app/lib
RUN set -eux; \
    sed -i 's/\r$//' /app/entrypoint.sh /app/download-models.sh; \
    chmod +x /app/entrypoint.sh /app/download-models.sh; \
    ldd /app/llama-omni-server | tee /tmp/ldd.txt; \
    if grep 'not found' /tmp/ldd.txt | grep -v 'libcuda.so'; then exit 1; fi

# Defaults measured for a 12 GB card; see tools/omni/voice/README.md.
ENV OMNI_MODELS=/models \
    OMNI_CTX=8192 \
    OMNI_PORT=19060 \
    OMNI_TTS_N_CTX=4096 \
    OMNI_ROUTER_CTX=2048 \
    OMNI_VOICE_CACHE_DIR=/cache/voices

EXPOSE 19060
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs "http://127.0.0.1:${OMNI_PORT}/health" || exit 1
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
