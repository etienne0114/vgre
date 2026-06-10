# VGRE — Virtual GPU Runtime Engine: production runtime image (Track 19).
#
# Multi-stage: a builder compiles the runtime + vgre-worker from source against
# LLVM-18, and a slim runtime stage carries only the shared libraries, the
# worker binary, and the runtime dependencies. Multi-arch (amd64/arm64) is built
# by the release workflow via buildx; this Dockerfile is arch-agnostic.
#
#   docker build -t vgre:dev .
#   docker run --rm vgre:dev --version        # prints build-info JSON
#   docker run --rm -e VGRE_METRICS_PORT=9090 -p 9090:9090 vgre:dev
#
# Metrics/health (when VGRE_METRICS_PORT is set): GET /metrics /healthz /readyz.

# ── Builder ──────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates \
        clang llvm-dev libclang-dev libomp-dev \
        libsqlite3-dev liblapack-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Build the runtime + worker (Release, no tests/dashboard in the image build).
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/vgre \
        -DVGRE_BUILD_TESTS=OFF \
        -DVGRE_BUILD_DASHBOARD=OFF \
    && cmake --build build --parallel \
    && cmake --install build

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# Runtime-only deps: clang/llvm runtime for the JIT (the engine forks clang++ at
# launch), OpenMP, BLAS/LAPACK, SQLite.
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang llvm libomp5 liblapack3 libsqlite3-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 vgre

COPY --from=builder /opt/vgre/ /opt/vgre/
ENV LD_LIBRARY_PATH=/opt/vgre/lib \
    PATH=/opt/vgre/bin:$PATH \
    VGRE_IPC_MODE=OFF

# OCI labels (version/revision injected by the release workflow build args).
ARG VGRE_VERSION=0.0.0
ARG VGRE_REVISION=unknown
LABEL org.opencontainers.image.title="vgre" \
      org.opencontainers.image.description="Virtual GPU Runtime Engine — CUDA-on-CPU" \
      org.opencontainers.image.version="${VGRE_VERSION}" \
      org.opencontainers.image.revision="${VGRE_REVISION}" \
      org.opencontainers.image.source="https://github.com/etienne0114/virtual-gpu-runtime"

USER vgre
WORKDIR /home/vgre

# Liveness: the worker prints build info and exits 0 — proves the libraries load.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["vgre-worker", "--version"]

ENTRYPOINT ["vgre-worker"]
CMD ["--help"]
