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
# Zero-burden build: only a C++17 compiler + CMake/Ninja — no LLVM/Clang dev libs,
# no OpenMP. The engine runs kernels through its from-scratch four-tier CPU backend
# (PTX interpreter / compiled-fiber / native x86-64 JIT / SSA), so the image needs
# no LLVM at build or run time (which also fixes the Ubuntu llvm-dev LLVMExports
# CMake breakage and shrinks the image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates clang-18 \
        libsqlite3-dev liblapack-dev libkeyutils-dev libssl-dev zlib1g-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Build the runtime + worker (Release, LLVM-free, no tests/dashboard in the image).
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/vgre \
        -DVGRE_ENABLE_JIT=OFF -DVGRE_ENABLE_OPENMP=OFF \
        -DVGRE_BUILD_TESTS=OFF -DVGRE_BUILD_DASHBOARD=OFF \
        -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
    && cmake --build build --parallel \
    && cmake --install build

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# Runtime-only deps: BLAS/LAPACK + SQLite for the optional feature paths. No LLVM,
# no Clang, no OpenMP — the LLVM-free engine forks nothing at launch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        liblapack3 libsqlite3-0 libkeyutils1 zlib1g ca-certificates \
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
      org.opencontainers.image.source="https://github.com/etienne0114/vgre"

USER vgre
WORKDIR /home/vgre

# Liveness: the worker prints build info and exits 0 — proves the libraries load.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["vgre-worker", "--version"]

ENTRYPOINT ["vgre-worker"]
CMD ["--help"]
