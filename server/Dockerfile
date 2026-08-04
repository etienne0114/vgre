# VGRE inference server image — multi-stage.
#
# The builder compiles the lightweight, LLVM-free libvgre_nn (LLVM is only needed
# at *build* time for the project's configure; the produced .so has no LLVM
# dependency). The runtime stage is a slim Python image carrying only
# libvgre_nn.so + the pure-Python bindings + the stdlib HTTP server — no GPU, no
# web framework, no heavy runtime deps.
#
#   docker build -f server/Dockerfile -t vgre-server .
#   docker run --rm -p 8080:8080 vgre-server
#   curl -s localhost:8080/health

# ── Builder ──────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build clang-18 llvm-18-dev libclang-18-dev \
        libomp-18-dev libssl-dev zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# Configure the project (LLVM present) and build ONLY the LLVM-free ML library.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
        -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
    && cmake --build build --target vgre_nn --parallel \
    && test -f build/src/xla/libvgre_nn.so

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime
RUN pip install --no-cache-dir numpy
WORKDIR /app
# Only the artifacts the server needs — no LLVM, no build toolchain.
COPY --from=builder /src/build/src/xla/libvgre_nn.so /app/lib/libvgre_nn.so
COPY --from=builder /src/bindings/python/vgre /app/bindings/python/vgre
COPY --from=builder /src/server/vgre_server.py /app/server/vgre_server.py
ENV VGRE_LIB_PATH=/app/lib/libvgre_nn.so \
    PYTHONPATH=/app/bindings/python \
    VGRE_HOST=0.0.0.0 \
    VGRE_PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)" || exit 1
CMD ["python", "/app/server/vgre_server.py"]
