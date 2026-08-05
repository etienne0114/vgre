# Hugging Face Spaces (Docker SDK) runtime image for the VGRE inference server.
#
# HF Spaces are free and need no payment card. To keep the Space build fast and
# reliable, the LLVM-free libvgre_nn.so is compiled in GitHub CI (the verified
# toolchain) and uploaded prebuilt alongside this file — so the Space just runs
# Python, no compilation. Assembled + pushed by .github/workflows/deploy-hf.yml.
#
# Expected files next to this Dockerfile in the Space repo:
#   lib/libvgre_nn.so, vgre/ (the Python package), vgre_server.py, README.md

FROM python:3.12-slim
RUN pip install --no-cache-dir numpy
WORKDIR /app
COPY lib/libvgre_nn.so /app/lib/libvgre_nn.so
COPY vgre /app/bindings/python/vgre
COPY vgre_server.py /app/server/vgre_server.py
ENV VGRE_LIB_PATH=/app/lib/libvgre_nn.so \
    PYTHONPATH=/app/bindings/python \
    VGRE_HOST=0.0.0.0 \
    VGRE_PORT=7860 \
    VGRE_DEMO_STEPS=400
EXPOSE 7860
CMD ["python", "/app/server/vgre_server.py"]
