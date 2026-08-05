---
title: VGRE Inference
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# VGRE inference

Text generation on a **CPU, no GPU**, served by VGRE's own from-scratch runtime.

- `GET /health` — liveness
- `GET /info` — model + backend info (says whether it's the demo model)
- `POST /generate` — `{"prompt": "...", "max_tokens": 64, "temperature": 0.8}`
- `/` — a tiny web playground

Out of the box this serves a small byte-level **demo model** trained at startup, so
it runs anywhere. For real quality, point it at a trained checkpoint via
`VGRE_MODEL_PATH` + `VGRE_MODEL_CFG`. Source: https://github.com/etienne0114/vgre
