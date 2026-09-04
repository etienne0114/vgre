---
title: VGRE Inference
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# VGRE inference

Text generation on a **CPU, no GPU**, served by VGRE's own from-scratch runtime.

Out of the box this serves a small byte-level **demo model** trained at startup,
so it runs anywhere on the free CPU tier. For real quality, point the engine at a
trained checkpoint. Source: https://github.com/etienne0114/vgre
