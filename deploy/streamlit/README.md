# VGRE demo on Streamlit Community Cloud (free, no card)

Hugging Face now requires a **PRO** subscription to host server-side (Gradio/
Docker) Spaces. **Streamlit Community Cloud** hosts this same demo for free — no
credit card, no subscription — straight from this GitHub repo.

## How it works

- The LLVM-free `libvgre_nn.so` is built in CI on ubuntu-22.04 (glibc 2.35, so
  it loads on Streamlit's Debian) and published as a **GitHub Release** asset
  (`nn-so-latest`) by [`.github/workflows/nn-so-release.yml`](../../.github/workflows/nn-so-release.yml).
- [`streamlit_app.py`](streamlit_app.py) downloads that `.so` at startup, points
  `VGRE_LIB_PATH` at it, imports the in-repo `vgre` package, trains a tiny
  byte-level demo model once (cached), and serves a text-generation UI.
- Everything runs on a plain CPU — no GPU anywhere.

## Deploy it (one-time, ~1 minute)

1. Make sure the release exists: the **Publish libvgre_nn.so** workflow must have
   run green once (it publishes `libvgre_nn.so` to the `nn-so-latest` release).
2. Go to <https://share.streamlit.io> and sign in with **GitHub**.
3. **Create app** → **Deploy a public app from GitHub**:
   - Repository: `etienne0114/vgre`
   - Branch: `main`
   - Main file path: `deploy/streamlit/streamlit_app.py`
4. **Deploy**. First load downloads the runtime and trains the demo model (a few
   seconds); after that it's cached. Your public URL will look like
   `https://<something>.streamlit.app`.

No secrets or environment variables are required.
