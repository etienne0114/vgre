"""VGRE inference — free **Streamlit Community Cloud** app (CPU, no GPU).

HF now requires a PRO subscription to host server-side (Gradio/Docker) Spaces;
Streamlit Community Cloud is free with no card and no subscription. It clones
this repo and runs this file, but it can't build the native runtime (that needs
LLVM 18 to configure). So the LLVM-free libvgre_nn.so is built in CI
(.github/workflows/nn-so-release.yml) and published as a GitHub Release asset;
we download it here at startup and drive VGRE's own from-scratch language model
through it. Nothing here needs a GPU.

Deploy: point Streamlit Community Cloud at this file
(deploy/streamlit/streamlit_app.py) on the main branch.
"""
import os
import sys
import urllib.request
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # deploy/streamlit -> repo root
# Rolling release asset built on ubuntu-22.04 (glibc 2.35), loads on Streamlit's
# newer Debian glibc. Republished on every push to main that touches the native
# sources, so this always tracks the latest runtime.
_SO_URL = "https://github.com/etienne0114/vgre/releases/download/nn-so-latest/libvgre_nn.so"

_CORPUS = (
    "VGRE runs and trains machine learning models on ordinary CPUs, with no GPU. "
    "It is built from scratch and stays lightweight, so it works everywhere. "
    "You can run it on a laptop, a server, or a cluster of ordinary computers. "
    "The goal is simple: make AI available to everyone, not only to those who own "
    "expensive graphics cards. "
) * 12

st.set_page_config(page_title="VGRE — CPU inference", page_icon="🧠")


@st.cache_resource(show_spinner="Fetching the VGRE runtime (libvgre_nn.so)…")
def _load_vgre():
    """Download the prebuilt native lib once, then import the vgre package."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "vgre"
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / "libvgre_nn.so"
    if not so.exists() or so.stat().st_size == 0:
        urllib.request.urlretrieve(_SO_URL, so)
    os.environ["VGRE_LIB_PATH"] = str(so)
    sys.path.insert(0, str(_REPO / "bindings" / "python"))
    import vgre  # noqa: E402  (import after VGRE_LIB_PATH/sys.path are set)
    return vgre


@st.cache_resource(show_spinner="Training the byte-level demo model (a few seconds)…")
def _train_model():
    """A tiny byte-level model trained once, then shared across all reruns/users."""
    vgre = _load_vgre()
    lm = vgre.LanguageModel(vocab=256, n_layer=2, d_model=128, n_head=4,
                            d_ff=256, max_seq=128, seed=0)
    data = list(_CORPUS.encode("utf-8"))
    T = 64
    steps = int(os.environ.get("VGRE_DEMO_STEPS", "400"))
    for step in range(steps):
        s = (step * 17) % (len(data) - T - 1)
        seq = data[s:s + T + 1]
        lm.train_step(seq[:-1], seq[1:], lr=3e-3)
    return lm


def _generate(lm, prompt, max_tokens, temperature):
    ids = list(prompt.encode("utf-8"))[-120:] or [0]
    out = lm.generate(ids, n_new=int(max_tokens), temperature=float(temperature))
    gen = out[len(ids):]
    return prompt + bytes(int(t) & 0xFF for t in gen).decode("utf-8", "replace")


st.title("🧠 VGRE inference — CPU, no GPU")
st.markdown(
    "Text generation from **VGRE**'s own from-scratch runtime, running on a plain "
    "CPU with no GPU. This is a small byte-level **demo model** trained live at "
    "startup — point the engine at a trained checkpoint for real quality.  \n"
    "Source: [github.com/etienne0114/vgre](https://github.com/etienne0114/vgre)"
)

with st.form("gen"):
    prompt = st.text_input("Prompt", value="VGRE runs machine learning")
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider("Max new tokens", 1, 128, 64)
    with col2:
        temperature = st.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.8, 0.1)
    go = st.form_submit_button("Generate", type="primary")

if go:
    try:
        lm = _train_model()
        with st.spinner("Generating on CPU…"):
            text = _generate(lm, prompt, max_tokens, temperature)
        st.text_area("Output", value=text, height=200)
    except Exception as exc:  # surface any runtime/link error to the page
        st.error(f"Generation failed: {exc}")
        st.exception(exc)
else:
    st.info("Enter a prompt and click **Generate**. The first run downloads the "
            "runtime and trains the demo model (a few seconds); after that it's cached.")
