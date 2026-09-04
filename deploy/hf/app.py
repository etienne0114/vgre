"""VGRE inference — free Hugging Face **Gradio** Space (CPU, no GPU).

HF's Docker SDK is paid; Gradio Spaces are free. This app runs VGRE's own
from-scratch language model through the prebuilt, LLVM-free libvgre_nn.so shipped
next to it (no compilation on HF). It serves a small web UI plus a /generate API
(Gradio exposes the function as an API automatically).

Assembled + uploaded by .github/workflows/deploy-hf.yml, which puts this file,
requirements.txt, README.md, the vgre/ package, and libvgre_nn.so in the Space.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# The vgre package and the native lib are shipped alongside this file.
sys.path.insert(0, _HERE)
os.environ.setdefault("VGRE_LIB_PATH", os.path.join(_HERE, "libvgre_nn.so"))

import numpy as np  # noqa: E402
import gradio as gr  # noqa: E402
import vgre  # noqa: E402

_CORPUS = (
    "VGRE runs and trains machine learning models on ordinary CPUs, with no GPU. "
    "It is built from scratch and stays lightweight, so it works everywhere. "
    "You can run it on a laptop, a server, or a cluster of ordinary computers. "
    "The goal is simple: make AI available to everyone, not only to those who own "
    "expensive graphics cards. "
) * 12

# A tiny byte-level demo model, trained at startup so the Space runs with zero
# setup. It is a demo — small and CPU-only — not a frontier model.
LM = vgre.LanguageModel(vocab=256, n_layer=2, d_model=128, n_head=4,
                        d_ff=256, max_seq=128, seed=0)
_data = list(_CORPUS.encode("utf-8"))
_T = 64
for _step in range(int(os.environ.get("VGRE_DEMO_STEPS", "400"))):
    _s = (_step * 17) % (len(_data) - _T - 1)
    _seq = _data[_s:_s + _T + 1]
    LM.train_step(_seq[:-1], _seq[1:], lr=3e-3)


def generate(prompt: str, max_tokens: int, temperature: float) -> str:
    ids = list(prompt.encode("utf-8")) or [0]
    ids = ids[-120:]
    out = LM.generate(ids, n_new=int(max_tokens), temperature=float(temperature))
    gen = out[len(ids):]
    return prompt + bytes(int(t) & 0xFF for t in gen).decode("utf-8", "replace")


_iface_kwargs = dict(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", value="VGRE runs machine learning"),
        gr.Slider(1, 128, value=64, step=1, label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.8, step=0.1, label="Temperature (0 = greedy)"),
    ],
    outputs=gr.Textbox(label="Output"),
    title="VGRE inference — CPU, no GPU",
    description=(
        "Text generation from VGRE's own from-scratch runtime on a CPU. This is a "
        "small byte-level **demo model**; point the engine at a trained checkpoint "
        "for real quality. Source: https://github.com/etienne0114/vgre"
    ),
)

# The kwarg that disables the flag button was renamed across Gradio majors
# (4.x: allow_flagging="never"; 5.x: flagging_mode="never"; 6.x dropped it).
# HF picks the Gradio version, so try each name and fall back to none — the app
# must never die on an unknown-keyword TypeError just to hide a button.
for _flag in ({"allow_flagging": "never"}, {"flagging_mode": "never"}, {}):
    try:
        demo = gr.Interface(**_iface_kwargs, **_flag)
        break
    except TypeError:
        continue

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
