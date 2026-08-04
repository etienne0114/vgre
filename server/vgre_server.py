#!/usr/bin/env python3
"""VGRE inference server — a dependency-free HTTP API over VGRE's own CPU LM.

Serves text generation from VGRE's in-tree language model with **no web
framework** (Python standard library only) and **no GPU**. It runs on the
lightweight LLVM-free ``libvgre_nn`` so the container stays small.

Out of the box it trains a tiny byte-level demo model at startup, so the server
runs anywhere with zero setup. Point it at your own checkpoint for real use.

Endpoints
  GET  /            -> a tiny HTML playground
  GET  /health      -> {"status": "ok"}
  GET  /info        -> model + backend info (and whether it's the demo model)
  POST /generate    -> {"prompt": "...", "max_tokens": 64, "temperature": 0.8,
                        "top_k": 0, "top_p": 1.0, "seed": 0}
                       returns {"text", "prompt_tokens", "generated_tokens"}

Environment
  VGRE_LIB_PATH    path to libvgre_nn.so / libvgre.so (else auto-discovered)
  VGRE_MODEL_PATH  checkpoint (.safetensors) to load instead of the demo model
  VGRE_MODEL_CFG   JSON with the checkpoint's config, e.g.
                   '{"vocab":256,"n_layer":4,"d_model":256,"n_head":4,"d_ff":1024,"max_seq":512}'
  VGRE_DEMO_STEPS  training steps for the demo model (default 400)
  VGRE_HOST        bind host (default 0.0.0.0)
  VGRE_PORT        bind port (default 8080)
  VGRE_MAX_TOKENS  hard cap on generated tokens per request (default 256)
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Make the in-tree bindings importable whether run from the repo or a container.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "bindings", "python"),
           os.path.join(_HERE, "bindings", "python")):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))

import vgre  # noqa: E402


# ── A tiny built-in byte tokenizer (no external files → runs anywhere) ───────
class ByteTokenizer:
    """UTF-8 byte-level tokenizer: every byte 0..255 is a token. Lossless and
    dependency-free — the honest zero-config default. Swap in a real BPE
    tokenizer (vgre.Tokenizer.load_hf) for production text quality."""
    vocab_size = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


# A small, self-contained corpus so the demo model learns *something* coherent.
_DEMO_CORPUS = (
    "VGRE runs and trains machine learning models on ordinary CPUs, with no GPU. "
    "It is built from scratch and stays lightweight, so it works everywhere. "
    "You can run it on a laptop, a server, or a cluster of ordinary computers. "
    "The goal is simple: make AI available to everyone, not only to those who own "
    "expensive graphics cards. VGRE trains, generates, and serves models on the "
    "machines people already have. "
) * 12


class Model:
    """Wraps a vgre.LanguageModel + a tokenizer, with a lock so concurrent HTTP
    requests serialize around the single native model (generation mutates C++
    KV-cache state, so it is not internally re-entrant)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.is_demo = True
        self.tok = ByteTokenizer()
        model_path = os.environ.get("VGRE_MODEL_PATH")
        if model_path and os.path.isfile(model_path):
            self._load_checkpoint(model_path)
        else:
            self._train_demo()

    def _load_checkpoint(self, path):
        cfg = json.loads(os.environ.get("VGRE_MODEL_CFG", "{}"))
        required = ("vocab", "n_layer", "d_model", "n_head", "d_ff", "max_seq")
        if not all(k in cfg for k in required):
            raise SystemExit(
                "VGRE_MODEL_PATH is set but VGRE_MODEL_CFG is missing keys "
                f"{required}. Provide the checkpoint's config as JSON.")
        self.cfg = cfg
        self.lm = vgre.LanguageModel(
            vocab=cfg["vocab"], n_layer=cfg["n_layer"], d_model=cfg["d_model"],
            n_head=cfg["n_head"], d_ff=cfg["d_ff"], max_seq=cfg["max_seq"], seed=0)
        self.lm.load(path)
        # A real tokenizer if provided, else bytes (vocab must match).
        tok_json = os.environ.get("VGRE_TOKENIZER")
        if tok_json and os.path.isfile(tok_json):
            self.tok = vgre.Tokenizer().load_hf(tok_json)
        self.is_demo = False
        print(f"[vgre-server] loaded checkpoint {path} (vocab={cfg['vocab']})",
              flush=True)

    def _train_demo(self):
        # Tiny model + a few hundred steps → runs in seconds, produces plausible
        # (not frontier-quality) text. Clearly reported as the demo model.
        self.cfg = dict(vocab=256, n_layer=2, d_model=128, n_head=4,
                        d_ff=256, max_seq=128)
        self.lm = vgre.LanguageModel(
            vocab=256, n_layer=2, d_model=128, n_head=4, d_ff=256,
            max_seq=128, seed=0)
        data = self.tok.encode(_DEMO_CORPUS)
        T = 64
        steps = int(os.environ.get("VGRE_DEMO_STEPS", "400"))
        print(f"[vgre-server] training a tiny demo model ({steps} steps)…",
              flush=True)
        for step in range(steps):
            s = (step * 17) % (len(data) - T - 1)
            seq = data[s:s + T + 1]
            self.lm.train_step(seq[:-1], seq[1:], lr=3e-3)
        print("[vgre-server] demo model ready.", flush=True)

    def generate(self, prompt, max_tokens, temperature, top_k, top_p, seed):
        ids = self.tok.encode(prompt) or [0]
        ids = ids[-self.cfg["max_seq"] + max_tokens - 1:]  # keep within context
        with self.lock:
            out = self.lm.generate(ids, n_new=max_tokens, temperature=temperature,
                                   top_k=top_k, top_p=top_p, seed=seed)
        gen = out[len(ids):]
        return self.tok.decode(gen), len(ids), len(gen)


MODEL = None  # set in main()
MAX_TOKENS_CAP = int(os.environ.get("VGRE_MAX_TOKENS", "256"))

_PLAYGROUND = """<!doctype html><meta charset=utf-8>
<title>VGRE inference</title>
<style>body{font-family:system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem}
textarea,input,button{font:inherit;width:100%;margin:.3rem 0;padding:.5rem}
pre{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:.5rem}</style>
<h1>VGRE inference</h1>
<p>Text generation on a CPU, no GPU. This is a small demo model unless a
checkpoint is loaded — see <code>/info</code>.</p>
<textarea id=p rows=3>VGRE runs machine learning</textarea>
<label>max tokens <input id=n type=number value=64></label>
<label>temperature <input id=t type=number step=0.1 value=0.8></label>
<button onclick=go()>Generate</button>
<pre id=o></pre>
<script>
async function go(){o.textContent='…';
 const r=await fetch('/generate',{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify({prompt:p.value,max_tokens:+n.value,temperature:+t.value})});
 const j=await r.json(); o.textContent=r.ok?(p.value+j.text):('error: '+JSON.stringify(j));}
</script>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # concise one-line logs
        sys.stderr.write("[vgre-server] " + (a[0] % a[1:]) + "\n")

    def do_GET(self):
        if self.path == "/health":
            self._send(200, json.dumps({"status": "ok"}))
        elif self.path == "/info":
            self._send(200, json.dumps({
                "backend": "vgre-cpu", "gpu": False, "demo_model": MODEL.is_demo,
                "config": MODEL.cfg, "tokenizer": type(MODEL.tok).__name__,
                "max_tokens_cap": MAX_TOKENS_CAP,
            }))
        elif self.path in ("/", "/index.html"):
            self._send(200, _PLAYGROUND, "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            prompt = str(req.get("prompt", ""))
            max_tokens = max(1, min(MAX_TOKENS_CAP, int(req.get("max_tokens", 64))))
            temperature = float(req.get("temperature", 0.0))
            top_k = int(req.get("top_k", 0))
            top_p = float(req.get("top_p", 1.0))
            seed = int(req.get("seed", 0))
        except Exception as e:
            self._send(400, json.dumps({"error": f"bad request: {e}"}))
            return
        try:
            text, ptoks, gtoks = MODEL.generate(
                prompt, max_tokens, temperature, top_k, top_p, seed)
            self._send(200, json.dumps({
                "text": text, "prompt_tokens": ptoks, "generated_tokens": gtoks,
            }))
        except Exception as e:
            self._send(500, json.dumps({"error": f"generation failed: {e}"}))


def main():
    global MODEL
    MODEL = Model()
    host = os.environ.get("VGRE_HOST", "0.0.0.0")
    port = int(os.environ.get("VGRE_PORT", "8080"))
    srv = ThreadingHTTPServer((host, port), Handler)
    kind = "demo model" if MODEL.is_demo else "loaded checkpoint"
    print(f"[vgre-server] serving {kind} on http://{host}:{port}  "
          f"(GET /health /info, POST /generate)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
