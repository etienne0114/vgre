"""vgre command-line interface.

Installed by the wheel as the ``vgre`` console script (see pyproject
``[project.scripts]``), so after ``pip install vgre`` the commands below are on
your PATH. Everything here uses only the in-tree engine — no GPU, no network.

    vgre --version              # version + native backend status
    vgre info                   # detailed backend / library / platform info
    vgre generate --prompt "…"  # generate text (trains a tiny demo model if
                                #   no --model is given, so it works out of the box)
    vgre train  --corpus f.txt --out model.vgre
    vgre tokenize --encode "…"  # byte / BPE tokenization helpers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__

# A tiny corpus so `vgre generate` produces something without any download.
_DEMO_CORPUS = (
    "the virtual gpu runtime runs cuda on the cpu. "
    "it trains and runs a language model with no gpu and no network. "
    "kernels execute on four bit-exact cpu tiers. "
) * 24


# ── native backend introspection ────────────────────────────────────────────
def _native_info():
    """Return (available, library_path, native_version_string)."""
    try:
        from . import _native  # noqa: WPS433 (local import: keeps --help cheap)
        avail = bool(getattr(_native, "NATIVE_AVAILABLE", False))
        path = _native._find_library() if avail else None
        ver = None
        lib = getattr(_native, "_lib", None)
        if avail and lib is not None:
            try:
                v = lib.vgre_get_version()
                ver = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            except Exception:
                ver = None
        return avail, path, ver
    except Exception:
        return False, None, None


def _require_native() -> None:
    avail, _, _ = _native_info()
    if not avail:
        sys.stderr.write(
            "error: the native VGRE library is not available in this install, so "
            "model commands cannot run.\n"
            "       Reinstall the wheel (it bundles the native library) into a venv:\n"
            "         pip install --force-reinstall vgre\n"
        )
        raise SystemExit(2)


# ── a byte-level tokenizer (no training / no files needed) ───────────────────
class _ByteTokenizer:
    """UTF-8 byte tokenizer — the zero-setup default; vocab is 256."""

    vocab = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


def _load_tokenizer(tok_path: Optional[str]):
    """HF tokenizer.json when given, else the byte tokenizer."""
    if tok_path:
        from . import Tokenizer
        return Tokenizer().load_hf(tok_path), None
    return _ByteTokenizer(), 256


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_version(_args) -> int:
    avail, _, ver = _native_info()
    print(f"vgre {__version__}")
    print(f"native backend: {'available' if avail else 'unavailable (import-only)'}")
    if ver:
        print(f"native library: {ver}")
    return 0


def cmd_info(_args) -> int:
    import platform

    avail, path, ver = _native_info()
    rows = [
        ("vgre (python)", __version__),
        ("native backend", "available" if avail else "unavailable"),
        ("native version", ver or "-"),
        ("native library", path or "-"),
        ("python", f"{platform.python_version()} ({platform.machine()})"),
        ("platform", f"{platform.system()} {platform.release()}"),
    ]
    try:
        import numpy
        rows.append(("numpy", numpy.__version__))
    except Exception:
        rows.append(("numpy", "(not installed)"))
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k.ljust(width)} : {v}")
    return 0


def _train_demo(steps: int, seq: int = 64):
    """Build + briefly train a tiny byte-level model; returns (lm, tokenizer)."""
    from . import LanguageModel

    tok = _ByteTokenizer()
    lm = LanguageModel(vocab=256, n_layer=2, d_model=128, n_head=4, d_ff=256, max_seq=256)
    data = tok.encode(_DEMO_CORPUS)
    for _ in range(max(1, steps)):
        for i in range(0, max(1, len(data) - seq - 1), seq):
            chunk = data[i : i + seq + 1]
            if len(chunk) < 2:
                continue
            lm.train_step(chunk[:-1], chunk[1:], lr=3e-3)
    return lm, tok


def cmd_generate(args) -> int:
    _require_native()
    from . import LanguageModel

    if args.model:
        cfg_path = args.model + ".json"
        if not os.path.exists(cfg_path):
            sys.stderr.write(
                f"error: model config sidecar not found: {cfg_path}\n"
                "       Models saved by `vgre train` write a <model>.json next to "
                "the weights.\n"
            )
            return 2
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        lm = LanguageModel(
            vocab=cfg["vocab"], n_layer=cfg["n_layer"], d_model=cfg["d_model"],
            n_head=cfg["n_head"], d_ff=cfg.get("d_ff", 0), max_seq=cfg.get("max_seq", 256),
        )
        loader = {".gguf": lm.load_gguf}.get(os.path.splitext(args.model)[1], lm.load)
        loader(args.model)
        tok, _ = _load_tokenizer(args.tokenizer or cfg.get("tokenizer"))
    else:
        sys.stderr.write("[vgre] no --model given; training a tiny demo model…\n")
        lm, tok = _train_demo(args.demo_steps)

    ids = tok.encode(args.prompt) or [0]
    out = lm.generate(
        ids, n_new=args.max_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    print(tok.decode(out[len(ids):]))
    return 0


def cmd_train(args) -> int:
    _require_native()
    from . import LanguageModel, Tokenizer

    with open(args.corpus, encoding="utf-8", errors="replace") as fh:
        corpus = fh.read()
    if not corpus.strip():
        sys.stderr.write("error: corpus is empty\n")
        return 2

    if args.merges > 0:
        tok = Tokenizer().train(corpus, num_merges=args.merges)
        vocab = 256 + args.merges
        tok_sidecar = None  # BPE tokenizer state is internal to this run
    else:
        tok = _ByteTokenizer()
        vocab = 256
        tok_sidecar = None

    lm = LanguageModel(
        vocab=vocab, n_layer=args.layers, d_model=args.d_model,
        n_head=args.heads, d_ff=args.d_ff, max_seq=args.max_seq,
    )
    data = tok.encode(corpus)
    seq = min(args.max_seq, 128)
    print(f"[vgre] training {lm.num_parameters:,} params on {len(data):,} tokens "
          f"for {args.steps} epoch(s)…")
    for epoch in range(args.steps):
        total, n = 0.0, 0
        for i in range(0, max(1, len(data) - seq - 1), seq):
            chunk = data[i : i + seq + 1]
            if len(chunk) < 2:
                continue
            total += lm.train_step(chunk[:-1], chunk[1:], lr=args.lr)
            n += 1
        print(f"  epoch {epoch + 1}/{args.steps}  loss={total / max(1, n):.4f}")

    lm.save(args.out)
    cfg = dict(vocab=vocab, n_layer=args.layers, d_model=args.d_model,
               n_head=args.heads, d_ff=args.d_ff, max_seq=args.max_seq,
               tokenizer=tok_sidecar)
    with open(args.out + ".json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[vgre] saved {args.out} (+ {args.out}.json). "
          f"Generate with: vgre generate --model {args.out} --prompt \"…\"")
    return 0


def cmd_tokenize(args) -> int:
    _require_native()
    if args.encode is not None:
        tok, _ = _load_tokenizer(args.tokenizer)
        print(",".join(str(i) for i in tok.encode(args.encode)))
        return 0
    if args.decode is not None:
        tok, _ = _load_tokenizer(args.tokenizer)
        ids = [int(x) for x in args.decode.replace(",", " ").split()]
        print(tok.decode(ids))
        return 0
    if args.train_corpus is not None:
        from . import Tokenizer
        with open(args.train_corpus, encoding="utf-8", errors="replace") as fh:
            corpus = fh.read()
        Tokenizer().train(corpus, num_merges=args.merges)
        print(f"[vgre] trained BPE tokenizer: {256 + args.merges} tokens "
              f"({args.merges} merges) from {args.train_corpus}")
        return 0
    sys.stderr.write("error: give one of --encode, --decode, or --train\n")
    return 2


# ── argument parser ──────────────────────────────────────────────────────────
# Cluster / node subcommands are provided by the source install (vgre_sync.sh /
# install_local.sh) as sibling `vgre-<name>` tools. Delegate to them so the pip
# `vgre` command and the source dispatcher expose the same `vgre <subcommand>`
# interface (otherwise `vgre token` / `vgre connect-check` would error here).
_CLUSTER_SUBCOMMANDS = {
    "start": "vgre-start",
    "worker": "vgre-worker",
    "token": "vgre-token",
    "discover": "vgre-discover",
    "connect-check": "vgre-connect-check",
    "dashboard": "vgre-dashboard",
}


def _find_cluster_tool(exe: str) -> Optional[str]:
    import shutil
    found = shutil.which(exe)
    if found:
        return found
    candidates = [os.path.expanduser(os.path.join("~", ".local", "bin", exe))]
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            base = os.path.join(local, "VGRE", "scripts", exe)
            candidates += [base + ".bat", base + ".ps1"]
    return next((c for c in candidates if os.path.exists(c)), None)


def _delegate_cluster(name: str, rest: List[str]) -> int:
    import subprocess
    path = _find_cluster_tool(_CLUSTER_SUBCOMMANDS[name])
    if not path:
        sys.stderr.write(
            f"vgre: '{name}' is a cluster/runtime tool that comes with the source "
            f"install, not the pip wheel.\n"
            f"      Set it up with:\n"
            f"        git clone https://github.com/etienne0114/vgre && cd vgre && ./scripts/vgre_sync.sh\n"
            f"      (the model commands — generate/train/tokenize/info — work from the wheel.)\n")
        return 127
    return subprocess.call([path] + rest)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vgre",
        description="Virtual GPU Runtime — run CUDA and a local language model on the CPU.",
        epilog="Cluster / node subcommands (vgre start, vgre worker, vgre token, "
               "vgre discover, vgre connect-check, vgre dashboard) come from the "
               "source install (./scripts/vgre_sync.sh); this wheel provides the "
               "model/runtime commands above and delegates the cluster ones.",
    )
    p.add_argument("--version", action="store_true", help="print version and native backend status")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("version", help="print version and native backend status").set_defaults(func=cmd_version)
    sub.add_parser("info", help="detailed backend / library / platform info").set_defaults(func=cmd_info)

    g = sub.add_parser("generate", help="generate text (trains a tiny demo if no --model)")
    g.add_argument("--model", help="checkpoint saved by `vgre train`, or a .gguf file")
    g.add_argument("--tokenizer", help="HF tokenizer.json (default: byte-level)")
    g.add_argument("--prompt", default="the virtual gpu runtime ", help="text prompt")
    g.add_argument("--max-tokens", dest="max_tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", dest="top_k", type=int, default=40)
    g.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    g.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--demo-steps", dest="demo_steps", type=int, default=30,
                   help="training epochs for the built-in demo model")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train a small language model on a text corpus")
    t.add_argument("--corpus", required=True, help="UTF-8 text file to train on")
    t.add_argument("--out", required=True, help="output checkpoint path")
    t.add_argument("--merges", type=int, default=0, help="BPE merges (0 = byte-level, vocab 256)")
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--d-model", dest="d_model", type=int, default=256)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--d-ff", dest="d_ff", type=int, default=0)
    t.add_argument("--max-seq", dest="max_seq", type=int, default=256)
    t.add_argument("--steps", type=int, default=3, help="training epochs")
    t.add_argument("--lr", type=float, default=3e-3)
    t.set_defaults(func=cmd_train)

    k = sub.add_parser("tokenize", help="byte / BPE tokenization helpers")
    k.add_argument("--encode", metavar="TEXT", help="encode TEXT to token ids")
    k.add_argument("--decode", metavar="IDS", help="decode comma/space-separated IDS to text")
    k.add_argument("--train", dest="train_corpus", metavar="FILE", help="train a BPE tokenizer on FILE")
    k.add_argument("--tokenizer", help="HF tokenizer.json for --encode/--decode")
    k.add_argument("--merges", type=int, default=512, help="BPE merges for --train")
    k.set_defaults(func=cmd_tokenize)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Delegate cluster subcommands to the installed vgre-<name> tools before
    # argparse (which would reject them as an invalid choice).
    if argv and argv[0] in _CLUSTER_SUBCOMMANDS:
        return _delegate_cluster(argv[0], argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        return cmd_version(args)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except FileNotFoundError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
