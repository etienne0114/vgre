"""Minimal, dependency-free YAML loader for VGRE deploy manifests.

VGRE's "real, no-stub, develop-missing-packages-from-scratch" standard means the
manifest validators must not require a third-party PyYAML install. This module
implements the YAML 1.1 subset the project's Kubernetes / Helm / Kustomize /
ArgoCD manifests actually use:

  * block mappings (``key: value`` and ``key:`` + indented children)
  * block sequences (``- item`` and ``-`` + indented children)
  * scalars: str, int, float, bool (true/false/yes/no/on/off), null (``~``/null)
  * single- and double-quoted strings (with the common escapes)
  * ``#`` comments (outside quotes) and ``---`` multi-document streams
  * inline flow collections ``[a, b]`` and ``{a: b}``

It is intentionally NOT a full YAML 1.2 parser (no anchors/aliases, tags, block
scalars, or complex keys) — just enough to load and structurally validate the
manifests. The public surface mirrors the slice of PyYAML the validators use:
``safe_load``, ``safe_load_all``, and ``YAMLError``.
"""
from __future__ import annotations

from typing import Any, List, Tuple


class YAMLError(Exception):
    """Raised on malformed input, matching pyyaml's exception name."""


# ── lexing helpers ──────────────────────────────────────────────────────────
def _strip_comment(line: str) -> str:
    out, quote = [], None
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if quote:
            out.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:   # keep escapes intact
                out.append(line[i + 1]); i += 2; continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch; out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _split_flow(s: str) -> List[str]:
    out, depth, cur, quote = [], 0, [], None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch; cur.append(ch)
        elif ch in "[{":
            depth += 1; cur.append(ch)
        elif ch in "]}":
            depth -= 1; cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    tail = "".join(cur)
    if tail.strip():
        out.append(tail)
    return out


def _scalar(tok: str) -> Any:
    tok = tok.strip()
    if tok == "" or tok == "~" or tok.lower() == "null":
        return None
    low = tok.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        inner = tok[1:-1]
        if tok[0] == '"':
            inner = (inner.replace('\\"', '"').replace("\\n", "\n")
                          .replace("\\t", "\t").replace("\\\\", "\\"))
        return inner
    if tok.startswith("[") and tok.endswith("]"):
        body = tok[1:-1].strip()
        return [_scalar(x) for x in _split_flow(body)] if body else []
    if tok.startswith("{") and tok.endswith("}"):
        body = tok[1:-1].strip()
        d: dict = {}
        for pair in _split_flow(body):
            if ":" in pair:
                k, v = pair.split(":", 1)
                d[_scalar(k)] = _scalar(v)
        return d
    for conv in (int, float):
        try:
            return conv(tok)
        except ValueError:
            pass
    return tok


# ── block parser (recursive over an indentation-grouped line list) ──────────
def _content_lines(text: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if line.strip() == "":
            continue
        rows.append((_indent(line), line.strip()))
    return rows


def _parse(rows: List[Tuple[int, str]], pos: int, indent: int) -> Tuple[Any, int]:
    if pos >= len(rows):
        return None, pos
    is_seq = rows[pos][1].startswith("- ") or rows[pos][1] == "-"
    if is_seq:
        seq: List[Any] = []
        while pos < len(rows) and rows[pos][0] == indent and \
                (rows[pos][1].startswith("- ") or rows[pos][1] == "-"):
            item = rows[pos][1][1:].strip()  # drop leading '-'
            if item == "":
                child, pos = _parse(rows, pos + 1, _next_indent(rows, pos + 1, indent))
                seq.append(child)
            elif ":" in _unquoted_head(item):
                # inline first key of a mapping item; re-read with synthetic indent
                rows[pos] = (indent + 2, item)
                child, pos = _parse(rows, pos, indent + 2)
                seq.append(child)
            else:
                seq.append(_scalar(item)); pos += 1
        return seq, pos

    mapping: dict = {}
    while pos < len(rows) and rows[pos][0] == indent:
        line = rows[pos][1]
        if line.startswith("- "):
            break
        if ":" not in _unquoted_head(line):
            raise YAMLError(f"expected 'key: value', got: {line!r}")
        key, _, val = line.partition(":")
        key = _scalar(key.strip())
        val = val.strip()
        if val == "":
            ni = _next_indent(rows, pos + 1, indent)
            if ni > indent:
                child, pos = _parse(rows, pos + 1, ni)
                mapping[key] = child
            else:
                mapping[key] = None; pos += 1
        else:
            mapping[key] = _scalar(val); pos += 1
    return mapping, pos


def _unquoted_head(line: str) -> str:
    # the part of the line up to the first ':' that is not inside quotes/flow
    depth, quote, out = 0, None, []
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch; out.append(ch)
        elif ch in "[{":
            depth += 1; out.append(ch)
        elif ch in "]}":
            depth -= 1; out.append(ch)
        elif ch == ":" and depth == 0:
            out.append(":"); break
        else:
            out.append(ch)
    return "".join(out)


def _next_indent(rows: List[Tuple[int, str]], pos: int, parent: int) -> int:
    return rows[pos][0] if pos < len(rows) else parent


# ── public API (PyYAML-compatible subset) ───────────────────────────────────
def safe_load(text: str) -> Any:
    docs = list(safe_load_all(text))
    return docs[0] if docs else None


def safe_load_all(text):
    if hasattr(text, "read"):
        text = text.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    # split the stream on '---' / '...' document markers
    docs, cur = [], []
    for raw in text.splitlines():
        s = raw.strip()
        if s in ("---", "..."):
            docs.append("\n".join(cur)); cur = []
        else:
            cur.append(raw)
    docs.append("\n".join(cur))
    for d in docs:
        rows = _content_lines(d)
        if not rows:
            continue
        try:
            value, _ = _parse(rows, 0, rows[0][0])
        except IndexError as e:                       # malformed structure
            raise YAMLError(str(e)) from e
        yield value
