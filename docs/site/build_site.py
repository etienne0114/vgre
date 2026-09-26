#!/usr/bin/env python3
"""Generate the VGRE documentation website (static HTML) from the page content
defined below. No framework, no build dependency — emits plain files under
docs/site/ that open directly in a browser (file://) or from any static host.

    python3 docs/site/build_site.py

Content is authored here as lightweight markup (headings, paragraphs, code
blocks, tables, callouts, cards) and rendered into the shared shell (header,
left nav, right auto-TOC, search). Everything is real: commands, env vars, and
API signatures are taken verbatim from README.md / the shipped tools /
docs/api_reference.md and the shipped tools.
"""
import hashlib
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def asset_ver(rel):
    """Short content hash of an asset, appended as ?v=… so browsers refetch it the
    moment it changes (otherwise a cached style.css/JS hides content/CSS updates)."""
    try:
        with open(os.path.join(HERE, rel), "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:10]
    except OSError:
        return "0"

# Public site URL used for canonical links / the dashboard "Docs" button.
SITE_URL = "https://etienne0114.github.io/vgre"

# ── Navigation (also the page order for prev/next) ────────────────────────────
NAV = [
    ("Getting Started", [
        ("index.html", "Introduction"),
        ("quickstart.html", "Quick Start"),
        ("downloads.html", "Downloads"),
        ("installation.html", "Installation"),
        ("system-requirements.html", "System Requirements"),
    ]),
    ("Guides", [
        ("running-cuda.html", "Running CUDA Apps"),
        ("cluster.html", "Distributed Clusters"),
        ("discovery.html", "Public IP Discovery"),
        ("token-management.html", "Token Management"),
        ("dashboard.html", "The Dashboard"),
        ("local-ai.html", "Local AI: LoRA · RAG · LM"),
        ("debugging.html", "Debugging PTX (GDB)"),
        ("advanced.html", "Advanced Features"),
    ]),
    ("Reference", [
        ("env-vars.html", "Environment Variables"),
        ("api.html", "API Reference"),
        ("cli.html", "CLI Tools"),
        ("production-status.html", "Production Status"),
    ]),
    ("About & Help", [
        ("architecture.html", "Architecture"),
        ("troubleshooting.html", "Troubleshooting"),
        ("faq.html", "FAQ"),
    ]),
]

FLAT = [(f, t) for _, items in NAV for f, t in items]


# ── Tiny content DSL ──────────────────────────────────────────────────────────
def h1(t, lead=""):
    s = f"<h1>{html.escape(t)}</h1>"
    if lead:
        s += f'<p class="lead">{lead}</p>'
    return s


def h2(t):
    return f"<h2>{html.escape(t)}</h2>"


def h3(t):
    return f"<h3>{html.escape(t)}</h3>"


def p(t):
    return f"<p>{t}</p>"


_LANG_LABEL = {
    "bash": "bash", "sh": "bash", "shell": "bash", "console": "bash",
    "powershell": "powershell", "ps1": "powershell",
    "python": "python", "py": "python",
    "c": "c", "cpp": "c++", "cxx": "c++",
    "json": "json", "yaml": "yaml", "text": "text",
}


def code(body, lang="bash"):
    """A terminal-styled code block: a title bar (traffic-light dots, the
    language label, and a Copy button) over the escaped source. Syntax colors
    and the copy action are wired up by assets/docs.js (progressive
    enhancement — the raw source is always visible and selectable)."""
    label = _LANG_LABEL.get(lang, lang)
    return (
        f'<div class="terminal" data-lang="{html.escape(label)}">'
        '<div class="terminal-bar">'
        '<span class="term-dots"><i></i><i></i><i></i></span>'
        f'<span class="term-lang">{html.escape(label)}</span>'
        '<button class="copy-btn" type="button" aria-label="Copy code">Copy</button>'
        '</div>'
        f'<pre><code class="lang-{lang}">{html.escape(body.strip())}</code></pre>'
        '</div>'
    )


def ul(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def ol(items):
    return "<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def table(headers, rows):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def callout(body, kind="", title=None):
    t = title or {"tip": "Tip", "warn": "Warning"}.get(kind, "Note")
    return f'<div class="callout {kind}"><div class="callout-title">{t}</div>{body}</div>'


def cards(items):  # (icon, title, desc, href)
    c = "".join(
        f'<a class="card" href="{href}"><div class="card-icon">{icon}</div>'
        f"<h3>{title}</h3><p>{desc}</p></a>"
        for icon, title, desc, href in items
    )
    return f'<div class="card-grid">{c}</div>'


def badges(items):  # (text, ok?)
    b = "".join(f'<span class="badge {"ok" if ok else ""}">{t}</span>' for t, ok in items)
    return f'<div class="badges">{b}</div>'


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_index():
    return "".join([
        h1("VGRE — Virtual GPU Runtime",
           "Run unmodified CUDA on any CPU — no GPU required. Plus a full local-AI "
           "stack: train, fine-tune with LoRA, retrieve with an in-tree vector "
           "index, and generate — all offline, all dependency-free."),
        badges([("Linux CI-green", True), ("macOS CI-green", True), ("Windows CI-green", True),
                ("LLVM-free build", True), ("CUDA ~95%", False), ("Zero vendor lock-in", True)]),
        p("VGRE intercepts CUDA and OpenCL API calls and executes kernels on the CPU. "
          "It <strong>does not need LLVM to run kernels</strong>: a from-scratch CUDA-C "
          "front-end (own lexer/parser → AST) feeds a four-tier CPU backend — a PTX "
          "interpreter, a compiled-closure tier with a cooperative fiber executor "
          "(<code>__shared__</code> / <code>__syncthreads</code> / the full warp-intrinsic "
          "surface), a hand-emitted native x86-64 JIT, and an optional own SSA optimizing "
          "backend — every tier held bit-exact against the others. An optional LLVM ORC "
          "JIT is the high-performance path when LLVM is present. The same engine ships an "
          "in-tree machine-learning stack so you go from CUDA experiments to real model "
          "training and RAG without a second toolchain."),
        cards([
            ("🚀", "Quick Start", "Install the wheel and run your first CUDA program on CPU in minutes.", "quickstart.html"),
            ("⬇️", "Downloads", "Prebuilt, self-contained wheels for Linux, macOS and Windows.", "downloads.html"),
            ("🧠", "Local AI", "LoRA fine-tuning, HNSW vector search, and the in-tree language model.", "local-ai.html"),
            ("🔗", "Clusters", "Scale across machines over WAN with encrypted, authenticated channels.", "cluster.html"),
            ("🐞", "PTX Debugging", "Step through PTX kernels with a stock gdb — CUDA-GDB-style, no GPU.", "debugging.html"),
            ("📚", "API Reference", "Python, C ABI, CUDA-runtime shim, and dashboard telemetry APIs.", "api.html"),
        ]),
        h2("Why VGRE"),
        ul([
            "<strong>No GPU needed</strong> — learn CUDA, run CI, and develop on any x86-64 or ARM64 machine.",
            "<strong>No toolchain needed</strong> — the LLVM-free build runs the full suite (379 tests) with only a C++17 compiler; the prebuilt wheel needs just Python + NumPy.",
            "<strong>One stack, both jobs</strong> — GPU emulation and a real ML training/inference/RAG stack in the same runtime.",
            "<strong>Built from scratch</strong> — the CUDA-C front-end, four execution tiers, JIT, tokenizer, vector index, collectives, and crypto are all in-tree, not third-party wrappers — no vendor lock-in, no per-seat dependency cost.",
            "<strong>Private by default</strong> — the entire embed → index → retrieve → generate → fine-tune loop runs offline; your data never leaves the machine.",
        ]),
        callout(p("<strong>All three platforms pass the full <code>ctest</code> suite in CI</strong> — "
                  "399 tests with LLVM and 379 in the LLVM-free build on Linux x86-64 (the required "
                  "job), and the whole suite runs green on macOS (Apple Silicon) and Windows "
                  "(clang-cl). Hardware-only gaps (physical GPU PMU counters, Metal MPS, GPUDirect "
                  "RDMA) are listed in the <a href=\"faq.html\">FAQ &amp; Troubleshooting</a>."), "tip"),
    ])


REL = "https://github.com/etienne0114/vgre/releases"
DL = REL + "/download/v0.1.0"


def page_downloads():
    return "".join([
        h1("Downloads", "Prebuilt, self-contained wheels — no compiler, no CUDA, no GPU."),
        p("Every <a href=\"" + REL + "\">GitHub Release</a> ships a <strong>self-contained, "
          "LLVM-free</strong> wheel per platform. Each bundles the native engine, so you need "
          "only <strong>Python 3.8+ and NumPy</strong>. The wheels are built and smoke-tested "
          "(import + train a language model) on Linux, macOS and Windows runners in CI."),
        h2("Latest release — v0.1.0"),
        p("Pick your platform, download the wheel, then copy the <code>pip install</code> "
          "command (each block has a <strong>Copy</strong> button):"),
        h3("🐧 Linux x86-64"),
        p('<a href="' + DL + '/vgre-0.1.0-py3-none-linux_x86_64.whl">⬇ Download vgre-0.1.0-py3-none-linux_x86_64.whl</a>'),
        code("pip install vgre-0.1.0-py3-none-linux_x86_64.whl"),
        h3("🍎 macOS (Apple Silicon)"),
        p('<a href="' + DL + '/vgre-0.1.0-py3-none-macosx_10_13_universal2.whl">⬇ Download vgre-0.1.0-py3-none-macosx_10_13_universal2.whl</a>'),
        code("pip install vgre-0.1.0-py3-none-macosx_10_13_universal2.whl"),
        h3("🪟 Windows x86-64"),
        p('<a href="' + DL + '/vgre-0.1.0-py3-none-win_amd64.whl">⬇ Download vgre-0.1.0-py3-none-win_amd64.whl</a>'),
        code("pip install vgre-0.1.0-py3-none-win_amd64.whl"),
        h2("Install straight from the release URL"),
        p("No manual download — <code>pip</code> fetches the wheel for you (swap the filename "
          "for your platform):"),
        code("pip install " + DL + "/vgre-0.1.0-py3-none-linux_x86_64.whl\n"
             "python -c \"import vgre; print('native:', vgre.NATIVE_AVAILABLE)\""),
        h2("Verify it works"),
        p("The bundled native library runs CUDA-C kernels and trains/serves the in-tree "
          "transformer LM on CPU:"),
        code("import vgre\n"
             "tok = vgre.Tokenizer().train(open('corpus.txt').read(), num_merges=1024)\n"
             "lm  = vgre.LanguageModel(vocab=tok.vocab_size, n_layer=6, d_model=256, n_head=8)\n"
             "# lm.train_step(...) then lm.generate(...) — CPU only, no GPU/LLVM/BLAS.",
             "python"),
        h2("Container image (GHCR)"),
        p("A multi-arch (amd64/arm64) LLVM-free image is published on a tagged release:"),
        code("docker run --rm ghcr.io/etienne0114/vgre:0.1.0 --version"),
        h2("Build from source"),
        p("Prefer to build your own? See the <a href=\"installation.html\">Installation</a> guide "
          "— including the zero-burden LLVM-free build and <code>bindings/python/build_wheel.sh</code> "
          "to produce a wheel for your exact platform."),
        callout(p("The macOS wheel is built on Apple Silicon (arm64). Intel-Mac (x86-64) users "
                  "should build from source until an x86-64 macOS wheel is added to the release."), "warn"),
    ])


def page_quickstart():
    return "".join([
        h1("Quick Start", "From a one-line install to a running CUDA kernel on CPU in minutes."),
        h2("Fastest: install the prebuilt wheel (no toolchain)"),
        p("Each GitHub Release ships a self-contained, <strong>LLVM-free</strong> wheel per "
          "platform (Linux / macOS / Windows). It bundles the native engine, so you need "
          "nothing but Python 3.8+ and NumPy — no compiler, no CUDA, no GPU:"),
        code("# Grab the wheel for your platform from the latest release, then:\n"
             "pip install vgre-0.1.0-py3-none-linux_x86_64.whl   # or -macosx_* / -win_amd64\n\n"
             "python -c \"import vgre; print('native:', vgre.NATIVE_AVAILABLE)\""),
        callout(p("Releases: <a href=\"https://github.com/etienne0114/vgre/releases\">"
                  "github.com/etienne0114/vgre/releases</a>. The wheel runs CUDA-C kernels "
                  "and trains/serves the in-tree transformer LM entirely on CPU."), "tip"),
        h2("Build from source (Linux / macOS)"),
        code("git clone https://github.com/etienne0114/vgre.git\n"
             "cd vgre\n"
             "bash install_local.sh"),
        p("<code>install_local.sh</code> detects and installs missing dependencies "
          "(CMake, LLVM, OpenMP, Flutter), builds the native engine and dashboard, "
          "writes <code>~/.vgre/env</code>, creates your auth token, and links the "
          "CLI tools into <code>~/.local/bin</code>."),
        p("Open a <strong>new terminal</strong>, then:"),
        code("vgre-dashboard          # launch the real-time monitor\n"
             "vgre-start --test       # local master + worker self-test"),
        h2("Run a CUDA program"),
        p("Build your CUDA source as usual, then run it against the VGRE runtime shim "
          "instead of the NVIDIA driver:"),
        code("# Linux — preload the CUDA-runtime shim\n"
             "LD_PRELOAD=$VGRE_INSTALL_DIR/lib/libvgre_cudart.so ./my_cuda_app\n\n"
             "# macOS\n"
             "DYLD_INSERT_LIBRARIES=$VGRE_INSTALL_DIR/lib/libvgre_cudart.dylib ./my_cuda_app"),
        callout(p("No source changes are needed. VGRE implements ~95% of the commonly "
                  "used CUDA Runtime and Driver APIs; unsupported calls return a clear "
                  "error rather than silently misbehaving."), "tip"),
        h2("Try the local language model"),
        code("import vgre\n"
             "tok = vgre.Tokenizer().train(open('corpus.txt').read(), num_merges=512)\n"
             "lm  = vgre.LanguageModel(vocab=tok.vocab_size, n_layer=4, d_model=256, n_head=4)\n"
             "# ... train, then:\n"
             "print(tok.decode(lm.generate(tok.encode('Once upon a time'), n_new=40)))",
             "python"),
        p("See <a href=\"local-ai.html\">Local AI</a> for LoRA fine-tuning and RAG."),
    ])


def page_installation():
    return "".join([
        h1("Installation", "Manual build steps and platform notes."),
        callout(p("The quickest way to get running is the prebuilt wheel — see "
                  "<a href=\"downloads.html\">Downloads</a>. Build from source when you want the "
                  "dashboard, the cluster tools, or to hack on the engine."), "tip"),
        h2("Prerequisites"),
        h3("Linux (Ubuntu / Debian)"),
        code("# Full build (with the optional LLVM JIT):\n"
             "sudo apt install -y cmake ninja-build clang llvm-18-dev \\\n"
             "    libomp-dev libssl-dev libsqlite3-dev libkeyutils-dev zlib1g-dev\n\n"
             "# LLVM-free build needs no llvm-18-dev / libomp-dev — just clang + the above."),
        h3("macOS (Homebrew — paths discovered automatically at configure time)"),
        code("brew install cmake llvm@18 libomp ninja sqlite lapack\n"
             "# LLVM-free build needs only:  brew install cmake ninja sqlite"),
        h3("Windows (PowerShell, VS 2022 present)"),
        code("# One-shot: installs LLVM 18 + Ninja + CMake and configures the build.\n"
             "powershell -ExecutionPolicy Bypass -File scripts\\Install-BuildTools.ps1\n"
             "powershell -ExecutionPolicy Bypass -File scripts\\Install-VGRETools.ps1\n\n"
             "# Or by hand with Chocolatey:\n"
             "choco install -y llvm ninja sqlite cmake", "powershell"),
        h3("Flutter (optional — only for the dashboard)"),
        code("sudo snap install flutter --classic     # Linux\n"
             "brew install --cask flutter              # macOS\n"
             "choco install flutter                    # Windows"),
        h2("Build — Linux / macOS (full, with LLVM JIT)"),
        code("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release\n"
             "cmake --build build -j$(nproc)\n"
             "ctest --test-dir build -j$(nproc)     # full suite: 399/399 on Linux"),
        h2("Build — Windows (clang-cl + Ninja)"),
        p("Windows builds LLVM-free-friendly with clang-cl; the full suite is green in CI "
          "on <code>windows-2022</code>. From a <em>Developer PowerShell for VS 2022</em>:"),
        code("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `\n"
             "  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `\n"
             "  -DVGRE_ENABLE_OPENMP=OFF -DVGRE_WARNINGS_AS_ERRORS=OFF\n"
             "cmake --build build --parallel\n"
             "ctest --test-dir build --output-on-failure", "powershell"),
        h2("Zero-burden build (no LLVM, no OpenMP)"),
        p("VGRE does not need LLVM to execute kernels — a from-scratch CUDA-C front-end "
          "feeds a four-tier CPU backend (PTX interpreter, compiled-fiber tier, native "
          "x86-64 JIT, and an optional SSA optimizing backend). Build with only a C++17 "
          "compiler:"),
        code("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \\\n"
             "    -DVGRE_ENABLE_JIT=OFF -DVGRE_ENABLE_OPENMP=OFF\n"
             "cmake --build build -j$(nproc)\n"
             "ctest --test-dir build -j$(nproc)     # LLVM-free suite: 379/379"),
        h2("Build the Python wheel"),
        code("bash bindings/python/build_wheel.sh build   # → bindings/python/dist/*.whl\n"
             "pip install bindings/python/dist/vgre-*.whl"),
        h2("System requirements"),
        table(["Component", "Minimum", "Recommended"], [
            ["CPU", "x86-64 (SSE4) or ARM64", "AVX2 / AVX-512, 8+ cores"],
            ["RAM", "4 GB", "16 GB+ (large-model work)"],
            ["OS", "Linux (glibc 2.31+)", "Ubuntu 22.04 / 24.04"],
            ["Compiler", "any C++17 (LLVM-free) / Clang 18 + LLVM 18 (JIT)", "Clang 18 + LLVM 18"],
        ]),
        h2("Platform support"),
        table(["Platform", "Status", "Notes"], [
            ["Linux x86-64", "✅ Verified", "Full suite green in CI (399 LLVM / 379 LLVM-free); the required job."],
            ["Linux ARM64", "✅ Builds/tests", "SIMD via NEON."],
            ["macOS ARM64", "✅ CI-green", "Full ctest suite runs green on Apple Silicon; auto Homebrew llvm@18."],
            ["Windows x86-64", "✅ CI-green", "Full ctest suite green (clang-cl on windows-2022)."],
        ]),
    ])


def page_running_cuda():
    return "".join([
        h1("Running CUDA Applications", "How VGRE intercepts and executes CUDA on the CPU."),
        h2("Execution model"),
        p("VGRE parses each CUDA-C kernel with its <strong>own from-scratch front-end</strong> "
          "(no Clang required) and runs it on the fastest of four bit-exact CPU tiers that "
          "accepts it: a PTX interpreter, a compiled-closure tier with a cooperative fiber "
          "executor, a hand-emitted native x86-64 JIT, and an optional SSA optimizing backend. "
          "When LLVM dev libraries are present, an optional Clang → LLVM IR → ORC-JIT path "
          "(<code>-O3 -march=native</code>) is available as the high-performance tier. Kernel "
          "grids run on an in-tree work-stealing thread pool with SIMD vectorization (OpenMP is "
          "optional); managed memory is backed by an OS page-fault handler. See "
          "<a href=\"architecture.html\">Architecture</a> for the tier table."),
        h2("What works"),
        ul([
            "CUDA Runtime API (~101 functions): memory, streams, events, graphs, textures/surfaces, cooperative launch, CDP.",
            "CUDA Driver API (~56 functions): contexts, modules, PTX load + JIT link.",
            "cuBLAS (L1/L2/L3, complex), cuDNN (conv/pool/norm/attention/RNN), cuFFT/cuRAND/cuSOLVER/cuSPARSE.",
            "NCCL collectives (~95%) and OpenCL 1.2.",
            "PTX ISA (~95% of common instructions), kernel fusion, persistent JIT cache.",
        ]),
        h2("Intercept mode — run an existing CUDA binary unchanged"),
        p("Preload the drop-in <code>libvgre_cudart</code> so an app's <code>cuda*</code> calls "
          "route into VGRE. No recompilation, no source changes:"),
        code("# Linux\n"
             "export LD_PRELOAD=\"$HOME/.local/share/VGRE/lib/libvgre_cudart.so\"\n"
             "python my_pytorch_script.py\n\n"
             "# macOS\n"
             "export DYLD_INSERT_LIBRARIES=\"$HOME/.local/share/VGRE/lib/libvgre_cudart.dylib\"\n"
             "python my_pytorch_script.py"),
        callout(p("<strong>Windows:</strong> copy <code>%LOCALAPPDATA%\\VGRE\\vgre_cudart.dll</code> "
                  "into the application directory and rename it <code>cudart64_120.dll</code> (match "
                  "your CUDA version suffix)."), "tip"),
        h2("Python bindings"),
        p("Allocate, copy, compile a kernel from source, launch, and read back — all on the CPU:"),
        code("import vgre\n"
             "import numpy as np\n\n"
             "rt = vgre.Runtime()\n"
             "rt.init(enable_profiling=True)\n\n"
             "a = rt.alloc(1024 * np.float32().itemsize)\n"
             "rt.memcpy_h2d(a, np.ones(1024, dtype=np.float32))\n\n"
             "kernel = vgre.Kernel(\"vector_scale\", \"\"\"\n"
             "    __global__ void vector_scale(float* a, float scale, int n) {\n"
             "        int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "        if (i < n) a[i] *= scale;\n"
             "    }\n"
             "\"\"\")\n"
             "rt.launch(kernel, grid=(4, 1, 1), block=(256, 1, 1), args=[a, 2.0, 1024])\n"
             "rt.synchronize()\n"
             "print(np.frombuffer(rt.memcpy_d2h(a, 1024 * 4), dtype=np.float32)[:8])", "python"),
        h2("C API"),
        code("#include <vgre/api/vgre_c_api.h>\n\n"
             "int main(void) {\n"
             "    vgre_init();\n"
             "    void* ptr = NULL;\n"
             "    vgre_malloc(&ptr, 1024 * sizeof(float));\n"
             "    // ... launch kernels ...\n"
             "    vgre_free(ptr);\n"
             "    vgre_shutdown();\n"
             "}", "c"),
        h2("Performance expectations"),
        p("CPU execution is typically 10–50× slower than a real GPU for compute-bound "
          "kernels; AVX-512 + OpenMP narrow the gap substantially for memory-bound work. "
          "Call <code>getMemoryBandwidthStats()</code> to measure your effective "
          "bandwidth and the estimated GPU speedup factor for your specific workload."),
        callout(p("Best fit: learning CUDA, CI/CD test runs, moderate vector/matrix/graph "
                  "workloads, and distributed CPU-cluster compute. Not a replacement for a "
                  "GPU on latency-critical, heavy-FLOP production inference."), "tip"),
    ])


def page_cluster():
    return "".join([
        h1("Distributed Clusters",
           "Scale kernels across machines over LAN or WAN. Channels are authenticated with "
           "HMAC-SHA256, encrypted with AES-256-CTR, and compressed with LZ4."),
        h2("vgre-start commands"),
        code("vgre-start --master                              # start a master node (launches dashboard)\n"
             "vgre-start --worker                              # start a worker (LAN auto-discovery)\n"
             "vgre-start --worker --is-master                  # headless master (K8s / container mode)\n"
             "vgre-start --worker --master-ip <IP>             # LAN: connect to a specific master IP\n"
             "vgre-start --worker --master-address <HOST:PORT> # WAN: hostname, IPv4, or IPv6\n"
             "vgre-start --test                                # local self-test (master + worker)\n"
             "vgre-start --port <N>                            # custom TCP port (default 7777)\n"
             "vgre-start --threads <N>                         # worker thread count (default: auto)"),

        h2("LAN cluster — step by step"),
        p("<strong>1. Install</strong> on every node "
          "(<a href=\"installation.html\">Installation</a>). "
          "<strong>2. Generate</strong> the token on the master and "
          "<strong>3. distribute</strong> it to workers "
          "(<a href=\"token-management.html\">Token Management</a>). "
          "<strong>4. Start</strong> the services:"),
        code("# Master:\n"
             "vgre-start --master\n\n"
             "# Workers (LAN auto-discovery):\n"
             "vgre-start --worker\n\n"
             "# Workers (explicit LAN IP):\n"
             "vgre-start --worker --master-ip 192.168.1.10"),

        h2("WAN cluster"),
        p("For workers on a <strong>different network</strong>, UDP broadcast cannot cross routers — "
          "use a direct TCP connection. Three methods:"),
        h3("Method A — explicit IP"),
        code("# On master: find your public IP\n"
             "vgre-discover        # -> Public IP: 78.45.12.99  (prints the worker command)\n\n"
             "# On worker (any network):\n"
             "vgre-start --worker --master-address 78.45.12.99:7777"),
        h3("Method B — automated token-keyed discovery"),
        code("vgre-discover --register     # on master (run once, or after the IP changes)\n"
             "#   -> Bucket ID: aBcDeFgH   (share this with workers)\n"
             "vgre-discover --find aBcDeFgH  # on each worker (same token required)\n"
             "vgre-start --worker --master-address 78.45.12.99:7777"),
        p("See <a href=\"discovery.html\">Public IP Discovery</a> for how the token-keyed KV store works."),
        h3("Method C — VPN overlay (Tailscale / ZeroTier)"),
        code("# Both machines on the same Tailscale network; master's Tailscale IP 100.64.0.1\n"
             "vgre-start --worker --master-ip 100.64.0.1"),

        h2("Firewall rules"),
        code("# Linux (UFW)\n"
             "sudo ufw allow 7777/tcp     # TCP cluster\n"
             "sudo ufw allow 7778/udp     # UDP discovery (LAN only)\n\n"
             "# Windows PowerShell (admin)\n"
             "New-NetFirewallRule -DisplayName \"VGRE TCP\" -Direction Inbound -Protocol TCP -LocalPort 7777 -Action Allow"),

        h2("Automatic reconnection"),
        p("Workers reconnect after a master restart or a network drop using exponential backoff:"),
        table(["State", "Behavior"], [
            ["Connection drops", "Retry immediately, then 1 s → 2 → 4 → … → 120 s"],
            ["Master restarts", "Worker reconnects within one backoff cycle"],
            ["IP changes", "Re-run <code>vgre-discover --register</code> on master; workers re-run <code>--find</code>"],
        ]),
        p("The backoff ceiling is configurable: <code>export VGRE_CLUSTER_MAX_BACKOFF_SEC=60</code>."),

        h2("Dashboard disconnect detection"),
        p("The dashboard detects worker disconnections automatically (TCP keepalive, ~12 s after a drop): "
          "the node card turns <strong>red</strong> with a <code>DISCONNECTED</code> countdown, a "
          "floating notification appears, and the node is removed from the topology after 30 s — the "
          "3D canvas updates in real time."),

        h2("Cross-platform support"),
        p("The wire protocol is OS-agnostic (the <code>CrossPlatform*</code> socket/error/memory tests "
          "pass everywhere). Any OS can be master or worker:"),
        table(["Master", "Worker", "Status"], [
            ["Linux", "Linux / macOS / Windows", "Fully supported (Linux CI-verified)"],
            ["macOS", "Linux / Windows", "Fully supported (macOS CI-green)"],
            ["Windows", "Linux / macOS", "Fully supported (Windows CI-green, full suite)"],
        ]),
    ])


def page_dashboard():
    return "".join([
        h1("The Dashboard", "A real-time Flutter monitor for the running engine."),
        p("Launch with <code>vgre-dashboard</code>. It loads the native library "
          "in-process and polls live telemetry — no mock data. Pages:"),
        table(["Page", "What it shows"], [
            ["Dashboard", "Live utilization gauges, workload throughput, UVM activity, log stream."],
            ["Kernel Explorer", "JIT-compiled kernels, launch counts, occupancy, per-kernel timing."],
            ["Cluster Topology", "Connected nodes, per-node load, network heatmap."],
            ["Hardware Tuning", "Toggle background compute, block-threads, profiler; set cache model."],
            ["Memory Analysis", "Allocation map, bandwidth stats, UVM page residency."],
            ["Settings", "Polling interval, database retention, connection config."],
        ]),
        h2("Documentation links"),
        p("The dashboard's <strong>Help &amp; Docs</strong> section opens this site "
          "directly in your browser — the Documentation, Quick Start, and API "
          "Reference items are wired to the pages here."),
        callout(p("The connection badge distinguishes <strong>LIVE</strong>, "
                  "<strong>DELAYED</strong>, and <strong>STALE</strong> so the UI never "
                  "presents stale telemetry as current."), "tip"),
    ])


def page_local_ai():
    return "".join([
        h1("Local AI: LoRA · RAG · LM",
           "A complete offline machine-learning stack built into the runtime — no "
           "PyTorch, no FAISS, no external tokenizer library."),
        h2("LoRA fine-tuning"),
        p("<code>vgre.nn.LoRALinear</code> adds trainable low-rank adapters over a "
          "frozen base weight, so you can fine-tune on a CPU/laptop with a fraction of "
          "the parameters. Adapters merge back to the base for zero-overhead inference."),
        code("from vgre import nn\n"
             "layer = nn.LoRALinear.from_linear(pretrained, r=8, alpha=16)\n"
             "opt   = nn.AdamW(layer.adapter_parameters(), lr=1e-3)   # trains A,B only\n"
             "# ... training loop ...\n"
             "layer.merge()                 # fold adapter into W for fast inference\n"
             "layer.save_adapter('style.npz')   # kilobytes, not the whole model",
             "python"),
        h2("Vector search (HNSW)"),
        p("<code>vgre.vector.VectorIndex</code> is a from-scratch Malkov–Yashunin HNSW "
          "index — cosine or L2, binary save/load, recall@10 ≈ 0.99 vs brute force."),
        code("import numpy as np\n"
             "from vgre.vector import VectorIndex\n"
             "ix = VectorIndex(dim=64, metric='cosine')\n"
             "ix.add(embeddings, ids=np.arange(len(embeddings)))\n"
             "ids, dists = ix.search(query_vec, k=10)\n"
             "ix.save('docs.vgvx')",
             "python"),
        h2("Tokenizer + language model"),
        p("Load any modern Hugging Face <code>tokenizer.json</code> (GPT-2, Llama-3, "
          "Qwen, Phi, DeepSeek) and get that model's exact token ids — verified "
          "token-for-token against the real HF <code>tokenizers</code> library."),
        code("import vgre\n"
             "tok = vgre.Tokenizer().load_hf('tokenizer.json')\n"
             "ids = tok.encode('Hello, world!')\n"
             "assert tok.decode(ids) == 'Hello, world!'",
             "python"),
        callout(p("Together these complete a fully local <strong>embed → index → "
                  "retrieve → generate → fine-tune</strong> RAG loop that never touches "
                  "the network."), "tip"),
    ])


def page_debugging():
    return "".join([
        h1("Debugging PTX with GDB",
           "CUDA-GDB-style single-stepping through PTX kernels using a stock gdb — no "
           "GPU, no cuda-gdb install."),
        h2("How it works"),
        p("VGRE ships a PTX single-step interpreter and a GDB remote-serial-protocol "
          "server. Each CUDA thread appears as a GDB thread (a lane); the kernel's own "
          "virtual PTX registers are exposed to <code>info registers</code>; and "
          "breakpoints, <code>stepi</code>, and memory inspection all behave normally "
          "in an unmodified gdb."),
        h2("Start the stub"),
        code("vgre_ptx_gdbserver kernel.ptx saxpy 1 4 2000 \\\n"
             "    f32:2.0 buf:f32:4=1,2,3,4 buf:f32:4=10,20,30,40 u32:4\n"
             "# prints the instruction listing, buffer addresses, and:\n"
             "#   gdb remote stub listening on 127.0.0.1:2000"),
        h2("Attach a normal gdb"),
        code("gdb -ex 'target remote 127.0.0.1:2000'\n"
             "(gdb) info threads          # CUDA lanes of the running CTA\n"
             "(gdb) break *0x12           # breakpoint by instruction index\n"
             "(gdb) continue\n"
             "(gdb) info registers ptx_f4 rip\n"
             "(gdb) stepi\n"
             "(gdb) x/4fw 0x...           # read global memory"),
        callout(p("The interpreter covers the arithmetic/control/memory core nvcc emits "
                  "for compute kernels, with <code>bar.sync</code>-aware lane scheduling. "
                  "Verified end-to-end against a real gdb 15 client."), "tip"),
    ])


def page_env_vars():
    tbl = lambda rows: table(["Variable", "Default", "Description"], rows)
    core = tbl([
        ["<code>VGRE_LIB_PATH</code>", "auto", "Absolute path to <code>libvgre.so</code> / <code>vgre.dll</code>."],
        ["<code>LD_LIBRARY_PATH</code>", "extended", "Directory containing the VGRE shared libraries."],
        ["<code>VGRE_LOG_LEVEL</code>", "<code>INFO</code>", "Verbosity: <code>DEBUG</code> | <code>INFO</code> | <code>WARN</code> | <code>ERROR</code>."],
        ["<code>VGRE_INSTALL_DIR</code>", "<code>~/.local/share/VGRE</code>", "Installation directory."],
    ])
    backend = tbl([
        ["<code>VGRE_EXEC_BACKEND</code>", "auto",
         "Execution tier: <code>interp</code> (PTX interpreter), <code>cp</code> (compiled-fiber), "
         "or <code>ssa</code> (Tier-2 SSA). Unset = fastest tier that accepts the kernel."],
        ["<code>VGRE_DISABLE_NATIVE</code>", "<code>0</code>",
         "Set to <code>1</code> to skip the native x86-64 JIT and use the compiled tier."],
        ["<code>VGRE_SSA_ARM_NATIVE</code>", "on",
         "Native AArch64 codegen for the SSA backend is on by default; set to <code>0</code> to "
         "force the portable evaluator."],
        ["<code>VGRE_IPC_MODE</code>", "<code>OFF</code>",
         "MPS-style IPC multiplexing (<code>ON</code> shares one runtime across processes)."],
    ])
    k8s = tbl([
        ["<code>VGRE_DEVICE_PLUGIN_PATH</code>", "<code>/var/lib/kubelet/device-plugins/kubelet.sock</code>",
         "Override the kubelet socket for K8s device-plugin registration."],
    ])
    cluster = tbl([
        ["<code>VGRE_PORT</code>", "<code>7777</code>", "TCP port for master and worker nodes."],
        ["<code>VGRE_CLUSTER_MASTER_ADDRESS</code>", "—", "WAN direct connect: <code>IP:PORT</code>, hostname, or <code>[::1]:PORT</code>."],
        ["<code>VGRE_CLUSTER_ADVERTISED_ADDRESS</code>", "—", "Master public address embedded in UDP pings (NAT)."],
        ["<code>VGRE_CLUSTER_NODES</code>", "—", "Comma-separated worker <code>IP:PORT</code> list (master side)."],
        ["<code>VGRE_MESH_PEERS</code>", "—", "Comma-separated peer list for full-mesh topology."],
        ["<code>VGRE_CLUSTER_DISCOVERY</code>", "enabled", "Set to <code>OFF</code> for pure WAN / explicit-address mode."],
        ["<code>VGRE_TCP_AUTH_TOKEN_FILE</code>", "<code>~/.vgre/token</code>", "Path to the shared cluster auth token file."],
        ["<code>VGRE_TCP_AUTH_TOKEN</code>", "—", "Raw token string (prefer the file)."],
    ])
    wan = tbl([
        ["<code>VGRE_CLUSTER_CONNECT_TIMEOUT_SEC</code>", "<code>10</code>", "TCP connect timeout, seconds (1–120)."],
        ["<code>VGRE_CLUSTER_MAX_BACKOFF_SEC</code>", "<code>120</code>", "Maximum reconnection retry interval (10–3600)."],
        ["<code>VGRE_CLUSTER_IDLE_EVICT_SEC</code>", "<code>300</code>", "Silence before an idle connection is evicted (0 = off)."],
    ])
    disc = tbl([
        ["<code>VGRE_DISCOVERY_BUCKET_ID</code>", "auto", "kvdb.io bucket ID for cross-LAN token-keyed discovery."],
        ["<code>VGRE_CLUSTER_UDP_ANNOUNCE_PORT</code>", "<code>7778</code>", "UDP port the master broadcasts on (must match on all nodes)."],
        ["<code>VGRE_CLUSTER_UDP_WORKER_PORT</code>", "<code>7779</code>", "UDP port workers broadcast on (must match on all nodes)."],
    ])
    cache = tbl([
        ["<code>VGRE_L1_CACHE_KB</code>", "<code>32</code>", "Per-block L1 cache (16 | 32 | 64 | 128)."],
        ["<code>VGRE_L2_CACHE_MB</code>", "<code>6</code>", "Per-device L2 cache (2 | 6 | 20 | 40)."],
    ])
    perf = tbl([
        ["<code>VGRE_ENABLE_NUMA</code>", "<code>0</code>", "NUMA-aware thread scheduling (Linux)."],
        ["<code>VGRE_WORKER_THREADS</code>", "auto (nproc)", "Override the worker thread count."],
        ["<code>VGRE_SIMD_LEVEL</code>", "auto", "Force <code>SSE4</code> | <code>AVX</code> | <code>AVX2</code> | <code>AVX512</code>."],
    ])
    conf = tbl([
        ["<code>VGRE_CONFIG_FILE</code>", "—", "Path to a JSON or YAML config file; loads its variables when set."],
        ["<code>VGRE_CONFIG_HOT_RELOAD</code>", "<code>false</code>", "Watch the config file and hot-reload on change."],
        ["<code>VGRE_DEPLOYMENT_PROFILE</code>", "<code>development</code>", "Profile: <code>development</code> | <code>staging</code> | <code>production</code> | <code>custom</code>."],
    ])
    return "".join([
        h1("Environment Variables",
           "Written to <code>~/.vgre/env</code> by <code>install_local.sh</code> and loaded "
           "automatically. On Windows they are set in User scope by <code>vgre_sync.bat</code> / "
           "<code>vgre_env.ps1</code>."),
        h2("Core"), core,
        h2("Execution backend"), backend,
        h2("Cluster / networking"), cluster,
        h2("WAN / connection tuning"), wan,
        h2("Discovery"), disc,
        h2("GPU cache model"), cache,
        h2("Performance tuning"), perf,
        h2("Configuration management"), conf,
        h2("Kubernetes / container"), k8s,
        callout(p("Every variable is optional — unset ones fall back to the defaults above."), "tip"),
    ])


def page_api():
    return "".join([
        h1("API Reference", "Python, C ABI, and the CUDA-runtime shim."),
        h2("Python bindings"),
        p("The <code>vgre</code> package wraps the native library via ctypes:"),
        table(["Symbol", "Purpose"], [
            ["<code>vgre.Tokenizer</code>", "BPE tokenizer; <code>train</code>, <code>load_hf</code>, <code>encode</code>, <code>decode</code>."],
            ["<code>vgre.LanguageModel</code>", "In-tree GPT; <code>train</code>, <code>generate</code>, <code>save</code>, <code>load</code>."],
            ["<code>vgre.nn.LoRALinear</code>", "Low-rank adapters; <code>adapter_parameters</code>, <code>merge</code>."],
            ["<code>vgre.vector.VectorIndex</code>", "HNSW ANN; <code>add</code>, <code>search</code>, <code>save</code>, <code>load</code>."],
        ]),
        h2("C ABI (vgre_c_api.h)"),
        p("The stable C surface the wheel and dashboard bind to. Highlights:"),
        code("vgre_init();\n"
             "vgre_set_profiler_enabled(1);\n"
             "vgre_set_block_threads(0);\n"
             "vgre_get_memory_bandwidth_stats(&stats);\n"
             "// tokenizer\n"
             "vgre_bpe* t = vgre_bpe_create();\n"
             "vgre_bpe_load_hf(t, \"tokenizer.json\");\n"
             "int n = vgre_bpe_encode(t, \"hi\", ids, 256);",
             "cpp"),
        h2("CUDA runtime shim (libvgre_cudart)"),
        p("Drop-in <code>libcudart</code> replacement covering ~95% of the common CUDA "
          "Runtime + Driver API (memory, streams, events, graphs, textures, cooperative "
          "launch) plus the cuBLAS / cuDNN / cuFFT / cuRAND / cuSOLVER / cuSPARSE / NCCL "
          "shims. Preload it (Linux <code>LD_PRELOAD</code>, macOS "
          "<code>DYLD_INSERT_LIBRARIES</code>) to route an existing CUDA binary's "
          "<code>cuda*</code> calls into VGRE — no source changes."),
        h2("Dashboard telemetry"),
        p("The dashboard polls a JSON telemetry schema; the top-level groups are:"),
        table(["Group", "Fields"], [
            ["<code>utilization</code>", "sm / memory / pcie percentages, sampled over time"],
            ["<code>kernels</code>", "name, grid/block dims, launch count, total &amp; average time, tier"],
            ["<code>memory</code>", "allocations, pools, free/used bytes, bandwidth &amp; est. GPU speedup"],
            ["<code>cluster</code>", "per-node id, OS/arch/hostname, in-flight &amp; cumulative kernels, secure-channel state"],
            ["<code>logs</code>", "the last N log messages at the active <code>VGRE_LOG_LEVEL</code>"],
        ]),
    ])


def page_cli():
    return "".join([
        h1("CLI Tools", "Every command installed with VGRE, what it does, and how to use it."),
        p("<code>install_local.sh</code> links the runtime tools into <code>~/.local/bin</code> "
          "(add it to <code>PATH</code> — the installer does this for you). On Windows the same "
          "commands ship as <code>.bat</code> / <code>.ps1</code> scripts (see the note at the "
          "bottom). Every command below prints usage with <code>--help</code>."),

        h2("Cluster &amp; runtime"),
        table(["Command", "What it does", "Common usage"], [
            ["<code>vgre-start</code>", "Start a master or worker node (auto-sources <code>~/.vgre/env</code>).",
             "<code>vgre-start --master</code> · <code>--worker</code> · <code>--worker --master-ip 10.0.0.5</code> · <code>--worker --port 7778</code> · <code>--test</code> (master+worker on one box)"],
            ["<code>vgre-worker</code>", "The worker binary itself (usually launched by <code>vgre-start</code>).",
             "<code>vgre-worker --version</code> (prints build-info JSON) · <code>--is-master</code> (headless master in containers)"],
            ["<code>vgre-dashboard</code>", "Launch the real-time Flutter monitor (utilization, kernels, cluster, memory, logs).",
             "<code>vgre-dashboard</code>"],
            ["<code>vgre-token</code>", "Manage the shared cluster auth token (HMAC-SHA256 / AES-256-CTR channels).",
             "<code>vgre-token generate</code> · <code>fingerprint</code> · <code>set &lt;TOKEN&gt;</code> · <code>copy</code> · <code>push user@HOST</code>"],
            ["<code>vgre-discover</code>", "Public-IP discovery + token-keyed cross-LAN registration (find a master over WAN).",
             "<code>vgre-discover</code> (show public IP) · <code>--set-master</code> · <code>--register</code> · <code>--find</code> · <code>--unregister</code>"],
            ["<code>vgre-connect-check</code>", "Verify WAN/LAN connectivity to a master before starting a worker.",
             "<code>vgre-connect-check &lt;master-ip&gt; [port]</code>"],
        ]),

        h2("Cluster quick recipes"),
        code("# One machine — spin up a master + worker and self-test:\n"
             "vgre-start --test\n\n"
             "# Master node:\n"
             "vgre-token generate           # create the shared token (prints its fingerprint)\n"
             "vgre-token copy               # prints the scp command to share it with workers\n"
             "vgre-start --master           # starts the master + dashboard\n\n"
             "# Worker node (after receiving the token):\n"
             "vgre-token set <TOKEN>        # or: vgre-token push user@worker  (from the master)\n"
             "vgre-connect-check 10.0.0.5   # confirm reachability first\n"
             "vgre-start --worker --master-ip 10.0.0.5"),

        h2("Developer tools"),
        p("Built binaries land in <code>build/tools/</code> (run them from there, or add it to "
          "<code>PATH</code>):"),
        table(["Command", "What it does", "Common usage"], [
            ["<code>vgre_ptx_gdbserver</code>", "Serve a PTX kernel to a stock <code>gdb</code> for source-level step debugging (CUDA-GDB-style, no GPU).",
             "<code>vgre_ptx_gdbserver kernel.ptx</code> then <code>gdb</code> → <code>target remote :1234</code> (see <a href=\"debugging.html\">Debugging PTX</a>)"],
            ["<code>gpt2_infer</code>", "Real GPT-2 inference from a safetensors / GGUF checkpoint (matches Hugging Face bit-for-bit).",
             "<code>gpt2_infer model.safetensors --gen \"Hello\"</code>"],
        ]),

        h2("Setup helpers"),
        table(["Script", "What it does"], [
            ["<code>install_local.sh</code>", "One-command Linux/macOS install: deps, build, <code>~/.vgre/env</code>, auth token, CLI symlinks."],
            ["<code>scripts/vgre-cli-install.sh</code>", "(Re)install the CLI symlinks and ensure <code>~/.local/bin</code> is on <code>PATH</code>."],
            ["<code>scripts/vgre_sync.sh</code>", "Refresh <code>~/.vgre/env</code> and re-link the CLIs after a rebuild."],
            ["<code>scripts/vgre-mac-worker-setup.sh</code>", "macOS worker bring-up (launchd, DYLD paths)."],
            ["<code>scripts/vgre-print-linux-setup.sh</code>", "Linux desktop-launcher / print setup for the dashboard."],
        ]),

        callout(p("<strong>Windows:</strong> the same commands ship as scripts under "
                  "<code>scripts\\</code> — <code>vgre-start.bat</code>, <code>vgre-token.bat</code> / "
                  "<code>vgre-token.ps1</code>, <code>vgre-discover.bat</code> / <code>.ps1</code>, plus "
                  "PowerShell installers <code>Install-VGRETools.ps1</code>, "
                  "<code>Setup-VGRECluster.ps1</code>, and <code>Start-VGRE.ps1</code>. Env vars are set in "
                  "User scope by <code>vgre_sync.bat</code> / <code>vgre_env.ps1</code>."), "tip"),
    ])


def page_architecture():
    return "".join([
        h1("Architecture", "How the pieces fit together."),
        h2("Layers"),
        ol([
            "<strong>Interception</strong> — CUDA/OpenCL API shims capture calls at load time (a drop-in <code>libcudart</code> preload, or the C ABI / Python bindings).",
            "<strong>Front-end</strong> — a from-scratch CUDA-C lexer/parser lowers kernels to an AST (no Clang required); an optional Clang/LLVM path is used when LLVM dev libs are present.",
            "<strong>Execution (four CPU tiers, fastest-first, bit-exact)</strong> — see the table below.",
            "<strong>Memory</strong> — UVM via an OS page-fault handler; stream-ordered pools; LZ4-compressed cluster transfers.",
            "<strong>Distribution</strong> — authenticated, encrypted TCP transport; NCCL-style + software-RDMA collectives; tensor/pipeline parallelism and a GSPMD auto-partitioner.",
            "<strong>ML stack</strong> — autograd, SIMD BLAS GEMM, tokenizer, samplers, LoRA, HNSW, checkpoint loaders (safetensors / GGUF).",
        ]),
        h2("The four execution tiers"),
        p("A parsed kernel runs on the fastest tier that accepts it; anything outside a "
          "tier's subset falls through to the next. All tiers are held bit-exact against "
          "one another by differential fuzzing."),
        table(["Tier", "What it is", "Notes"], [
            ["0 — PTX interpreter", "Interprets PTX directly", "The guaranteed fallback; always available, no toolchain."],
            ["1 — Compiled closures + fibers", "AST → bound C++ closures with a cooperative stackful-fiber executor",
             "Runs the whole cooperative surface: <code>__shared__</code>, <code>__syncthreads</code>, warp shuffle/vote/reduce/match, <code>__syncwarp</code>, <code>__activemask</code>."],
            ["1b — Native x86-64 JIT", "Hand-emitted machine code (no LLVM)", "~18–118× the compiled tier; differential-fuzzed bit-exact; Linux/x86-64."],
            ["2 — Own SSA backend", "VGRE-IR → const-fold/GVN/LICM/DCE → linear-scan regalloc → x86-64 + AArch64 (native by default)",
             "<code>VGRE_EXEC_BACKEND=ssa</code>. Feature-complete for the scalar + shared-memory + warp subset, native on x86-64."],
        ]),
        p("An optional <strong>LLVM ORC JIT</strong> (Clang AST → LLVM IR → native, persistent "
          "disk+memory cache) is the high-performance path when LLVM is present; the engine "
          "needs none of it to run kernels."),
        h2("Design principles"),
        ul([
            "Real implementations only — no stubs, mocks, or placeholder math in shipped paths.",
            "From-scratch primitives (CUDA-C front-end, four execution tiers, tokenizer, vector index, crypto, collectives) to avoid vendor lock-in.",
            "Warnings-as-errors, and a full test suite gating every change: <strong>399 tests with LLVM, 379 in the LLVM-free build</strong>, green on Linux / macOS / Windows in CI.",
        ]),
    ])


def page_faq():
    return "".join([
        h1("Frequently Asked Questions"),
        h3("Does VGRE need a GPU?"),
        p("No. That is the point — kernels execute on the CPU. A GPU is never required."),
        h3("Is it as fast as a real GPU?"),
        p("No. Expect 10–50× slower for compute-bound kernels; memory-bound kernels are "
          "closer with AVX-512. VGRE targets learning, CI, development, and moderate "
          "workloads — not latency-critical production inference."),
        h3("Which platforms are supported?"),
        p("<strong>All three are CI-green.</strong> The full <code>ctest</code> suite passes on "
          "<strong>Linux x86-64</strong> (399 tests with LLVM, 379 in the LLVM-free build — the "
          "required CI job), on <strong>macOS</strong> (Apple Silicon, auto-detected Homebrew "
          "<code>llvm@18</code>), and on <strong>Windows</strong> (clang-cl on windows-2022). Linux "
          "ARM64 builds and tests via NEON. Prebuilt wheels ship for Linux, macOS (arm64) and "
          "Windows — see <a href=\"downloads.html\">Downloads</a>."),
        h3("macOS: OpenMP or JIT failures?"),
        ul([
            "Install <code>brew install llvm@18 libomp</code> and reconfigure: "
            "<code>rm -f build/CMakeCache.txt && cmake -B build</code>.",
            "Ensure Clang is discoverable: <code>export PATH=\"$(brew --prefix llvm@18)/bin:$PATH\"</code> "
            "or <code>export VGRE_CLANG_PATH=$(brew --prefix llvm@18)/bin/clang++</code>.",
            "Run tests serially: <code>ctest -j1 --timeout 300</code>.",
        ]),
        h3("The dashboard won't start / can't find the library."),
        ul([
            "Rebuild and reinstall, then open a <strong>new</strong> terminal so <code>~/.vgre/env</code> is loaded.",
            "Check library path: Linux <code>LD_LIBRARY_PATH</code>, macOS <code>DYLD_LIBRARY_PATH</code> — both set in <code>~/.vgre/env</code> after install.",
            "Check <code>VGRE_LIB_PATH</code> points to a real <code>libvgre.so</code> / <code>libvgre.dylib</code> / <code>vgre.dll</code>.",
            "Windows: ensure <code>%LOCALAPPDATA%\\VGRE</code> and <code>...\\VGRE\\lib</code> are on <code>PATH</code>.",
        ]),
        h3("A CUDA call returns an error."),
        p("VGRE implements ~95% of common CUDA APIs and returns a clear error for the "
          "rest rather than misbehaving. File an issue with the failing call and we can "
          "prioritize it."),
    ])


def page_system_requirements():
    return "".join([
        h1("System Requirements",
           "What VGRE needs to build and run — and the optional libraries that unlock extra features."),
        h2("All platforms"),
        table(["Requirement", "Minimum", "Recommended"], [
            ["Operating system", "Linux 5.4 / Windows 10 22H2 / macOS 11", "Ubuntu 22.04 / Windows 11 / macOS 14"],
            ["CPU", "x86-64, 2 cores", "x86-64 with AVX-512 or ARM64, 8+ cores"],
            ["RAM", "4 GB", "16 GB+"],
            ["CMake", "3.18", "3.25+"],
            ["LLVM / Clang", "16 <em>(optional — only for the JIT build)</em>", "18"],
            ["C++ standard", "C++17", "C++20"],
            ["Flutter SDK", "3.16 <em>(dashboard only)</em>", "3.24+"],
            ["PowerShell", "5.1 (Windows built-in)", "7+"],
        ]),
        callout(p("LLVM is <strong>optional</strong>. The zero-burden build "
                  "(<code>-DVGRE_ENABLE_JIT=OFF</code>) needs nothing but a C++17 compiler and "
                  "runs the full suite on the from-scratch four-tier CPU backend — see "
                  "<a href=\"installation.html\">Installation</a>."), "tip"),
        h2("Optional libraries — auto-detected"),
        p("These activate silently when present at configure time; the build never fails for "
          "a missing optional dependency, it downgrades gracefully with a CMake warning."),
        table(["Package", "Feature unlocked", "Detection"], [
            ["<code>libomp-dev</code>", "Multi-threaded kernel execution", "Recommended"],
            ["<code>libssl-dev</code>", "Secure cluster transport (TLS)", "Recommended"],
            ["<code>libtss2-dev</code>", "Hardware TPM 2.0 token storage (Linux)", "<strong>Auto</strong>"],
            ["<code>libibverbs-dev</code>", "RDMA / RoCE zero-copy transport", "<strong>Auto</strong>"],
            ["<code>libsecret-1-dev</code>", "GNOME Keyring token storage (Linux)", "<strong>Auto</strong>"],
            ["<code>libgrpc++-dev</code>", "gRPC cluster transport", "Manual (<code>-DVGRE_ENABLE_GRPC=ON</code>)"],
            ["Intel AMX CPU", "AMX tile acceleration", "<strong>Auto</strong> (CPUID at build time)"],
        ]),
    ])


def page_token_management():
    return "".join([
        h1("Token Management",
           "<code>vgre-token</code> manages the shared 256-bit cluster auth token — the secret "
           "that authenticates every node. It works from any directory on all three platforms."),
        h2("Commands"),
        code("vgre-token generate       # generate a new secure 256-bit token\n"
             "vgre-token show           # print the stored token value\n"
             "vgre-token fingerprint    # print the SHA-256 fingerprint\n"
             "vgre-token set [TOKEN]    # store a token pasted from another machine\n"
             "vgre-token verify         # check the env var matches the stored file\n"
             "vgre-token copy           # print the scp / robocopy command for workers\n"
             "vgre-token revoke         # delete the stored token (with confirmation)\n"
             "vgre-token install        # install vgre-token to the User PATH (first run)"),
        h2("Typical workflow"),
        code("# ── On the MASTER machine ─────────────────────────────────────\n"
             "vgre-token generate\n"
             "#   Token saved to ~/.vgre/token\n"
             "#   Token fingerprint (SHA-256): a3f9bcd2...\n\n"
             "# Share the token with workers (pick one):\n"
             "vgre-token copy                      # prints the scp / robocopy command\n"
             "scp ~/.vgre/token worker@192.168.1.50:~/.vgre/token\n\n"
             "# ── On every WORKER machine ───────────────────────────────────\n"
             "vgre-token set a3f9bcd2xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # option B: paste the value\n\n"
             "# ── Verify both machines match ────────────────────────────────\n"
             "vgre-token fingerprint               # run on BOTH — the output must be identical"),
        h2("Token storage"),
        table(["Platform", "Token file", "Env var set in"], [
            ["Linux / macOS", "<code>~/.vgre/token</code> (chmod 600)", "<code>~/.vgre/env</code> → shell profile"],
            ["Windows", "<code>%USERPROFILE%\\.vgre\\token</code>", "User-scope env (no restart needed)"],
        ]),
        callout(p("The token never travels over the network — only its HMAC-SHA256 challenge does. "
                  "Distribute the file over a secure channel (<code>scp</code>) or paste the value "
                  "directly; the fingerprint lets you confirm a match without revealing the secret."), "tip"),
    ])


def page_discovery():
    return "".join([
        h1("Public IP Discovery",
           "<code>vgre-discover</code> solves the WAN bootstrap problem — finding the master's "
           "real public IP and handing it to workers without manual configuration."),
        h2("Commands"),
        code("vgre-discover                  # show the real public IP + worker connect command\n"
             "vgre-discover --set-master     # set VGRE_CLUSTER_ADVERTISED_ADDRESS for this session\n"
             "vgre-discover --register       # register the master IP to a token-keyed KV store\n"
             "vgre-discover --find [BUCKET]  # (worker) retrieve the master IP using the local token\n"
             "vgre-discover --unregister     # remove the registration from the KV store"),
        h2("How it works"),
        p("<strong>Public IP detection</strong> tries five HTTP endpoints in order and uses the "
          "first valid IPv4 response: <code>api.ipify.org</code>, <code>ipecho.net/plain</code>, "
          "<code>icanhazip.com</code>, <code>checkip.amazonaws.com</code>, <code>api4.my-ip.io/ip</code>."),
        p("<strong>Master auto-broadcast</strong> — <code>vgre-start --master</code> calls "
          "<code>vgre-discover --set-master</code> first, setting "
          "<code>VGRE_CLUSTER_ADVERTISED_ADDRESS=&lt;public-ip&gt;:7777</code>. The UDP announcer "
          "embeds that real address in every broadcast so workers behind NAT receive the correct "
          "IP instead of the sender's internal one."),
        p("<strong>Cross-LAN token-keyed discovery</strong> derives the KV-store key from the token:"),
        code("key = \"vgre-\" + first_32_hex_chars_of( SHA256(token) )", "text"),
        p("Only nodes holding the same token can compute or look up the same key. The bucket ID is "
          "a non-secret opaque identifier shared once with workers — like a meeting-room ID, not a "
          "password."),
        h2("Quick start: WAN cluster with automated discovery"),
        code("# ── Master ────────────────────────────────────────────────────\n"
             "vgre-token generate            # if not already done\n"
             "vgre-discover --register       # prints the BUCKET_ID to share\n"
             "#   [OK] Master registered.  Bucket ID: aBcDeFgH   Master IP: 78.45.12.99:7777\n"
             "vgre-start --master            # auto-detects IP and sets the advertised address\n\n"
             "# ── Workers (any network, same token) ─────────────────────────\n"
             "vgre-discover --find aBcDeFgH  # -> Master found: 78.45.12.99:7777\n"
             "vgre-start --worker --master-address 78.45.12.99:7777"),
        h2("Discovery modes: LAN vs WAN"),
        table(["Scenario", "How discovery works"], [
            ["Same LAN", "UDP broadcast — automatic, no configuration"],
            ["Different LAN, same token", "<code>vgre-discover --register</code> → <code>--find</code>"],
            ["Different LAN, manual", "<code>vgre-discover</code> shows the IP → share the one-liner"],
            ["VPN (Tailscale / ZeroTier)", "Same as LAN — auto-discovery works"],
            ["Dynamic IP (home ISP)", "Re-run <code>vgre-discover --register</code> after the IP changes"],
        ]),
        callout(p("<code>vgre-discover</code> finds your <strong>external</strong> IP, but workers can "
                  "only reach the master if your router port-forwards TCP 7777, or you use a VPN "
                  "(<a href=\"https://tailscale.com\">Tailscale</a> / "
                  "<a href=\"https://www.zerotier.com\">ZeroTier</a> — no port-forward needed), or the "
                  "master runs on a VPS / cloud VM with a direct public IP."), "warn"),
    ])


def page_advanced():
    return "".join([
        h1("Advanced Features",
           "Cache modelling, SIMD/AMX acceleration, Hopper PTX emulation, MPS, and the optional "
           "RDMA / gRPC transports."),
        h2("GPU L1 / L2 cache model"),
        code("export VGRE_L1_CACHE_KB=64    # 16 | 32 | 64 | 128\n"
             "export VGRE_L2_CACHE_MB=20    # 2 | 6 | 20 | 40"),
        code("vgre_cache_stats_t cs;\n"
             "vgre_get_cache_stats(&cs);\n"
             "printf(\"L2 hit rate: %.1f%%  hits=%llu  misses=%llu\\n\",\n"
             "       cs.l2_hit_rate * 100.0, cs.l2_hits, cs.l2_misses);", "c"),
        h2("AMX + AVX-512 WMMA acceleration"),
        ul([
            "<strong>AMX path</strong> — 16×16×16 BF16 matrix tiles (Sapphire Rapids and newer).",
            "<strong>AVX-512 path</strong> — N=16 tiles, ~16× over scalar.",
            "<strong>Scalar fallback</strong> — always available.",
        ]),
        code("VGRE_LOG_LEVEL=DEBUG vgre-dashboard 2>&1 | grep -E \"AMX|AVX-512|SIMD\""),
        h2("Hopper PTX emulation"),
        table(["PTX instruction", "CPU emulation"], [
            ["<code>wgmma.mma_async.*</code>", "Full M×N×K GEMM via <code>vgre_wgmma_*</code> (AVX-512 when N=256)"],
            ["<code>cp.async.bulk.tensor.*</code>", "Synchronous <code>memcpy</code>"],
            ["<code>wgmma.fence</code> / <code>wgmma.wait_group</code>", "<code>__atomic_thread_fence(SEQ_CST)</code>"],
            ["<code>mma.sync.aligned.m16n8k16</code>", "<code>vgre_mma_m16n8k16_f32_f16</code> scalar tile GEMM"],
        ]),
        h2("CUDA MPS (multi-process)"),
        code("export VGRE_MPS_PIPE=/tmp/vgre_mps.sock\n"
             "# Each client process connects automatically when VGRE_MPS_PIPE is set."),
        h2("RDMA / RoCE transport"),
        code("cmake -S . -B build -DVGRE_ENABLE_RDMA=ON\n"
             "# Requires: sudo apt-get install libibverbs-dev rdma-core\n\n"
             "# Soft-RoCE loopback for testing:\n"
             "sudo rdma link add rxe0 type rxe netdev eth0\n"
             "export VGRE_RDMA_DEVICE=rxe0"),
        h2("gRPC cluster transport"),
        code("cmake -S . -B build -DVGRE_ENABLE_GRPC=ON\n"
             "export VGRE_GRPC_PORT=50051\n"
             "vgre-start --master"),
    ])


def page_troubleshooting():
    return "".join([
        h1("Troubleshooting", "Fixes for the issues you are most likely to hit, by symptom."),

        h3("Dashboard fails to load (\"Failed to load native library\")"),
        code("source ~/.vgre/env\n"
             "ls -la ~/.local/share/VGRE/lib/libvgre.so\n"
             "ldd ~/.local/share/VGRE/lib/libvgre.so | grep \"not found\"\n"
             "sudo apt-get install libomp-dev        # if libomp.so is missing"),

        h3("vgre-start not recognised (Windows)"),
        code("Test-Path \"$env:LOCALAPPDATA\\VGRE\\scripts\\vgre-start.bat\"\n"
             "$env:PATH = \"$env:LOCALAPPDATA\\VGRE\\scripts;$env:PATH\"   # if PATH not yet updated\n"
             "vgre-start --help\n"
             ".\\scripts\\vgre_sync.bat                                    # reinstall", "powershell"),

        h3("Worker cannot connect to master"),
        code("vgre-token fingerprint                       # 1. run on BOTH — must be identical\n"
             "nc -zv MASTER_IP 7777                        # 2. test reachability (Linux/macOS)\n"
             "Test-NetConnection MASTER_IP -Port 7777      #    (Windows PowerShell)\n"
             "sudo ufw allow 7777/tcp                      # 3. open the firewall (Linux)\n"
             "vgre-discover                                # 4. WAN: verify the public IP\n"
             "export VGRE_CLUSTER_CONNECT_TIMEOUT_SEC=30   # 5. raise the timeout on slow links"),

        h3("UDP auto-discovery not working across subnets"),
        p("UDP broadcast (<code>255.255.255.255</code>) cannot cross routers. Use a direct "
          "connection or token-keyed discovery instead:"),
        code("vgre-start --worker --master-ip 192.168.1.10           # same LAN, different subnet\n"
             "vgre-start --worker --master-address 78.45.12.99:7777  # WAN / internet\n"
             "vgre-discover --register   # on master\n"
             "vgre-discover --find <ID>  # on worker"),

        h3("Mismatched token fingerprints"),
        code("vgre-token copy          # on master — prints the scp command\n"
             "scp ~/.vgre/token worker@WORKER_IP:~/.vgre/token\n"
             "vgre-token fingerprint   # verify on both machines"),

        h3("CMake cannot find LLVM"),
        code("# Linux\n"
             "sudo apt-get install llvm-18 llvm-18-dev clang-18\n"
             "export LLVM_DIR=$(llvm-config-18 --cmakedir)\n\n"
             "# macOS (CMake auto-detects; override only if needed)\n"
             "brew install llvm@18 libomp\n"
             "export LLVM_DIR=\"$(brew --prefix llvm@18)/lib/cmake/llvm\"\n\n"
             "# Windows\n"
             "winget install LLVM.LLVM\n"
             "set LLVM_DIR=C:\\Program Files\\LLVM\\lib\\cmake\\llvm"),
        callout(p("Or skip LLVM entirely: <code>cmake -S . -B build -DVGRE_ENABLE_JIT=OFF</code> "
                  "builds with just a C++17 compiler."), "tip"),

        h3("Flutter build fails (Windows)"),
        code(".\\scripts\\vgre_sync.bat            # re-run — it pre-caches engine artifacts\n\n"
             "flutter precache --windows         # or, manually:\n"
             "cd vgre_dashboard\n"
             "flutter pub get\n"
             "flutter build windows --release", "powershell"),

        h3("Low compute performance"),
        p("SIMD is auto-detected at configure time. Confirm which level was selected, or force one:"),
        code("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -i \"simd\\|avx\\|amx\"\n"
             "export VGRE_SIMD_LEVEL=AVX512        # force a level at runtime\n"
             "VGRE_LOG_LEVEL=DEBUG ./build/examples/matrix_multiply 2>&1 | grep \"SIMD\\|AVX\\|AMX\""),

        h3("OpenMP not working (all kernels run on one thread)"),
        code("ldd ~/.local/share/VGRE/lib/libvgre.so | grep omp\n"
             "sudo apt-get install libomp-dev        # Ubuntu/Debian\n"
             "sudo dnf install libomp-devel          # Fedora\n"
             "# macOS: brew install llvm@18 libomp && rm -f build/CMakeCache.txt && cmake -B build"),

        h3("Windows: DLL load failure (0xC000001D / error 1114)"),
        code("dumpbin /dependents $env:LOCALAPPDATA\\VGRE\\vgre.dll\n"
             "Get-ChildItem \"$env:LOCALAPPDATA\\VGRE\\lib\" -Filter \"*.dll\" | Select-Object Name\n"
             ".\\scripts\\vgre_sync.bat            # re-run to restore missing DLLs", "powershell"),

        h3("Windows: socket / port-bind failures (ACCESS_VIOLATION 0xC0000005)"),
        p("These are handled inside the runtime; if a custom integration hits them:"),
        ul([
            "<strong>WSAStartup lifecycle</strong> (WinSock 10093) — VGRE manages WSAStartup/"
            "WSACleanup with a thread-safe <code>WindowsSocketManager</code> (Meyers singletons + "
            "<code>std::call_once</code>); ensure no external library closes WinSock early.",
            "<strong>Struct packing</strong> — all cross-platform packet headers use "
            "<code>#pragma pack(push, 1)</code> / <code>__attribute__((packed))</code> for bit-"
            "identical layout across MSVC and GCC/Clang.",
            "<strong>Port conflicts</strong> (TIME_WAIT) — set <code>VGRE_PORT</code> to a free "
            "port or raise <code>VGRE_CLUSTER_CONNECT_TIMEOUT_SEC</code>.",
        ]),
    ])


def page_production_status():
    def rows(pairs):
        return [[f, s] for f, s in pairs]
    return "".join([
        h1("Production Status",
           "Every shipped capability and its maturity. \"Production\" means real CPU math with no "
           "runtime stubs, exercised by the CI test suite."),
        table(["Feature", "Status"], rows([
            ("CUDA runtime intercept (<code>cudart</code>)", "Production"),
            ("LLVM JIT kernel compilation", "Production"),
            ("From-scratch four-tier CPU backend (no LLVM)", "Production"),
            ("cuBLAS INT8 / FP32 / FP64", "Production"),
            ("cuDNN convolution (INT8, FP32)", "Production"),
            ("NCCL AllReduce (ring + barrier-tree)", "Production"),
            ("CUDA Graphs (IF / WHILE / SWITCH)", "Production"),
            ("Virtual memory (<code>cuMemCreate</code> / <code>cuMemMap</code>)", "Production"),
            ("External semaphores (eventfd / Win32)", "Production"),
            ("UVM managed memory", "Production"),
            ("NVTX profiling markers", "Production"),
            ("AVX-512 WMMA acceleration", "Production"),
            ("Intel AMX tile acceleration", "Production (Sapphire Rapids+)"),
            ("Hopper PTX (wgmma, TMA, cp.async.bulk)", "Production"),
            ("GPU L1 / L2 cache model", "Production"),
            ("TCP cluster + TLS", "Production"),
            ("WAN cluster (hostname / IPv6 / NAT)", "<strong>Production</strong>"),
            ("Worker auto-disconnect detection", "<strong>Production</strong>"),
            ("RDMA / RoCE transport", "Optional (<code>-DVGRE_ENABLE_RDMA=ON</code>)"),
            ("gRPC cluster transport", "Optional (<code>-DVGRE_ENABLE_GRPC=ON</code>)"),
            ("CUDA MPS multi-process", "Production (Unix socket)"),
            ("<code>cuMemMulticast</code>", "Production"),
            ("Flutter real-time dashboard", "Production"),
            ("<code>vgre-token</code> CLI", "Production (Linux / macOS / Windows)"),
            ("<code>vgre-start</code> cluster launcher", "Production (Linux / macOS / Windows)"),
            ("<code>vgre-discover</code> IP discovery", "<strong>Production</strong>"),
            ("OpenTelemetry <code>hw.gpu.*</code> export", "Production"),
        ])),
    ])


PAGES = {
    "index.html": page_index,
    "quickstart.html": page_quickstart,
    "downloads.html": page_downloads,
    "installation.html": page_installation,
    "system-requirements.html": page_system_requirements,
    "running-cuda.html": page_running_cuda,
    "cluster.html": page_cluster,
    "discovery.html": page_discovery,
    "token-management.html": page_token_management,
    "dashboard.html": page_dashboard,
    "local-ai.html": page_local_ai,
    "debugging.html": page_debugging,
    "advanced.html": page_advanced,
    "env-vars.html": page_env_vars,
    "api.html": page_api,
    "cli.html": page_cli,
    "production-status.html": page_production_status,
    "architecture.html": page_architecture,
    "troubleshooting.html": page_troubleshooting,
    "faq.html": page_faq,
}


# ── Shell rendering ───────────────────────────────────────────────────────────
def render_nav(active):
    out = []
    for group, items in NAV:
        links = "".join(
            f'<a class="nav-link{" active" if f == active else ""}" href="{f}">{html.escape(t)}</a>'
            for f, t in items
        )
        out.append(f'<div class="nav-group"><div class="nav-group-title">'
                    f'{html.escape(group)}</div>{links}</div>')
    return "".join(out)


def page_nav_footer(active):
    idx = [f for f, _ in FLAT].index(active)
    prev_html = next_html = ""
    if idx > 0:
        f, t = FLAT[idx - 1]
        prev_html = (f'<a href="{f}"><div class="dir">← Previous</div>'
                     f'<div class="ttl">{html.escape(t)}</div></a>')
    else:
        prev_html = "<span></span>"
    if idx < len(FLAT) - 1:
        f, t = FLAT[idx + 1]
        next_html = (f'<a class="next" href="{f}"><div class="dir">Next →</div>'
                     f'<div class="ttl">{html.escape(t)}</div></a>')
    else:
        next_html = "<span></span>"
    return f'<div class="page-nav">{prev_html}{next_html}</div>'


def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s)


def render(active, body):
    title = dict(FLAT).get(active, "VGRE Docs")
    shell = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · VGRE Documentation</title>
<meta name="description" content="VGRE — run CUDA on CPU, plus a full local-AI stack. {html.escape(strip_tags(title))}.">
<link rel="canonical" href="{SITE_URL}/{active}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Orbitron:wght@700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css?v={asset_ver('assets/style.css')}">
</head>
<body>
<header class="site-header">
  <button class="menu-toggle" aria-label="Menu">☰</button>
  <a class="brand" href="index.html">
    <span class="brand-mark">◈</span>
    <span class="brand-name">VGRE</span>
    <span class="brand-tag">VIRTUAL GPU RUNTIME</span>
  </a>
  <div class="header-spacer"></div>
  <input class="header-search" type="search" placeholder="Search docs…" aria-label="Search">
  <nav class="header-links">
    <a href="index.html">Docs</a>
    <a href="api.html">API</a>
    <a href="https://github.com/etienne0114/vgre">GitHub</a>
  </nav>
</header>
<div class="layout">
  <aside class="sidebar">{render_nav(active)}</aside>
  <main class="content">
    {body}
    {page_nav_footer(active)}
  </main>
  <aside class="toc">
    <div class="toc-title">On this page</div>
    <nav class="toc-list"></nav>
  </aside>
</div>
<script src="assets/search-index.js?v={asset_ver('assets/search-index.js')}"></script>
<script src="assets/docs.js?v={asset_ver('assets/docs.js')}"></script>
</body>
</html>
"""
    return shell


def build_search_index():
    index = []
    for f, title in FLAT:
        body = PAGES[f]()
        text = strip_tags(body)
        text = re.sub(r"\s+", " ", text).strip()
        # Split into H2 sections for better search granularity.
        index.append({"url": f, "title": title, "section": "Overview", "text": text[:600]})
    return index


def main():
    for f, fn in PAGES.items():
        html_out = render(f, fn())
        with open(os.path.join(HERE, f), "w", encoding="utf-8") as fh:
            fh.write(html_out)
    idx = build_search_index()
    with open(os.path.join(HERE, "assets", "search-index.js"), "w", encoding="utf-8") as fh:
        fh.write("window.VGRE_SEARCH_INDEX = " + json.dumps(idx, ensure_ascii=False) + ";\n")
    print(f"Generated {len(PAGES)} pages + search index in {HERE}")


if __name__ == "__main__":
    main()
