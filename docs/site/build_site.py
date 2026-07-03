#!/usr/bin/env python3
"""Generate the VGRE documentation website (static HTML) from the page content
defined below. No framework, no build dependency — emits plain files under
docs/site/ that open directly in a browser (file://) or from any static host.

    python3 docs/site/build_site.py

Content is authored here as lightweight markup (headings, paragraphs, code
blocks, tables, callouts, cards) and rendered into the shared shell (header,
left nav, right auto-TOC, search). Everything is real: commands, env vars, and
API signatures are taken verbatim from README.md / docs/USER_GUIDE.md /
docs/api_reference.md and the shipped tools.
"""
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Public site URL used for canonical links / the dashboard "Docs" button.
SITE_URL = "https://vgre.dev/docs"

# ── Navigation (also the page order for prev/next) ────────────────────────────
NAV = [
    ("Getting Started", [
        ("index.html", "Introduction"),
        ("quickstart.html", "Quick Start"),
        ("installation.html", "Installation"),
    ]),
    ("Guides", [
        ("running-cuda.html", "Running CUDA Apps"),
        ("cluster.html", "Distributed Clusters"),
        ("dashboard.html", "The Dashboard"),
        ("local-ai.html", "Local AI: LoRA · RAG · LM"),
        ("debugging.html", "Debugging PTX (GDB)"),
    ]),
    ("Reference", [
        ("env-vars.html", "Environment Variables"),
        ("api.html", "API Reference"),
        ("cli.html", "CLI Tools"),
    ]),
    ("About", [
        ("architecture.html", "Architecture"),
        ("faq.html", "FAQ & Troubleshooting"),
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


def code(body, lang="bash"):
    return f'<pre><code class="lang-{lang}">{html.escape(body.strip())}</code></pre>'


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
        badges([("Linux verified", True), ("macOS build-verified", True),
                ("CUDA ~95%", False), ("PTX ~95%", False), ("Zero external vendor lock-in", True)]),
        p("VGRE intercepts CUDA and OpenCL API calls and executes kernels on the CPU "
          "using an LLVM ORC JIT, OpenMP + AVX2/AVX-512 SIMD, OS-level unified memory, "
          "and an authenticated TCP cluster transport. The same engine ships an "
          "in-tree machine-learning stack so you can go from CUDA experiments to real "
          "model training and RAG without a second toolchain."),
        cards([
            ("🚀", "Quick Start", "Install and run your first CUDA program on CPU in five minutes.", "quickstart.html"),
            ("🧠", "Local AI", "LoRA fine-tuning, HNSW vector search, and the in-tree language model.", "local-ai.html"),
            ("🔗", "Clusters", "Scale across machines over WAN with encrypted, authenticated channels.", "cluster.html"),
            ("🐞", "PTX Debugging", "Step through PTX kernels with a stock gdb — CUDA-GDB-style, no GPU.", "debugging.html"),
            ("📊", "Dashboard", "Real-time telemetry, kernel explorer, cluster topology, memory analysis.", "dashboard.html"),
            ("📚", "API Reference", "Python, C ABI, CUDA-runtime shim, and dashboard telemetry APIs.", "api.html"),
        ]),
        h2("Why VGRE"),
        ul([
            "<strong>No GPU needed</strong> — learn CUDA, run CI, and develop on any x86-64 or ARM64 machine.",
            "<strong>One stack, both jobs</strong> — GPU emulation and a real ML training/inference/RAG stack in the same runtime.",
            "<strong>Built from scratch</strong> — the JIT, tokenizer, vector index, collectives, and crypto are in-tree, not third-party wrappers, so there is no vendor lock-in and no per-seat dependency cost.",
            "<strong>Private by default</strong> — the entire embed → index → retrieve → generate → fine-tune loop runs offline; your data never leaves the machine.",
        ]),
        callout(p("VGRE is <strong>Linux-verified</strong> (full <code>ctest</code> green). "
                  "<strong>macOS</strong> is build-verified on Apple Silicon — native engine, "
                  "JIT, and integration tests pass locally with auto-detected Homebrew "
                  "<code>llvm@18</code>. <strong>Windows</strong> is code-complete but not yet "
                  "CI-verified. Hardware-only gaps (Metal MPS, NVIDIA PMU) are in "
                  '<a href="faq.html">FAQ</a> / <code>missingFeatures.md</code>.'), "warn"),
    ])


def page_quickstart():
    return "".join([
        h1("Quick Start", "From clone to a running CUDA kernel on CPU in a few commands."),
        h2("Automated install (Linux / macOS)"),
        code("git clone https://github.com/vgre-org/vgre-runtime.git\n"
             "cd vgre-runtime\n"
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
        h2("Prerequisites"),
        code("# Ubuntu / Debian\n"
             "sudo apt install -y cmake ninja-build clang llvm-18-dev \\\n"
             "    libomp-dev libssl-dev libsqlite3-dev\n\n"
             "# macOS (Homebrew — paths discovered automatically at configure time)\n"
             "brew install cmake llvm@18 libomp ninja sqlite\n\n"
             "# Flutter (for the dashboard) — one of:\n"
             "sudo snap install flutter --classic\n"
             "# or: brew install --cask flutter"),
        h2("Build"),
        code("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release\n"
             "cmake --build build -j$(nproc)\n"
             "ctest --test-dir build -j$(nproc)     # full suite, 300/300 on Linux"),
        h2("System requirements"),
        table(["Component", "Minimum", "Recommended"], [
            ["CPU", "x86-64 (SSE4) or ARM64", "AVX2 / AVX-512, 8+ cores"],
            ["RAM", "4 GB", "16 GB+ (large-model work)"],
            ["OS", "Linux (glibc 2.31+)", "Ubuntu 22.04 / 24.04"],
            ["Compiler", "Clang 16 / GCC 11", "Clang 18 + LLVM 18"],
        ]),
        h2("Platform support"),
        table(["Platform", "Status", "Notes"], [
            ["Linux x86-64", "✅ Verified", "Full test suite green; NUMA + keyring."],
            ["Linux ARM64", "✅ Builds/tests", "SIMD via NEON."],
            ["macOS ARM64/Intel", "✅ Build-verified", "Auto Homebrew llvm@18; JIT + integration tests pass locally; full serial ctest bring-up ongoing."],
            ["Windows", "⚠️ Experimental", "Code-complete (CredMan, Winsock); not yet CI-verified."],
        ]),
    ])


def page_running_cuda():
    return "".join([
        h1("Running CUDA Applications", "How VGRE intercepts and executes CUDA on the CPU."),
        h2("Execution model"),
        p("VGRE parses CUDA-C kernels through a Clang AST, lowers them to LLVM IR, and "
          "JIT-compiles to native code with <code>-O3 -march=native</code>. Kernel "
          "grids run on an OpenMP thread pool with SIMD vectorization; managed memory "
          "is backed by an OS page-fault handler."),
        h2("What works"),
        ul([
            "CUDA Runtime API (~101 functions): memory, streams, events, graphs, textures/surfaces, cooperative launch, CDP.",
            "CUDA Driver API (~56 functions): contexts, modules, PTX load + JIT link.",
            "cuBLAS (L1/L2/L3, complex), cuDNN (conv/pool/norm/attention/RNN), cuFFT/cuRAND/cuSOLVER/cuSPARSE.",
            "NCCL collectives (~95%) and OpenCL 1.2.",
            "PTX ISA (~95% of common instructions), kernel fusion, persistent JIT cache.",
        ]),
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
        h1("Distributed Clusters", "Scale kernels across machines over LAN or WAN."),
        h2("Token setup"),
        p("Every node in a cluster shares one auth token. Generate it on the master and "
          "distribute it to workers:"),
        code("# On the MASTER\n"
             "vgre-token generate            # writes ~/.vgre/token, prints SHA-256 fingerprint\n\n"
             "# On every WORKER (choose one)\n"
             "scp master:~/.vgre/token ~/.vgre/token\n"
             "vgre-token set <paste-token-here>\n\n"
             "# Verify both sides match\n"
             "vgre-token fingerprint"),
        h2("Start the cluster"),
        code("# Master (LAN auto-discovery on)\n"
             "vgre-start --master\n\n"
             "# Workers on the same LAN\n"
             "vgre-start --worker\n\n"
             "# Workers over WAN (explicit master address)\n"
             "vgre-start --worker --master-address 78.45.12.99:7777"),
        h2("WAN connectivity"),
        p("For NAT traversal, the master advertises its public address in UDP pings. "
          "<code>vgre-discover</code> finds your public IP and registers it:"),
        code("vgre-discover --set-master     # master: publish public IP\n"
             "vgre-discover --register       # cross-LAN token-keyed discovery"),
        p("Channels are authenticated with HMAC-SHA256 and encrypted with AES-256-CTR; "
          "transfers use LZ4 compression. Tuning knobs are in "
          "<a href=\"env-vars.html\">Environment Variables</a>."),
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
    core = table(["Variable", "Default", "Description"], [
        ["<code>VGRE_LIB_PATH</code>", "auto", "Absolute path to the shared library."],
        ["<code>VGRE_LOG_LEVEL</code>", "<code>INFO</code>", "DEBUG | INFO | WARN | ERROR."],
        ["<code>VGRE_INSTALL_DIR</code>", "<code>~/.local/share/VGRE</code>", "Installation directory."],
        ["<code>VGRE_WORKER_THREADS</code>", "nproc", "Override worker thread count."],
        ["<code>VGRE_SIMD_LEVEL</code>", "auto", "Force SSE4 | AVX | AVX2 | AVX512."],
        ["<code>VGRE_ENABLE_NUMA</code>", "<code>0</code>", "NUMA-aware scheduling (Linux)."],
    ])
    cluster = table(["Variable", "Default", "Description"], [
        ["<code>VGRE_PORT</code>", "<code>7777</code>", "TCP port for master and workers."],
        ["<code>VGRE_CLUSTER_MASTER_ADDRESS</code>", "—", "WAN direct connect (IP:PORT / host / [::1]:PORT)."],
        ["<code>VGRE_CLUSTER_ADVERTISED_ADDRESS</code>", "—", "Master public address in UDP pings (NAT)."],
        ["<code>VGRE_CLUSTER_NODES</code>", "—", "Comma-separated worker IP:PORT list (master)."],
        ["<code>VGRE_TCP_AUTH_TOKEN_FILE</code>", "<code>~/.vgre/token</code>", "Shared cluster auth token file."],
        ["<code>VGRE_CLUSTER_CONNECT_TIMEOUT_SEC</code>", "<code>10</code>", "TCP connect timeout (1–120)."],
    ])
    cache = table(["Variable", "Default", "Description"], [
        ["<code>VGRE_L1_CACHE_KB</code>", "<code>32</code>", "Per-block L1 cache (16 | 32 | 64 | 128)."],
        ["<code>VGRE_L2_CACHE_MB</code>", "<code>6</code>", "Per-device L2 cache (2 | 6 | 20 | 40)."],
    ])
    return "".join([
        h1("Environment Variables",
           "Written to <code>~/.vgre/env</code> by the installer and loaded automatically."),
        h2("Core"), core,
        h2("Cluster / networking"), cluster,
        h2("GPU cache model"), cache,
        callout(p("On Windows these are set in User scope by <code>vgre_sync.bat</code> / "
                  "<code>vgre_env.ps1</code>. The full list lives in "
                  "<code>docs/USER_GUIDE.md §5</code>."), ""),
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
        p("Drop-in <code>libcudart</code> replacement. Preload it (Linux "
          "<code>LD_PRELOAD</code>, macOS <code>DYLD_INSERT_LIBRARIES</code>) to route "
          "an existing CUDA binary's <code>cuda*</code> calls into VGRE."),
        h2("Dashboard telemetry"),
        p("The dashboard polls a JSON telemetry schema (utilization, kernels, cluster "
          "nodes, memory, logs). Full field list in <code>docs/api_reference.md §3–4</code>."),
    ])


def page_cli():
    return "".join([
        h1("CLI Tools", "The command-line utilities installed with VGRE."),
        table(["Command", "Purpose"], [
            ["<code>vgre-dashboard</code>", "Launch the real-time Flutter monitor."],
            ["<code>vgre-start</code>", "Start a master or worker node (<code>--master</code> / <code>--worker</code> / <code>--test</code>)."],
            ["<code>vgre-token</code>", "Generate / set / fingerprint the shared cluster auth token."],
            ["<code>vgre-discover</code>", "Public-IP discovery and cross-LAN registration."],
            ["<code>vgre_ptx_gdbserver</code>", "Serve a PTX kernel to a stock gdb for step debugging."],
            ["<code>gpt2_infer</code>", "Real GPT-2 inference from safetensors/GGUF (matches Hugging Face)."],
        ]),
        h2("Examples"),
        code("vgre-start --test                       # local self-test\n"
             "vgre-token generate                     # new token + fingerprint\n"
             "vgre-discover --set-master              # publish public IP\n"
             "gpt2_infer model.safetensors --gen 'Hello'"),
    ])


def page_architecture():
    return "".join([
        h1("Architecture", "How the pieces fit together."),
        h2("Layers"),
        ol([
            "<strong>Interception</strong> — CUDA/OpenCL API shims capture calls at load time.",
            "<strong>Compilation</strong> — Clang AST → LLVM IR → ORC JIT native code, with a persistent disk+memory cache.",
            "<strong>Execution</strong> — OpenMP thread pool + AVX2/AVX-512 SIMD; block/warp scheduling and shared-memory emulation.",
            "<strong>Memory</strong> — UVM via an OS page-fault handler; stream-ordered pools; LZ4-compressed cluster transfers.",
            "<strong>Distribution</strong> — authenticated, encrypted TCP transport; NCCL-style + software-RDMA collectives; tensor/pipeline parallelism and a GSPMD auto-partitioner.",
            "<strong>ML stack</strong> — autograd, BLAS GEMM, tokenizer, samplers, LoRA, HNSW, checkpoint loaders (safetensors / GGUF).",
        ]),
        h2("Design principles"),
        ul([
            "Real implementations only — no stubs, mocks, or placeholder math in shipped paths.",
            "From-scratch primitives (JIT, tokenizer, vector index, crypto, collectives) to avoid vendor lock-in.",
            "Warnings-as-errors, and a 300-test suite gating every change on Linux.",
        ]),
        p("Full detail in <code>docs/ARCHITECTURE.md</code>."),
    ])


def page_faq():
    return "".join([
        h1("FAQ & Troubleshooting"),
        h3("Does VGRE need a GPU?"),
        p("No. That is the point — kernels execute on the CPU. A GPU is never required."),
        h3("Is it as fast as a real GPU?"),
        p("No. Expect 10–50× slower for compute-bound kernels; memory-bound kernels are "
          "closer with AVX-512. VGRE targets learning, CI, development, and moderate "
          "workloads — not latency-critical production inference."),
        h3("Which platforms are supported?"),
        p("<strong>Linux x86-64/ARM64</strong> is fully verified (full <code>ctest</code> green). "
          "<strong>macOS</strong> (Apple Silicon and Intel) is build-verified: the native engine "
          "compiles with auto-detected Homebrew <code>llvm@18</code>, runs JIT kernels, and passes "
          "integration tests locally — run <code>bash install_local.sh</code>. "
          "<strong>Windows</strong> is code-complete but not yet CI-verified."),
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


PAGES = {
    "index.html": page_index,
    "quickstart.html": page_quickstart,
    "installation.html": page_installation,
    "running-cuda.html": page_running_cuda,
    "cluster.html": page_cluster,
    "dashboard.html": page_dashboard,
    "local-ai.html": page_local_ai,
    "debugging.html": page_debugging,
    "env-vars.html": page_env_vars,
    "api.html": page_api,
    "cli.html": page_cli,
    "architecture.html": page_architecture,
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
<link rel="stylesheet" href="assets/style.css">
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
    <a href="https://github.com/vgre-org/vgre-runtime">GitHub</a>
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
<script src="assets/search-index.js"></script>
<script src="assets/docs.js"></script>
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
