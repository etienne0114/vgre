// VGRE docs — shared client behavior. No framework, no build step.
(function () {
  "use strict";

  // ── Mobile sidebar toggle ────────────────────────────────────────────────
  var toggle = document.querySelector(".menu-toggle");
  var sidebar = document.querySelector(".sidebar");
  if (toggle && sidebar) {
    toggle.addEventListener("click", function () { sidebar.classList.toggle("open"); });
    document.querySelectorAll(".sidebar .nav-link").forEach(function (a) {
      a.addEventListener("click", function () { sidebar.classList.remove("open"); });
    });
  }

  // ── Copy button (lives in each terminal's title bar) ─────────────────────
  function wireCopy(btn) {
    btn.addEventListener("click", function () {
      var host = btn.closest(".terminal") || btn.parentNode;
      var code = host.querySelector("pre code") || host.querySelector("code");
      var text = code ? code.innerText : "";
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1500);
      });
    });
  }
  document.querySelectorAll(".copy-btn").forEach(wireCopy);
  // Any bare <pre> not already inside a terminal still gets a copy button.
  document.querySelectorAll("pre").forEach(function (pre) {
    if (pre.closest(".terminal") || pre.querySelector(".copy-btn")) return;
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    pre.appendChild(btn);
    wireCopy(btn);
  });

  // ── Lightweight, dependency-free syntax highlighting ─────────────────────
  // A manual per-line scanner: it emits already-escaped text so it can never
  // corrupt its own markup, and it is language-aware — in shell blocks the
  // leading word of each command is coloured as a command and --flags stand
  // out, mirroring a real terminal. Runs on <code class="lang-*">.
  var KW = /^(sudo|export|import|from|class|def|return|if|elif|else|fi|then|for|while|do|done|switch|case|const|auto|void|int|long|float|double|char|bool|struct|enum|namespace|public|private|template|using|new|delete|true|false|null|nullptr|None|self|async|await|assert|include|define|print|printf)$/;
  var CMD = /^(bash|sh|zsh|cmake|make|ninja|pip|pip3|python|python3|node|npm|git|curl|wget|scp|ssh|nc|rsync|apt|apt-get|dnf|yum|snap|brew|winget|choco|ldd|otool|dumpbin|ls|cat|cd|rm|cp|mv|mkdir|echo|grep|sed|awk|tar|chmod|chown|kill|nproc|sysctl|ctest|ctest3|flutter|dart|docker|kubectl|helm|source|set|test|dnf|ufw|rdma|Test-Path|Test-NetConnection|New-NetFirewallRule|Get-ChildItem|Select-Object)$/;
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function span(cls, s) { return '<span class="' + cls + '">' + esc(s) + "</span>"; }
  function hlLine(line, shell) {
    var out = "", i = 0, n = line.length, seenCmd = false;
    while (i < n) {
      var c = line[i], j;
      if (/\s/.test(c)) { j = i; while (j < n && /\s/.test(line[j])) j++; out += esc(line.slice(i, j)); i = j; continue; }
      // Comments: # anywhere (shell/py), // when not part of a URL scheme.
      if (c === "#" || (c === "/" && line[i + 1] === "/" && line[i - 1] !== ":")) { out += span("tok-com", line.slice(i)); break; }
      // Strings: " ' or ` with escapes.
      if (c === '"' || c === "'" || c === "`") { j = i + 1; while (j < n) { if (line[j] === "\\") { j += 2; continue; } if (line[j] === c) { j++; break; } j++; } out += span("tok-str", line.slice(i, j)); i = j; seenCmd = true; continue; }
      // Flags: -x / --flag (only at a token boundary).
      if (c === "-" && /[A-Za-z-]/.test(line[i + 1] || "") && (i === 0 || /\s/.test(line[i - 1]))) { j = i; while (j < n && /[\w-]/.test(line[j])) j++; out += span("tok-flag", line.slice(i, j)); i = j; seenCmd = true; continue; }
      // Variables: $VAR / ${...} / %VAR%.
      if (c === "$") { j = i + 1; if (line[j] === "{") { while (j < n && line[j] !== "}") j++; j++; } else { while (j < n && /[\w]/.test(line[j])) j++; } out += span("tok-var", line.slice(i, j)); i = j; seenCmd = true; continue; }
      if (c === "%" && /[A-Za-z_]/.test(line[i + 1] || "")) { j = i + 1; while (j < n && line[j] !== "%") j++; j++; out += span("tok-var", line.slice(i, j)); i = j; seenCmd = true; continue; }
      // Numbers.
      if (/\d/.test(c) && (i === 0 || !/[\w]/.test(line[i - 1]))) { j = i; while (j < n && /[\d.xa-fA-F]/.test(line[j])) j++; out += span("tok-num", line.slice(i, j)); i = j; seenCmd = true; continue; }
      // Words (identifiers, commands, paths).
      if (/[A-Za-z_]/.test(c)) { j = i; while (j < n && /[\w.\-]/.test(line[j])) j++; var w = line.slice(i, j); var cls = "";
        if (shell && !seenCmd && (CMD.test(w) || /^vgre[\w-]*$/.test(w))) cls = "tok-cmd";
        else if (KW.test(w)) cls = "tok-kw";
        out += cls ? span(cls, w) : esc(w); i = j; seenCmd = true; continue; }
      // Operators; a pipe / ; / && starts a fresh command segment in shell.
      out += esc(c);
      if (shell && (c === "|" || c === ";" || c === "&")) seenCmd = false;
      i++;
    }
    return out;
  }
  function highlight(el) {
    var m = el.className.match(/lang-([\w+]+)/);
    var lang = m ? m[1] : "";
    var shell = lang === "bash" || lang === "sh" || lang === "shell" || lang === "console" || lang === "powershell" || lang === "ps1";
    el.innerHTML = el.textContent.split("\n").map(function (ln) { return hlLine(ln, shell); }).join("\n");
  }
  document.querySelectorAll('code[class*="lang-"]').forEach(highlight);

  // ── Active nav-link based on current page ────────────────────────────────
  var here = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".sidebar .nav-link").forEach(function (a) {
    var href = a.getAttribute("href");
    if (href === here) a.classList.add("active");
  });

  // ── Build the right-hand TOC from <h2>/<h3> and scroll-spy ────────────────
  var content = document.querySelector(".content");
  var toc = document.querySelector(".toc-list");
  if (content && toc) {
    var heads = content.querySelectorAll("h2, h3");
    heads.forEach(function (h) {
      if (!h.id) {
        h.id = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      }
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      if (h.tagName === "H3") a.className = "h3";
      toc.appendChild(a);
    });
    var links = toc.querySelectorAll("a");
    var spy = function () {
      var pos = window.scrollY + 120;
      var current = null;
      heads.forEach(function (h) { if (h.offsetTop <= pos) current = h.id; });
      links.forEach(function (a) {
        a.classList.toggle("active", a.getAttribute("href") === "#" + current);
      });
    };
    window.addEventListener("scroll", spy, { passive: true });
    spy();
  }

  // ── Client-side search over the static index ─────────────────────────────
  var search = document.querySelector(".header-search");
  if (search && window.VGRE_SEARCH_INDEX) {
    var box = document.createElement("div");
    box.style.cssText =
      "position:fixed;top:56px;right:24px;width:340px;max-height:60vh;overflow:auto;" +
      "background:#0f1115;border:1px solid rgba(255,255,255,.14);border-radius:12px;" +
      "padding:8px;z-index:200;display:none;box-shadow:0 10px 40px rgba(0,0,0,.6)";
    document.body.appendChild(box);
    function render(q) {
      q = q.trim().toLowerCase();
      if (q.length < 2) { box.style.display = "none"; return; }
      var hits = window.VGRE_SEARCH_INDEX.filter(function (e) {
        return e.title.toLowerCase().indexOf(q) >= 0 || e.text.toLowerCase().indexOf(q) >= 0;
      }).slice(0, 8);
      if (!hits.length) { box.innerHTML = '<div style="padding:10px;color:#62626c">No results</div>'; }
      else {
        box.innerHTML = hits.map(function (e) {
          return '<a href="' + e.url + '" style="display:block;padding:9px 12px;border-radius:8px;color:#e8e8e8">' +
            '<div style="font-weight:600;color:#00ffd1">' + e.title + '</div>' +
            '<div style="font-size:12px;color:#8a8a94">' + e.section + '</div></a>';
        }).join("");
      }
      box.style.display = "block";
    }
    search.addEventListener("input", function () { render(search.value); });
    search.addEventListener("blur", function () { setTimeout(function () { box.style.display = "none"; }, 200); });
  }
})();
