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

  // ── Copy buttons on every <pre> ──────────────────────────────────────────
  document.querySelectorAll("pre").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var code = pre.querySelector("code");
      var text = code ? code.innerText : pre.innerText;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = "Copy"; }, 1500);
      });
    });
    pre.appendChild(btn);
  });

  // ── Lightweight syntax highlighting ──────────────────────────────────────
  // Single-pass tokenizer over the *raw* text so it never re-scans its own
  // emitted markup (a sequential-replace approach corrupts spans whose
  // attributes contain keywords like "class"). Runs only on <code class="lang-*">.
  var KW = /^(sudo|bash|cmake|make|ninja|export|import|from|class|def|return|if|else|for|while|const|auto|void|int|float|struct|namespace|public|private|using|new|delete|true|false|null|None|self|async|await|pip|git|cd|source|assert)$/;
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  // One combined pattern; the winning alternative decides the token class.
  // '//' is a comment only when not part of a URL scheme (not preceded by ':').
  var TOKEN = /(#[^\n]*|(?<!:)\/\/[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)/g;
  function highlight(el) {
    var src = el.textContent; // raw, unescaped source
    var out = "";
    var last = 0;
    var m;
    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(src)) !== null) {
      out += esc(src.slice(last, m.index));
      if (m[1]) out += '<span class="tok-com">' + esc(m[1]) + "</span>";
      else if (m[2]) out += '<span class="tok-str">' + esc(m[2]) + "</span>";
      else if (m[3]) out += '<span class="tok-num">' + esc(m[3]) + "</span>";
      else if (m[4]) out += KW.test(m[4]) ? '<span class="tok-kw">' + m[4] + "</span>" : esc(m[4]);
      last = m.index + m[0].length;
    }
    out += esc(src.slice(last));
    el.innerHTML = out;
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
