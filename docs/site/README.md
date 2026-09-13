# VGRE Documentation Website

A professional, from-scratch static documentation site — no framework, no build
toolchain, no external vendor. Plain HTML/CSS/JS that opens directly in a browser
or serves from any static host (GitHub Pages, S3, nginx, `python -m http.server`).

The dashboard's **Help & Docs** sidebar links open these pages in the user's
browser (see `vgre_dashboard/lib/core/docs_links.dart`).

## Structure

```
docs/site/
├── build_site.py            # generator — authoritative page content lives here
├── index.html … faq.html    # generated pages (checked in)
└── assets/
    ├── style.css            # dark-neon theme matching the dashboard
    ├── docs.js              # TOC, scroll-spy, copy buttons, search, highlighting
    └── search-index.js      # generated client-side search index
```

## Regenerate after editing content

Page content is authored in `build_site.py` (a small markup DSL). After editing:

```bash
python3 docs/site/build_site.py     # rewrites the HTML pages + search index
```

## Preview locally

```bash
python3 -m http.server 8099 --directory docs/site
# open http://localhost:8099
```

## Deploy

Publish the `docs/site/` directory to any static host. Set the dashboard's docs
base URL to match at build time:

```bash
flutter build linux --release --dart-define=VGRE_DOCS_URL=https://your-host/docs
```

## Design

Patterns follow the 2026 developer-docs conventions (Mintlify / Docusaurus /
Stripe): grouped left navigation, an auto-generated right-hand "On this page"
TOC with scroll-spy, per-block copy buttons, client-side search, and callout /
card / table components — all implemented in ~250 lines of vanilla CSS+JS so
there is no dependency to maintain or license.
