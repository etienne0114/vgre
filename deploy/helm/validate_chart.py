#!/usr/bin/env python3
"""Validate the VGRE Helm chart without requiring a cluster.

Checks (always):
  * Chart.yaml is valid YAML with the required v2 fields.
  * values.yaml is valid YAML.
  * every template has balanced {{ }} delimiters and balanced if/end blocks.
  * every {{ .Values.<path> }} reference resolves to a key that exists in
    values.yaml (catches typos that `helm template` would only surface at deploy).
Additionally, if `helm` is on PATH: runs `helm lint` and `helm template`.
"""
import os
import re
import shutil
import subprocess
import sys

try:
    import yaml  # use PyYAML when available
except ImportError:  # otherwise fall back to the in-tree dependency-free loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import minyaml as yaml

CHART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vgre")
errors = []


def err(msg):
    errors.append(msg)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def value_exists(values, dotted):
    cur = values
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False
    return True


def main():
    # 1. Chart.yaml
    chart = load_yaml(os.path.join(CHART, "Chart.yaml"))
    for field in ("apiVersion", "name", "version", "type"):
        if field not in chart:
            err(f"Chart.yaml missing required field: {field}")
    if chart.get("apiVersion") != "v2":
        err("Chart.yaml apiVersion must be v2")

    # 2. values.yaml
    values = load_yaml(os.path.join(CHART, "values.yaml"))
    if not isinstance(values, dict):
        err("values.yaml must be a mapping")

    # 3. templates
    tdir = os.path.join(CHART, "templates")
    builtins_ok = ("Release.", "Chart.", "Capabilities.", "Files.", "Template.")
    for fn in sorted(os.listdir(tdir)):
        if fn in ("_helpers.tpl", "NOTES.txt"):
            continue
        if not fn.endswith(".yaml"):
            continue
        text = open(os.path.join(tdir, fn)).read()
        if text.count("{{") != text.count("}}"):
            err(f"{fn}: unbalanced {{{{ }}}} delimiters")
        ifs = len(re.findall(r"{{-?\s*if\b", text))
        ranges = len(re.findall(r"{{-?\s*range\b", text))
        ends = len(re.findall(r"{{-?\s*end\b", text))
        if ifs + ranges != ends:
            err(f"{fn}: unbalanced if/range vs end ({ifs+ranges} opens, {ends} ends)")
        for ref in re.findall(r"\.Values\.([A-Za-z0-9_.]+)", text):
            ref = ref.rstrip(".")
            if not value_exists(values, ref):
                err(f"{fn}: references undefined value .Values.{ref}")

    # 4. optional helm lint/template
    helm = shutil.which("helm")
    if helm:
        for args in (["lint", CHART], ["template", "vgre", CHART]):
            cp = subprocess.run([helm] + args, capture_output=True, text=True)
            if cp.returncode != 0:
                err(f"helm {args[0]} failed:\n{cp.stderr or cp.stdout}")
        print("helm lint + template: OK")
    else:
        print("helm not found — ran structural validation only")

    if errors:
        print("CHART VALIDATION FAILED:")
        for e in errors:
            print("  - " + e)
        return 1
    print(f"VGRE Helm chart OK ({len(os.listdir(tdir))} template files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
