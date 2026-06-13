#!/usr/bin/env python3
"""Validate the VGRE IaC + GitOps artifacts without a cluster/cloud.

Terraform (deploy/terraform): every .tf has balanced braces/quotes; a terraform{}
block with required_providers exists; and every `var.X` referenced is declared in
a `variable "X"` block. If the `terraform` CLI is present, also `terraform fmt -check`.

GitOps (deploy/gitops): every .yaml parses; kustomization files are kind
Kustomization; the ArgoCD app is kind Application. If `kustomize` is present,
`kustomize build` the prod overlay.
"""
import glob
import os
import re
import shutil
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
errors = []


def err(m):
    errors.append(m)


def check_terraform():
    tdir = os.path.join(ROOT, "terraform")
    tfs = glob.glob(os.path.join(tdir, "*.tf"))
    if not tfs:
        err("no .tf files found")
        return
    declared, used, has_required_providers = set(), set(), False
    for tf in tfs:
        text = open(tf).read()
        if text.count("{") != text.count("}"):
            err(f"{os.path.basename(tf)}: unbalanced braces")
        if text.count('"') % 2 != 0:
            err(f"{os.path.basename(tf)}: unbalanced quotes")
        declared |= set(re.findall(r'variable\s+"([A-Za-z0-9_]+)"', text))
        used |= set(re.findall(r"var\.([A-Za-z0-9_]+)", text))
        if "required_providers" in text:
            has_required_providers = True
    if not has_required_providers:
        err("terraform: no required_providers block")
    missing = used - declared
    if missing:
        err(f"terraform: undeclared variables referenced: {sorted(missing)}")
    tf_cli = shutil.which("terraform")
    if tf_cli:
        cp = subprocess.run([tf_cli, "fmt", "-check", "-recursive", tdir], capture_output=True, text=True)
        if cp.returncode != 0:
            err(f"terraform fmt -check failed:\n{cp.stdout}{cp.stderr}")
        print("terraform fmt -check: OK")
    else:
        print("terraform CLI not found — structural validation only")


def check_gitops():
    gdir = os.path.join(ROOT, "gitops")
    yamls = glob.glob(os.path.join(gdir, "**", "*.yaml"), recursive=True)
    if not yamls:
        err("no gitops yaml found")
        return
    saw_app = saw_kustomization = False
    for y in yamls:
        try:
            docs = list(yaml.safe_load_all(open(y)))
        except yaml.YAMLError as e:
            err(f"{os.path.relpath(y, gdir)}: invalid YAML: {e}")
            continue
        for d in docs:
            if not isinstance(d, dict) or "kind" not in d or "apiVersion" not in d:
                err(f"{os.path.relpath(y, gdir)}: missing apiVersion/kind")
                continue
            if d["kind"] == "Application":
                saw_app = True
            if d["kind"] == "Kustomization":
                saw_kustomization = True
    if not saw_app:
        err("gitops: no ArgoCD Application found")
    if not saw_kustomization:
        err("gitops: no Kustomization found")
    kz = shutil.which("kustomize")
    if kz:
        overlay = os.path.join(gdir, "kustomize", "overlays", "prod")
        cp = subprocess.run([kz, "build", overlay], capture_output=True, text=True)
        if cp.returncode != 0:
            err(f"kustomize build failed:\n{cp.stderr}")
        print("kustomize build overlays/prod: OK")
    else:
        print("kustomize CLI not found — structural validation only")


def main():
    check_terraform()
    check_gitops()
    if errors:
        print("IaC VALIDATION FAILED:")
        for e in errors:
            print("  - " + e)
        return 1
    print("VGRE IaC + GitOps artifacts OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
