#!/usr/bin/env python3
"""VGRE mutation-testing runner (docs/missingFeatures.md §6.3).

Applies single-point mutation operators to a target C/C++ source, rebuilds it
with its test, runs the test, and classifies each mutant as KILLED (the test
failed -> good) or SURVIVED (the test passed -> a gap in the test). Reports a
mutation score = killed / total. A surviving mutant means the test suite does
not actually exercise that behaviour.

Real, self-contained: no external mutation framework. Operators cover
relational / logical / arithmetic / increment operators and boolean literals.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

# (match, replacement) applied at non-overlapping operator tokens, longest first.
OPERATORS = [
    (">=", ">"), ("<=", "<"), ("==", "!="), ("!=", "=="),
    ("&&", "||"), ("||", "&&"), ("++", "--"), ("--", "++"),
    (">", ">="), ("<", "<="), ("+", "-"), ("-", "+"), ("*", "/"),
]
KEYWORDS = [("true", "false"), ("false", "true")]


def code_mask(src):
    """Mark which character positions are real code (not comment / string / char)."""
    n = len(src)
    mask = [True] * n
    i = 0
    state = "code"
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line"; mask[i] = mask[i + 1] = False; i += 2; continue
            if c == "/" and nxt == "*":
                state = "block"; mask[i] = mask[i + 1] = False; i += 2; continue
            if c == '"':
                state = "str"; mask[i] = False; i += 1; continue
            if c == "'":
                state = "char"; mask[i] = False; i += 1; continue
            i += 1
        elif state == "line":
            mask[i] = False
            if c == "\n": state = "code"
            i += 1
        elif state == "block":
            mask[i] = False
            if c == "*" and nxt == "/":
                mask[i + 1] = False; i += 2; state = "code"; continue
            i += 1
        elif state in ("str", "char"):
            mask[i] = False
            if c == "\\":
                if i + 1 < n: mask[i + 1] = False
                i += 2; continue
            if (state == "str" and c == '"') or (state == "char" and c == "'"):
                state = "code"
            i += 1
    return mask


def gen_mutants(src):
    """Yield (description, mutated_source) for each single-point code mutation."""
    n = len(src)
    mask = code_mask(src)
    i = 0
    # operator-token mutations (longest-match, non-overlapping scan, code only)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        matched = False
        for a, b in OPERATORS:
            if src.startswith(a, i) and all(mask[i:i + len(a)]):
                yield (f"{a}->{b}@{i}", src[:i] + b + src[i + len(a):])
                i += len(a)
                matched = True
                break
        if not matched:
            i += 1
    # boolean-literal mutations (word-boundary, code only)
    def is_word(c):
        return c.isalnum() or c == "_"
    for a, b in KEYWORDS:
        start = 0
        while True:
            j = src.find(a, start)
            if j < 0:
                break
            before = src[j - 1] if j > 0 else " "
            after = src[j + len(a)] if j + len(a) < n else " "
            if not is_word(before) and not is_word(after) and all(mask[j:j + len(a)]):
                yield (f"{a}->{b}@{j}", src[:j] + b + src[j + len(a):])
            start = j + len(a)


def compile_and_run(cxx, target_src_path, test_path, workdir):
    binp = os.path.join(workdir, "mutant_bin")
    cp = subprocess.run([cxx, "-std=c++17", "-O0", target_src_path, test_path, "-o", binp],
                        capture_output=True)
    if cp.returncode != 0:
        return "compile_error"  # a mutant that doesn't compile is also "killed"
    rp = subprocess.run([binp], capture_output=True)
    return "pass" if rp.returncode == 0 else "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    ap.add_argument("--min-score", type=float, default=1.0,
                    help="fail (exit 1) if mutation score is below this")
    args = ap.parse_args()

    with open(args.target) as f:
        original = f.read()

    with tempfile.TemporaryDirectory() as wd:
        mtgt = os.path.join(wd, "mutated_target" + os.path.splitext(args.target)[1])

        # Baseline: the unmutated target+test MUST pass.
        with open(mtgt, "w") as f:
            f.write(original)
        if compile_and_run(args.cxx, mtgt, args.test, wd) != "pass":
            print("BASELINE FAILED: target+test do not pass unmutated", file=sys.stderr)
            return 2

        killed = survived = 0
        survivors = []
        for desc, mutant in gen_mutants(original):
            if mutant == original:
                continue
            with open(mtgt, "w") as f:
                f.write(mutant)
            res = compile_and_run(args.cxx, mtgt, args.test, wd)
            if res == "pass":
                survived += 1
                survivors.append(desc)
            else:
                killed += 1  # fail or compile_error

        total = killed + survived
        score = (killed / total) if total else 1.0
        report = {"target": args.target, "mutants": total, "killed": killed,
                  "survived": survived, "mutation_score": round(score, 4),
                  "survivors": survivors}
        print(json.dumps(report, indent=2))
        return 0 if score >= args.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
