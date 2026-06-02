#!/usr/bin/env python3
"""Extract assembled prompt strings from QueryPlanningPromptTemplate.java verbatim.

Implements just enough of Java's string-literal, text-block, and ``+``
concatenation semantics to evaluate the ``public static final String``
constants, then assembles the runtime-needed prompts. Evaluating the Java
expressions (rather than hand-transcribing them) guarantees byte-for-byte
fidelity with the OpenSearch source — including smart quotes, bullets, ``≥``,
and text-block dedenting.

Usage:
    python scripts/extract_qpt_prompts.py <java_source> <output_dir>

The functions here are also imported by the prompt-fidelity tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Which assembled constants become which resource files.
EMIT = {
    "query_planning_system_prompt.txt": "DEFAULT_QUERY_PLANNING_SYSTEM_PROMPT",
    "query_planning_user_prompt.txt": "DEFAULT_QUERY_PLANNING_USER_PROMPT",
    "template_selection_system_prompt.txt": "DEFAULT_TEMPLATE_SELECTION_SYSTEM_PROMPT",
    "template_selection_user_prompt.txt": "DEFAULT_TEMPLATE_SELECTION_USER_PROMPT",
    "default_search_template.txt": "DEFAULT_SEARCH_TEMPLATE",
}


def unescape(s: str) -> str:
    """Process Java string-literal escapes."""
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}
            if nxt in mapping:
                out.append(mapping[nxt])
                i += 2
                continue
            if nxt == "u":
                out.append(chr(int(s[i + 2 : i + 6], 16)))
                i += 6
                continue
        out.append(c)
        i += 1
    return "".join(out)


def dedent_text_block(raw: str) -> str:
    """Apply the JLS text-block incidental-whitespace algorithm.

    ``raw`` is the content between the opening ``\"\"\"<newline>`` and the closing
    ``\"\"\"`` (including the whitespace on the closing delimiter's line).
    """
    lines = raw.split("\n")

    def indent(line: str) -> int:
        return len(line) - len(line.lstrip(" \t"))

    candidates = [indent(l) for l in lines if l.strip() != ""]
    candidates.append(indent(lines[-1]))  # closing-delimiter line always counts
    strip = min(candidates) if candidates else 0
    stripped = []
    for l in lines:
        removed = l[strip:] if len(l) >= strip else l.lstrip(" \t")
        stripped.append(removed.rstrip(" \t"))  # JLS strips trailing whitespace
    return "\n".join(stripped)


def extract_constants(java_text: str) -> dict[str, str]:
    """Evaluate every ``public static final String`` constant in the source."""
    constants: dict[str, str] = {}
    n = len(java_text)
    marker = "public static final String "
    i = 0
    while True:
        idx = java_text.find(marker, i)
        if idx == -1:
            break
        j = idx + len(marker)
        k = j
        while k < n and (java_text[k].isalnum() or java_text[k] == "_"):
            k += 1
        name = java_text[j:k]
        while k < n and java_text[k] != "=":
            k += 1
        k += 1  # past '='
        value_parts: list[str] = []
        p = k
        while p < n:
            while p < n and java_text[p] in " \t\r\n":
                p += 1
            if p >= n:
                break
            c = java_text[p]
            if c == ";":
                p += 1
                break
            if c == "+":
                p += 1
                continue
            if java_text.startswith('"""', p):
                nl = java_text.index("\n", p + 3)
                close = java_text.index('"""', nl + 1)
                raw = java_text[nl + 1 : close]
                value_parts.append(unescape(dedent_text_block(raw)))
                p = close + 3
                continue
            if c == '"':
                q = p + 1
                buf: list[str] = []
                while q < n:
                    if java_text[q] == "\\":
                        buf.append(java_text[q : q + 2])
                        q += 2
                        continue
                    if java_text[q] == '"':
                        break
                    buf.append(java_text[q])
                    q += 1
                value_parts.append(unescape("".join(buf)))
                p = q + 1
                continue
            # identifier reference (possibly ClassName.-qualified)
            q = p
            while q < n and (java_text[q].isalnum() or java_text[q] in "_."):
                q += 1
            ref = java_text[p:q].split(".")[-1]
            if ref not in constants:
                raise KeyError(f"Unresolved reference {ref!r} while building {name!r}")
            value_parts.append(constants[ref])
            p = q
        constants[name] = "".join(value_parts)
        i = p
    return constants


def assemble(java_text: str) -> dict[str, str]:
    """Return ``{output_filename: assembled_prompt_text}`` for the EMIT set."""
    constants = extract_constants(java_text)
    return {fname: constants[const] for fname, const in EMIT.items()}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src = Path(argv[1])
    out = Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    assembled = assemble(src.read_text(encoding="utf-8"))
    for fname, content in assembled.items():
        (out / fname).write_text(content, encoding="utf-8")
        print(f"wrote {fname}: {len(content)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
