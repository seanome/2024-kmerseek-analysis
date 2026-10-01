#!/usr/bin/env python3
"""PreToolUse hook: refuse to post an /analysis-review comment that has no Reviewer #2
section.

The analysis-review skill (.claude/skills/analysis-review/SKILL.md) asks for a check of the
review itself before it is posted: re-compute the headline number, try to show each finding
wrong, and name what was not tested. A rule in a skill is skipped when the skill is not
read closely, so this hook refuses the post instead. Added 2026-10-01 on PR #56.

A post counts as an analysis review when its body has the heading "## /analysis-review".
It is let through only when the body also has the SECTION heading below, at least one
"- " line under it, and a line that starts with "Not tested" (or "- Not tested").
Rename the section here and in the skill together.
"""
import json
import os
import re
import sys

SECTION = "Preparing for an Adversarial Reviewer #2: Stress-testing the findings"
# At a line start, or at the start of an inline `-f body=...` value.
REVIEW_HEADING = re.compile(r"(?:^|body=['\"]?)## /analysis-review\b", re.M)

# `gh` only where a command starts, so a commit message that mentions gh is not taken for one.
AT_CMD = r"(?:^|[;&|(\n{`]|\$\()\s*(?:[A-Z_][A-Z0-9_]*=\S*\s+)*"
GH_WRITE = re.compile(AT_CMD + r"gh\s+(api|pr\s+(comment|review|create|edit)|issue\s+(comment|create|edit))\b", re.M)
FILE_REFS = [
    re.compile(r"(?:body|description|message)=@(?!-)(['\"]?)([^\s'\"]+)\1"),
    re.compile(r"--body-file[=\s]+(['\"]?)([^\s'\"]+)\1"),
    re.compile(r"(?:^|\s)-F\s+(['\"]?)([^\s'\"=]+)\1(?=\s|$)"),
    re.compile(r"--input[=\s]+(['\"]?)([^\s'\"]+)\1"),
]


def deny(reason):
    json.dump({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                      "permissionDecision": "deny",
                                      "permissionDecisionReason": reason}}, sys.stdout)
    sys.exit(0)


def section_problem(body):
    """None if the body has a filled-in SECTION, else what is missing."""
    m = re.search(rf"^##\s+{re.escape(SECTION)}\s*$", body, re.M)
    if not m:
        return f'there is no "## {SECTION}" section'
    rest = body[m.end():]
    end = re.search(r"^##\s|^<!--", rest, re.M)
    section = rest[: end.start()] if end else rest
    bullets = [line for line in section.splitlines() if line.strip().startswith("- ")]
    if not bullets:
        return f'the "## {SECTION}" section has no "- " lines'
    if not re.search(r"^\s*(?:-\s+)?Not tested", section, re.M):
        return f'the "## {SECTION}" section has no "Not tested" line'
    return None


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return
    if data.get("tool_name") != "Bash":
        return
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not GH_WRITE.search(cmd) or re.search(r"(-X|--method)\s*['\"]?GET\b", cmd):
        return

    cwd = data.get("cwd") or os.getcwd()
    bodies = [cmd]
    for pat in FILE_REFS:
        for m in pat.finditer(cmd):
            path = os.path.expanduser(m.group(2))
            if not os.path.isabs(path):
                path = os.path.join(cwd, path)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    bodies.append(fh.read())
            except OSError:
                pass

    for body in bodies:
        if REVIEW_HEADING.search(body):
            problem = section_problem(body)
            if problem:
                deny(f"This posts an /analysis-review comment, but {problem}. Run Step 5 of "
                     ".claude/skills/analysis-review/SKILL.md (the are-you-sure pass on the "
                     f"review), then add:\n\n## {SECTION}\n- Headline re-computed: ...\n"
                     "- Outputs current: ...\n- Finding <n>: tried to show it wrong by ...\n"
                     "- Not tested: ...")


if __name__ == "__main__":
    main()
