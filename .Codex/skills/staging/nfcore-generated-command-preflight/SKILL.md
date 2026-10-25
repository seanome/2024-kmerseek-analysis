---
name: nfcore-generated-command-preflight
description: Use when changing nf-core extra tool arguments or resuming a Nextflow run after a tool command-line parsing failure.
---

# Test the command the workflow actually executes

Added 2026-10-25 after Botryllus trimming job 49368820 failed on duplicated --no_poly_g. The wrapper and pinned nf-core module both supplied the same singleton option.

1. Read the failed task's .command.sh and .command.err and the pinned module/config that injects arguments. A shell syntax check or successful config resolution does not test the composed tool command.
2. Give a workflow-managed singleton option one owner. Avoid guessing that every option is non-repeatable: use the pinned tool's documented behavior and the observed failure.
3. Retain the failed generated command as a regression fixture. Verify it fails the new check and the corrected command passes.
4. Run small paired test reads through the real pinned workflow, same container, same config and shared parameter file. Require successful task execution and validate paired output counts and content. A stub or --help alone is insufficient.
5. Stop before experimental processing on any failed or missing preflight output. Record the generated argv, module/config/parameter hashes and output checks. Re-run when these change.
6. Resume existing cached work only after the test, and rebuild dependencies cancelled by the original failure. Do not confuse submission, queued tests, successful tests and full analysis completion.

This project's implementation is scripts/botryllus_shazam/nfcore_preflight.py, wired into nfcore.sbatch before the experimental workflow. Run test_nfcore_preflight.py for the duplicate-option regression. Execute workflows on Sherlock, never on the laptop.
