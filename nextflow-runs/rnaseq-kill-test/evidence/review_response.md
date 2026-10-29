The original review above is reproduced without edits. Its line references describe the reviewed pre-run snapshot, before these fixes.

- Reject boolean recall gains before conversion to a number. The regression test sends True and requires an error.
- Require an NCBI nodes.dmp snapshot and derive each sequence species’ ancestors from its parent links. Reject unknown taxa, cycles and records that disagree. The regression test rejects a human record whose supplied lineage omits vertebrates. Correspondence between the audited members and a built index remains unimplemented.
- Require the existing report to match its last generated SHA-256 before changing any output. A regression test appends a measurement while retaining the incomplete-implementation sentence. Regeneration now fails and leaves the report unchanged.

These fixes do not supply biological evidence. Every scientific claim remains not measured. The biological workflows, calibration, data selection and figures remain unimplemented.

### Independent follow-up, unedited

Follow-up review, 2026-10-22:

- HOLDS — Both True and False are rejected as recall-gain measurements. I reran the original boolean counterexample.
- HOLDS — The truncated human lineage is rejected using ancestry derived from the supplied taxonomy. A nonvertebrate control remains accepted. This checks the audit logic; the taxonomy’s authenticity and its correspondence to actual index sequences remain unverified.
- HOLDS — Adding a biological measurement to a generated report blocks regeneration. I checked every file in the temporary directory before and after rejection; none changed. Regeneration of an unchanged report still succeeds.
- HOLDS — All 15 unit tests pass. I found no new regression in these three fixes.
- UNVERIFIABLE — All five biological claims remain unmeasured. The overall RNA-seq kill-test implementation remains incomplete.
