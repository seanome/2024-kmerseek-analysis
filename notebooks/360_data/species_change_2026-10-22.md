# Species change requested after the Ciona smoke results

Olga requested another species because Ciona was poorly annotated. The replacement
analysis uses Danio rerio (zebrafish), the fallback in the original request and the
second organism already included in the pipeline. Its answer key was built and
committed before its reads were simulated; it requires no new simulation or search.
The request followed the reported Ciona smoke results. This note was written on
2026-10-22 after the full run completed.

The two numerical pass/fail conditions remain unchanged. The notebook reports the
original Ciona result under the frozen plan and the requested zebrafish result under
the same two conditions. It does not describe the species change as preregistered.
The original tunicate-versus-vertebrate distance control remains in the notebook;
Ciona's annotation coverage limits its interpretation.

The stored provenance records 8,884 eligible zebrafish genes with matching peptide
MD5, one Pfam family, suitable transcript length, and that family in the search
database. The 200 selected genes represent 200 families and 200 clans: 50 up,
50 down, and 100 unchanged. Both paths use Swiss-Prot minus Danio, with the same
Pfam region table. The initial UniProt coverage check recorded 65,026 of 69,241
zebrafish proteins with a Pfam cross-reference; this count is not an assessment
of domain boundary accuracy.

Written by Codex (GPT-6) at Olga's request.
