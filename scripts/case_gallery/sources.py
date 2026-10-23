"""Where the reference-pair tab's kmerseek searches come from, in one place.

D241 holds notebook 241's 152 alphabet/k searches rerun with kmerseek PR #112
(olgabot/run-evalue-v2, 9878ff8), which gives every region an E-value: the
Karlin-Altschul one where the pair has a positive lambda, otherwise the run E-value.
Its plan.json, queries.fa, pair/, logs/ and ka_survival/ are links to notebook 241's
own folder (D241_ORIGINAL); only search/ and the files collect writes are new.

NB241 is a checkout of notebook 241's branch at the commit whose collect script and
random-protein null this tab reuses.
"""
from pathlib import Path

D241 = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases-pr112")
D241_ORIGINAL = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
KMERSEEK_PR = 112
KMERSEEK_COMMIT = "9878ff8"
NB241 = Path("/Users/olga/code/2024-kmerseek-analysis-241")
NB241_COMMIT = "b3294246d841d1082e327b4fc31e678771ed22f7"
