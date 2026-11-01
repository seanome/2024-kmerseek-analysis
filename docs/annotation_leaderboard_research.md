# A leaderboard for protein function annotation tools

2026-10-15 · Olga Botvinnik. Working copy: https://claude.ai/code/artifact/91b5c2f3-b81b-4bba-a9cb-58e01d105a17 (this file is the committed snapshot of that doc).

None of the 15 benchmarks checked below lets an Earth BioGenome Project annotator filter results by their own clade, gene type or region length and see the compute cost next to the score. The closest are LAFA (the host runs containers on a time split, but it scores whole-protein GO only and reports no cost) and CAID3 (the host runs containers and records run time, but it scores disorder only). A useful leaderboard needs four things: region-level truth from a time split, strata computed from the query proteome alone, cost in CPU-hours, RAM and dollars per 1,000 proteins, and a host that runs every tool itself.

How to read the table: every cell is backed by a page listed in that row's Sources column, and the references are at the end. "not found" means the fetched page is silent; "no" means the page says or shows the thing is absent.

## 1. What exists

Four of 15 have the host run every tool (LAFA, Open Problems, CAID3, LiveBench-1); of those only CAID3 records run time, and none of the 15 reports dollars. Strata columns read: organism · clade distance · length · disorder · feature type. Cost columns read: time · memory · GPU · dollars. R-numbers point to the References.

| Benchmark | Scored | Truth set | Who runs the tools | Strata (org · clade · len · dis · feat) | Reports (time · mem · GPU · $) | Containers | Operator | Built for | Sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [CAFA](https://biofunctionprediction.org/cafa/) (1 to 5) | Whole-protein GO and Human Phenotype Ontology terms | Proteins with no experimental function that gain experimental annotation after the prediction deadline | Submitters send predictions; CAFA 5 ran on Kaggle | partly (separate prokaryote and eukaryote target sets in CAFA 1) · not found · not found · not found · not found | not found ×4 | not found (a format checker only) | "CAFA organizers" | Method developers | R1 |
| [LAFA](https://arxiv.org/html/2604.20782v2) | Whole-protein GO (BP, MF, CC); best-threshold F1 with term weights | Experimental GO annotations gained by Swiss-Prot proteins between two UniProt-GOA releases (7,401 proteins, 9/2025 to 3/2026); releases about every 8 weeks | Host pulls containers from Docker Hub and runs them on its cluster | not found ×5 (only GO aspect, and NK/LK/PK knowledge level in the results repo) | no · no · no · no | Yes: Docker or Apptainer | Friedberg group, Iowa State | Method developers and "the scientific community at large" | R2, R3, R4 |
| [PROBE](https://github.com/kansil/PROBE) | Semantic similarity, GO prediction, drug-target family, binding affinity | Human Swiss-Prot; SKEMPI v1 | User runs scripts on embedding vectors; vectors of 20 published methods supplied | no (human only) · not found · not found · not found · not found | not found ×4 | No (Code Ocean capsule, pip) | kansil (GitHub) | Embedding authors | R5 |
| [BeProf](https://github.com/CSUBioGroup/BEPROF) | Whole-protein GO (BP, MF, CC); Fmax, Smin, AUPR, IC_AUPR, DP_AUPR | Per-protein GO sets plus a background set for information content | User runs the evaluation script on own predictions | not found ×5 | not found ×4 | not found | Wang and Li (csu.edu.cn addresses) | Deep-learning method developers | R6 |
| [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) | Variant fitness: 217 substitution and 74 indel DMS assays; clinical variants | DMS assay scores; pathogenic or benign labels | Submitters supply all scores and a scoring script by PR; model must be open source; maintainers score some popular baselines | yes (taxon) · yes (MSA depth) · not found · not found · yes (function category) | not found ×4 | No benchmark container | Marks lab | Fitness-model developers | R7 |
| [PEER](https://github.com/DeepGraphLearning/PEER_Benchmark) / TorchProtein | 17 tasks: function, localization, structure, interaction, ligand | Not described | Authors ran baselines; community submissions "in the future" | not found ×5 | not found · not found · suggested V100 setup only · not found | No (conda, pip) | Tang group (DeepGraphLearning) | ML authors (NeurIPS 2022 Datasets track) | R8 |
| [TAPE](https://github.com/songlab-cal/tape) | Secondary structure, contacts, remote homology, fluorescence, stability | PDB and CASP12, ProteinNet, SCOPe, GFP and stability screens | not found | not found ×5 | one throughput figure (200 seq/s on a Titan Xp) · not found · not found · not found | not found (pip) | Song lab | ML authors | R9 |
| [FLIP](https://github.com/J-SNACKKB/FLIP) | Fitness-landscape train/test splits | Not described | Maintainers ran baselines; no leaderboard | not found ×5 | not found ×4 | not found | J-SNACKKB | ML authors (submitted to the NeurIPS 2021 benchmark track) | R10 |
| [Open Problems](https://openproblems.bio/) (single cell) | 13 single-cell tasks, 135 methods | Curated "gold-standard" datasets | Host: "run by our computational infrastructure", on Seqera Platform with AWS Batch | not found ×5 (per dataset only) | not found ×4 on the results page; resource tags (lowmem, highcpu, gpu) only | Yes: Docker plus Viash | Open Problems project | Component builders and result users | R11 to R14 |
| [MTEB](https://github.com/embeddings-benchmark/mteb) | Text embeddings: classification, clustering, retrieval, reranking, similarity | Task datasets | Submitters run their own models with the mteb tool | language yes; task type yes; protein strata do not apply | not found ×4 on fetched pages | Dockerfile in repo | embeddings-benchmark | Embedding authors | R15 |
| [LiveBench-1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2373940/) (2001) | Fold recognition, first model per target (MaxSub, LGscore) | 125 new PDB structures with no BLAST hit (E < 0.1) to earlier PDB | Host meta-server sends targets to public servers | not found · yes (easy if PDB-BLAST finds it, else hard) · not found · not found · not found | no · no · no · no | Not applicable (public web servers) | Bujnicki, Elofsson, Fischer, Rychlewski | "Non-expert" users | R16 |
| [Foldseek SCOPe benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869269/) | Structure search: sensitivity to the first false positive | SCOPe 2.01 at 40% identity, 11,211 domains; family, superfamily, fold levels | Authors ran every tool | not found · yes (family, superfamily, fold) · no (averages only) · not found · Foldseek only: "similar across all six" SCOP classes | yes (all-against-all times on one 64-core server; Dali, TM-align and CE estimated from a 10% sample) · not measured per tool · not found · not found | not found | Steinegger lab | Tool authors | R17 |
| [QfO benchmark service](http://orthology.benchmarkservice.org/proxy/) | Orthology: species-tree discordance, reference orthology, function-based, VGNC, domain architecture (FAS) | QfO reference proteomes, 78 species | Developers upload predictions | per-species counts only · not found · not found · not found · not found (FAS scores domain architecture but does not split results by it) | not found ×4 (cost called "heavy", no figures) | not found | Dessimoz lab, QfO working group, OpenEBench | Method developers and comparative genomics users | R18 to R20 |
| [CAID3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/) (2024 to 2026) | Per-residue disorder, binding, linker (F-max, APS, AUC) | DisProt; PDB-observed residues as negatives in one set | Assessors run submitted software | not found · not found · not found · the scored property itself · yes (disorder, binding, linker sets) | yes (recorded; cap 4 h per sequence) · cap 47 GB only · not found · not found | Yes: Docker or Singularity | Tosatto and Piovesan, Padova | People choosing a predictor | R21 |
| [DGEB](https://github.com/tattabio/dgeb) (2024) | 6 embedding tasks on 18 datasets, protein and DNA | Curated datasets from Bacteria, Archaea, Eukarya | Submitters run dgeb and open a PR with a JSON file | not found ×5 on fetched page | not found ×4 | Dockerfile in repo | TattaBio | Sequence-model authors | R22 |

Not in the table: Kaggle competition pages returned no text when fetched, so CAFA 5's Kaggle rules are not cited. The PROTEA model card reports a LAFA score of 0.391 against 0.381 for TransFew; that is the authors' own claim (R35).

## 2. The gap

The function benchmarks score the whole protein against GO (CAFA, LAFA, BeProf, PROBE rows) and never split results by organism, clade distance, length, disorder or feature type; LAFA's paper splits by GO aspect only. The benchmarks that do split, ProteinGym by taxon and MSA depth, Foldseek by family, superfamily and fold, LiveBench-1 by easy and hard targets, score fitness or structure, not region-level function. The benchmarks that run tools on the host (LAFA, Open Problems, CAID3) are the only ones where cost could be measured fairly, and only CAID3 records run time, as a per-sequence cap rather than a cost per 1,000 proteins. The QfO row is the nearest fit for an Earth BioGenome annotator, since it uses whole reference proteomes and now scores domain architecture (FAS), but developers upload their own predictions, so neither cost nor run conditions are known. No row answers "will this tool work on my clade, my gene type, my region length, and what will it cost" in one place.

## 3. What the truth has to be

The truth has to be region-level, dated, and split by evolutionary distance, and every report has to say which part of the truth was findable by BLAST before the test began.

**The critique.** Gillis and Pavlidis (R23) read the first CAFA results and found that methods based on sequence similarity "have a dominant effect"; many useful predictions re-found known similarity or known literature. A null method that gives every target the term "Molecular function" scored 0.19 under the original CAFA metric. They called the gold standards "deeply problematic", argued that GO is a convenient system rather than a measured biological fact, and asked that tasks which use sequence similarity be scored as a separate category.

**How each truth design answers it, and the bias it keeps.**

| Truth design | What it answers | Bias it still carries |
| --- | --- | --- |
| Swiss-Prot feature annotations (domains, motifs, transmembrane segments, binding sites) at residue level | Scores where on the protein a function sits, not only which GO term; a tool cannot win by guessing frequent root terms | Curators annotate well-studied proteins in model organisms, and many features were themselves placed by similarity to a known protein, so BLAST-like tools are rewarded for finding what curators already found by similarity |
| Pfam domain regions | Large and covers many clades; boundaries come from one profile per family | Truth is "what an HMM already found": an HMMER-based tool is scored against its own method. Use it to score other tools, never to score HMMER or InterProScan |
| Structure-derived domains (SCOPe, CATH, ECOD-style) | Truth does not come from sequence search, so a sequence tool cannot recover it by copying | Only proteins with solved or confidently predicted structures; disordered regions are absent by construction. The Foldseek row shows the same set gives no length strata (R17) |
| Time split (CAFA, LAFA, LiveBench-1) | No tool can have seen the answer; LAFA also blocks leakage by freezing containers before the cutoff (R2) | New experimental annotations favour proteins someone chose to study: model organisms, and genes that already had a guess. Gillis and Pavlidis note many CAFA targets already had electronic (IEA) annotations (R23) |
| Leave-clade-out (truth proteins whose closest training or database relative is in another clade) | Measures the case an Earth BioGenome annotator faces: a new genus with no close sequenced relative. LiveBench-1's easy and hard split by PDB-BLAST is the oldest form (R16) | Clades with few sequenced genomes also have few annotated proteins, so the hardest stratum has the smallest truth set and the widest error bars |

The common bias: truth is what was already findable. The fix is not a perfect truth set but a column that records it. For each truth item, store whether an MMseqs2 or BLAST search of the pre-cutoff database already found it. Report every score twice: on all truth, and on truth that similarity search had not found. That second number is the one Gillis and Pavlidis asked for.

## 4. The strata

Every stratum must be computable from the query proteome alone, with no truth labels, so an annotator can compute it for their own proteome and read off the matching rows. An annotator should filter on these three first: identity to the nearest Swiss-Prot protein, transmembrane or disorder class, and BUSCO fragment status with length.

| Stratum | Exact computation from the query proteome | Bins | Source for the tool |
| --- | --- | --- | --- |
| Identity to nearest annotated protein (the clade-distance proxy) | `mmseqs easy-search query.faa uniprot_sprot.fasta hits.m8 tmp -s 7.0`; keep the best hit per query by E-value; read identity from column 3 of the default output. Also store the best hit's taxon and the rank of its last common ancestor with the query species (genus, family, order, class, phylum) | no hit at E < 1e-3 · < 30% · 30–50% · 50–90% · ≥ 90% | R33 |
| Predicted-structure confidence | Mean pLDDT per protein and fraction of residues with pLDDT < 50, from the AlphaFold DB model or a local ESMFold or ColabFold run | mean > 90 · 70–90 · 50–70 · < 50 (the bands EMBL-EBI teaches; below 50 is "likely disordered or flexible") | R34 |
| Membrane topology | DeepTMHMM on every protein; keep its predicted class | alpha-helical TM · beta-barrel · neither | R31 |
| Low complexity | SEG with default settings; fraction of residues masked | < 5% · 5–20% · > 20% | not fetched; settings to be pinned in the repo |
| Length | Residue count of the protein, and separately of each scored region | protein < 100 · 100–300 · 300–1,000 · > 1,000; region < 30 · 30–100 · > 100 | none needed |
| Family taxonomic breadth | Run the Pfam HMMs on the query; map each hit family to its clan; count the superkingdoms and phyla in which that family occurs in Pfam's reference proteomes | one superkingdom · two · all three; and phylum count | not fetched |
| Gene-model completeness | `busco -i query.faa -m proteins -l <closest lineage>`; flag proteins that are BUSCO "Fragmented" hits; record the proteome's C, F, M percentages as a covariate | fragmented BUSCO hit · complete hit · not a BUSCO gene | R32 |

Why these three first. Identity to Swiss-Prot answers "how far is my clade from anything annotated", which is the Gillis and Pavlidis point turned into a filter. Membrane and disorder class answer "my gene type", two gene types that no function benchmark in section 1 reports separately. BUSCO fragment status answers a problem specific to new assemblies: a fragmented gene model loses domains, so a tool can look wrong when the gene model is wrong.

## 5. The metrics

Each tool gets an accuracy block, a trust block and a cost block, per stratum, and the page draws score against cost instead of dividing one by the other.

| Metric | What it measures | How |
| --- | --- | --- |
| Fmax | Whole-protein term prediction at the tool's best threshold | Precision and recall averaged over proteins at each threshold; take the highest harmonic mean. LAFA and BeProf both use it (R2, R6) |
| Smin, IC-weighted AUPR, information-accretion-weighted F | The CAFA family: penalise wrong specific terms more than wrong general ones | As implemented in BeProf (R6) and LAFA (R35); pin one implementation and version |
| AUPRC | Precision-recall over all thresholds, for region calls too | Area under the curve over pooled (protein, term) or (residue, label) pairs |
| IoU | Region placement | Residues in both predicted and true region divided by residues in either |
| nDO, DBD | Domain boundaries: normalized domain overlap (CASP6) and domain boundary distance (CASP8) | Definitions to be pinned from the CASP assessment papers, which I did not fetch for this doc |
| Calibration | Does the tool's own confidence mean what it says | Bin predictions by the tool's score (10 bins of equal count); plot observed precision against mean score per bin; report the mean gap |
| Confident-wrong per 1,000 proteins | The errors that reach a genome paper | False predictions above the threshold where the tool's own calibration claims 90% precision, per 1,000 query proteins |
| CPU-h, peak RAM, GPU-h per 1,000 queries | Cost | Measured by the host from the workflow trace (wall time × CPUs, peak resident memory, GPU wall time), never self-reported |
| Dollars per 1,000 queries | Cost in one unit | CPU-h × $0.00375 + GPU-h × $1.00, the Hugging Face list prices for 8 vCPU at $0.03/h and one A10G at $1.00/h (R25); the rate is stated on the page and can be changed |

**Why Fmax per second is wrong: a worked example.** The numbers below are invented to show the arithmetic.

| Tool | Fmax | CPU-h per 1,000 | GPU-h per 1,000 | $ per 1,000 | Fmax per CPU-h |
| --- | --- | --- | --- | --- | --- |
| Naive (term frequency) | 0.30 | 0.001 | 0 | 0.000004 | 300 |
| k-mer search | 0.45 | 0.5 | 0 | 0.0019 | 0.9 |
| Embedding search on GPU | 0.60 | 2 | 1 | 1.01 | 0.3 |

The ratio ranks the naive baseline first, more than 300 times above the next tool, though it predicts nothing protein-specific. Gillis and Pavlidis showed this failure for a score alone (a null method scored 0.19, R23); a ratio makes it worse, because a near-zero cost sends any baseline score to the top. The ratio also has no fixed zero: subtract the naive score to get "gain per CPU-h" and the order changes again.

A score-against-cost plot puts each tool at its (dollars, score) point and draws the line through the tools that no other tool beats on both. The annotator draws their own budget on the cost axis. For a 20,000-protein proteome, the GPU tool costs about $20 and the k-mer tool under $0.04; the annotator can see they buy 0.15 Fmax for $20, and decide. The plot keeps both numbers; the ratio throws one away.

## 6. Who uses it and who submits

Annotators read the page; tool authors submit containers; agents read the same numbers as JSON. Each needs a different slice of the same tool card.

| Reader | Their question | What the tool card must show them |
| --- | --- | --- |
| Earth BioGenome annotator | Will this tool work on my proteome, and what will it cost? | Score per stratum with the number of truth items behind it; a filter for their three strata (section 4); dollars and wall time for 20,000 proteins; the confident-wrong rate; the exact command and container to reproduce the row |
| nf-core/proteinannotator user | Which module should I switch on? | Whether the tool is already a module (v1.1.0 runs HMMER against Pfam and FunFam, InterProScan, s4pred; R30); its added cost on top of the default run; which strata it improves over InterProScan alone |
| Tool authors: Foldseek, MMseqs2, InterProScan, PLMSearch, ProtT5 or ESM nearest-neighbour search, TM-Vec, DeepGO, ProtNLM | Was my tool run fairly, and where does it win? | Version, container digest and parameters as run; the hardware it ran on (CPU model, GPU model); the strata where it sits on the score-against-cost line; for structure tools (Foldseek, TM-Vec) the fraction of queries that had a structure; for GO tools (DeepGO, ProtNLM) the region scores marked "not applicable" rather than zero |
| Agents reading JSON | Which tool should I call for this proteome and budget? | A fixed schema per row: tool, version, container digest, dataset version, stratum key and value, metric, value, 95% interval, n, CPU-h, peak RAM GB, GPU-h, dollars, rate used, date run |

One JSON row, for illustration:

```json
{"tool": "mmseqs2", "version": "<pinned>", "container": "sha256:<digest>", "dataset": "qfo-pfam-regions-v1", "stratum": {"target_species": "Danio rerio", "length_bin": "100-300"}, "metric": "iou_fmax", "value": null, "ci95": [null, null], "n_truth": null, "cpu_h_per_1000": null, "peak_rss_gb": null, "gpu_h_per_1000": 0, "usd_per_1000": null, "usd_rate": {"cpu_h": 0.00375, "gpu_h": 1.0}, "run_date": "YYYY-MM-DD"}
```

## 7. Free hosting, a two-week plan, and the second dataset

The free pieces store and show results well, but none of them runs other people's tools at full size for free except LAFA (GO only) and nf-core's AWS tests (nf-core pipelines only). So version 1 runs every tool on Sherlock and publishes static results.

| Host | What it hosts | Cost | What it cannot do | Source |
| --- | --- | --- | --- | --- |
| GitHub Pages | Static site: tables, plots, JSON | Free on public repos (cost not stated on the limits page) | Site ≤ 1 GB; soft 100 GB/month bandwidth; soft 10 builds/hour; not for commercial use; runs no tools | R24 |
| Hugging Face Spaces | Static pages free; Gradio or Docker apps on CPU Basic (2 vCPU, 16 GB, 50 GB disk that is not kept) | Static free; Gradio or Docker Spaces need a paid plan, except up to 2 ZeroGPU Spaces for free personal accounts; GPUs from $0.40/h | Free Spaces sleep when unused; disk is wiped on restart; too small to run a full proteome through most tools | R25 |
| Hugging Face Datasets | Versioned result tables (Parquet), with a viewer | Public storage free on a "best-effort" basis; private 100 GB free | Under 100k files per repo and 10k per folder recommended; 500 GB per file hard limit; runs no tools | R26 |
| Zenodo | Frozen release of each leaderboard version, with a DOI per version | Free | 50 GB and 100 files per record (one-time raise to 200 GB); a record's files cannot change; no compute | R27, R28 |
| Kaggle community competitions | Not found: the Kaggle pages returned no text | not found | not found | none |
| LAFA container submission | The host runs your public Docker or Apptainer image on its cluster after each UniProt-GOA release | No fee stated | Scores whole-protein GO only; no region truth; no strata; no resource or cost report; no stated limits on memory, GPU or time | R2, R4 |
| nf-core full-size AWS tests | A full-size run of an nf-core pipeline at each release, results on the pipeline's Results tab | Paid by nf-core from an AWS grant | Only for nf-core pipelines; per-task resources visible to the core team in Tower, not published as a cost table | R29 |

**Two-week plan for version 1, from the QfO region benchmark already run.** That pipeline (branch `olgabot/qfo-pfam-region-benchmark`) searches human queries against QfO target proteomes with phmmer, jackhmmer, MMseqs2, HHblits, Foldseek and kmerseek, transfers Pfam families from the target onto the query region, and scores calls by IoU, with hmmscan against Pfam as the ceiling. Its Nextflow trace holds measured run times.

1. Days 1–2: export `all_domain_metrics.parquet` and the trace into the long JSON-row table of section 6. Use `recall_reachable`, since a family absent from a target proteome cannot be found by any tool.
2. Days 3–4: compute strata. Clade distance here is the target species (E. coli to mouse); per query, add length, DeepTMHMM class, mean pLDDT, SEG fraction and Pfam family breadth. Identity to Swiss-Prot is useless for human queries (all are in Swiss-Prot), so use identity to the nearest protein in the target proteome instead.
3. Days 5–6: cost columns from the trace: CPU-h, peak RSS, GPU-h per 1,000 queries, dollars at the section 5 rate. Mark rows whose trace was lost or truncated as "cost not measured" rather than dropping them.
4. Days 7–8: static page on GitHub Pages: one score-against-cost plot per stratum, a filter for the three first strata, the JSON for download.
5. Days 9–10: tool cards; freeze version 1 to Zenodo and mirror the Parquet to a Hugging Face dataset.
6. Days 11–12: send each tool card to its authors (Foldseek and MMseqs2 first) to check their parameters. Check Foldseek's per-species structure coverage before publishing its row; label the kmerseek image date, since those runs predate the 0.4 scoring columns.
7. Days 13–14: write the submission guide (a container plus a Nextflow module, same as LAFA asks plus a trace) and open the PR.

**The second dataset: a time split of Swiss-Prot region features.** Take Swiss-Prot entries whose domain, region or motif features were added after a cutoff date, run every tool against the database as it was before the cutoff, and score with IoU. This fixes version 1's biggest bias, Pfam truth that rewards HMM tools for repeating their own method (section 3), and adds the time split LAFA uses. If an Earth BioGenome partner can supply a curated non-model proteome, that becomes the third dataset and the first true leave-clade-out test.

## Adversarial pass on section 1

Each cited page was fetched a second time with the row's claims listed and the instruction to disprove them. 11 cells were WRONG and are already corrected in the table above; 9 "Built for" or operator cells are UNVERIFIABLE because the page implies the audience or affiliation without stating it. H = HOLDS, W = WRONG (fixed), U = UNVERIFIABLE.

| Row | Scored | Truth | Who runs | Strata | Reports | Containers | Operator | Built for | What the second read found |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAFA | H | H | H | W | H | H | H | U | Page lists separate prokaryote and eukaryote CAFA 1 target sets, so organism is "partly", not "not found"; no organizer names on the page |
| LAFA | H | H | H | W | H | H | U | W | Paper is silent on organism and length strata, so "no" became "not found"; it names "the scientific community at large" as an audience; corresponding author is not labelled |
| PROBE | H | H | H | H | H | H | H | U | Only a qualitative runtime remark; audience implied by the "benchmark your own model" section |
| BeProf | H | H | H | H | H | H | W | U | University is inferred from csu.edu.cn email addresses, not stated |
| ProteinGym | H | H | W | H | H | H | H | U | Submitters must supply all scores; maintainers score only some popular baselines |
| PEER | H | H | H | H | H | H | H | H | Submissions are a stated future plan |
| TAPE | H | H | H | H | H | H | H | U | Page also gives a relative 10–15% speed-up remark; does not map each source to its task |
| FLIP | H | H | H | H | H | H | H | W | Paper was "submitted" to NeurIPS 2021; acceptance not stated |
| Open Problems | H | H | H | H | W | H | H | H | Home page text was cut off, so "no" became "not found"; Seqera says task leaders also get access to run benchmarks |
| MTEB | H | H | H | H | H | H | H | U | No memory or speed figures on the repository page |
| LiveBench-1 | H | H | H | H | H | H | H | H | All eight claims confirmed with quotes |
| Foldseek SCOPe | H | H | H | W | W | H | H | U | "10 d, 33 h, 6 s" came from a different search in the paper; SCOP-class result covers Foldseek only; some baseline run times were estimated from a 10% sample |
| QfO service | H | H | H | W | W | H | H | H | No clade strata or cost on either page, so "no" became "not found"; cost is called "heavy" without figures |
| CAID3 | H | H | H | H | H | H | H | H | Timings recorded and posted on the CAID results site; GPU not mentioned |
| DGEB | H | H | H | H | H | H | H | U | Covers three domains of life, but no page says results are split by them |

Not checked adversarially: the cells of sections 3 to 7. Four items there rest on pages I could not fetch and are marked so in place: the nDO and DBD formulas, SEG default settings, Pfam family taxonomic breadth, and Kaggle community-competition rules.

## References

All pages accessed October 15, 2026. Every URL below is the one fetched.

- R1. CAFA organizers. "The CAFA Challenge." biofunctionprediction.org. https://biofunctionprediction.org/cafa/.
- R2. Phan, An, Yanli Wang, Frimpong Boadu, Jianlin Cheng, Predrag Radivojac, and Iddo Friedberg. "LAFA: A Framework for Reproducible Longitudinal Assessment of Protein Function Annotation Models." arXiv:2604.20782v2, June 10, 2026. https://arxiv.org/html/2604.20782v2.
- R3. Phan, An. "CAFA_forever." GitHub repository. https://github.com/anphan0828/CAFA_forever.
- R4. Phan, An. "LAFA_container_guide." GitHub repository. https://github.com/anphan0828/LAFA_container_guide.
- R5. kansil. "PROBE." GitHub repository. https://github.com/kansil/PROBE.
- R6. CSUBioGroup. "BeProf." GitHub repository. https://github.com/CSUBioGroup/BEPROF.
- R7. OATML-Markslab. "ProteinGym." GitHub repository. https://github.com/OATML-Markslab/ProteinGym.
- R8. DeepGraphLearning. "PEER_Benchmark." GitHub repository. https://github.com/DeepGraphLearning/PEER_Benchmark.
- R9. songlab-cal. "TAPE." GitHub repository. https://github.com/songlab-cal/tape.
- R10. J-SNACKKB. "FLIP." GitHub repository. https://github.com/J-SNACKKB/FLIP.
- R11. Open Problems. Home page. https://openproblems.bio/.
- R12. Open Problems. "Philosophy." https://openproblems.bio/documentation/fundamentals/philosophy/.
- R13. Open Problems. "Batch Integration." https://openproblems.bio/benchmarks/batch_integration.
- R14. Floden, Evan, and Robrecht Cannoodt. Blog post on the OpenProblems benchmarks (title not recorded). Seqera blog, October 14, 2025. https://seqera.io/blog/single-cell-benchmarks-openproblems/.
- R15. embeddings-benchmark. "MTEB." GitHub repository. https://github.com/embeddings-benchmark/mteb.
- R16. Bujnicki, Janusz M., Arne Elofsson, Daniel Fischer, and Leszek Rychlewski. "LiveBench-1: Continuous Benchmarking of Protein Structure Prediction Servers." *Protein Science* 10, no. 2 (2001): 352–61. https://pmc.ncbi.nlm.nih.gov/articles/PMC2373940/.
- R17. van Kempen, Michel, Stephanie S. Kim, Charlotte Tumescheit, Milot Mirdita, Jeongjae Lee, Cameron L. M. Gilchrist, Johannes Söding, and Martin Steinegger. "Fast and Accurate Protein Structure Search with Foldseek." *Nature Biotechnology* 42, no. 2 (2024): 243–46. https://pmc.ncbi.nlm.nih.gov/articles/PMC10869269/.
- R18. Dessimoz Lab and Quest for Orthologs benchmarking working group. "Orthology Benchmark Service." http://orthology.benchmarkservice.org/proxy/.
- R19. Nevers, Y., T. E. M. Jones, D. Jyothi, et al. "The Quest for Orthologs Orthology Benchmark Service in 2022." *Nucleic Acids Research* 50, no. W1 (2022): W623–32. https://pmc.ncbi.nlm.nih.gov/articles/PMC9252809/.
- R20. Altenhoff, A., Y. Nevers, V. Tran, et al. "New Developments for the Quest for Orthologs Benchmark Service." *NAR Genomics and Bioinformatics* 6, no. 4 (2024): lqae167. https://pmc.ncbi.nlm.nih.gov/articles/PMC11632614/.
- R21. Mehdiabadi, M., A. Del Conte, M. V. Nugnes, M. C. Aspromonte, S. C. E. Tosatto, and D. Piovesan. "Critical Assessment of Protein Intrinsic Disorder Round 3: Predicting Disorder in the Era of Protein Language Models." *Proteins* 94, no. 1 (2026): 414–24. https://pmc.ncbi.nlm.nih.gov/articles/PMC12750029/.
- R22. TattaBio. "DGEB: Diverse Genomic Embedding Benchmark." GitHub repository. https://github.com/tattabio/dgeb.
- R23. Gillis, Jesse, and Paul Pavlidis. "Characterizing the State of the Art in the Computational Assignment of Gene Function: Lessons from the First Critical Assessment of Functional Annotation (CAFA)." *BMC Bioinformatics* 14, suppl. 3 (2013): S15. https://pmc.ncbi.nlm.nih.gov/articles/PMC3633048/.
- R24. GitHub. "GitHub Pages Limits." GitHub Docs. https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits.
- R25. Hugging Face. "Spaces Overview." Hugging Face Hub documentation. https://huggingface.co/docs/hub/spaces-overview.
- R26. Hugging Face. "Storage Limits." Hugging Face Hub documentation. https://huggingface.co/docs/hub/storage-limits.
- R27. Zenodo. "What Are the Size Limitations of Zenodo?" Zenodo support. https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo.
- R28. Zenodo. "About Records." Zenodo help. https://help.zenodo.org/docs/deposit/about-records/.
- R29. nf-core. "Bytesize 19: AWS Megatests." 2021. https://nf-co.re/events/2021/bytesize-19-aws-megatests.
- R30. nf-core. "nf-core/proteinannotator." https://nf-co.re/proteinannotator.
- R31. Hallgren, J., K. D. Tsirigos, M. D. Pedersen, J. J. Almagro Armenteros, P. Marcatili, H. Nielsen, A. Krogh, and O. Winther. "DeepTMHMM Predicts Alpha and Beta Transmembrane Proteins Using Deep Neural Networks." bioRxiv, 2022. Cited from the DTU Health Tech service page: https://services.healthtech.dtu.dk/services/DeepTMHMM-1.0/.
- R32. Ezlab. "BUSCO User Guide." https://busco.ezlab.org/busco_userguide.html.
- R33. Söding lab. "MMseqs2 Wiki." GitHub. https://github.com/soedinglab/MMseqs2/wiki.
- R34. EMBL-EBI Training. "pLDDT: Understanding Local Confidence." https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/.
- R35. XaxiPiruli. "PROTEA fullgo: First Place on the LAFA Continuous Benchmark." Hugging Face model card. https://huggingface.co/XaxiPiruli/protea-lafa-fullgo.
