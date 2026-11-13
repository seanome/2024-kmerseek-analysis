# Global SRA sequence context through Logan

Requested by Olga on 2026-10-23. This is a proposed next phase; no global collection was downloaded or indexed. The current bivalve processing and full-depth extensions remain the immediate work.

## Data and search coverage are different

The [Logan v1.2 statistics](https://github.com/IndexThePlanet/Logan/blob/main/Stats-v1.2.md) describe a December 31, 2025 SRA freeze: 38.1 million unitig accessions, 37.3 million contig accessions, 4.19 PB of compressed unitigs and 623 TB of compressed contigs. These are published dataset statistics, not our indexed counts. Later public runs and missing assemblies need a separate coverage inventory.

The [hosted Logan Search documentation](https://docs.logan-search.org/) currently describes its All group as 23.4 million samples through 2023. It searches DNA/RNA 31-mers, accepts one sequence per submission with a 2.5 kb maximum, and exposes a coverage threshold from 0.25 to 1. This service differs from both the latest downloadable dataset and our any-nonzero sourmash comparison. Verify the actual indexed manifest before reporting global coverage. Do not submit millions of extensions through the public single-query interface.

## Proposed sequence context

1. Use the completed bivalve cohort to compare original trimmed reads, Logan unitigs and Logan contigs for the SAME accessions and queries. Measure retrieval, sketch/index bytes, runtime, empty queries and exact query/run matches lost at scaled100 versus10.
2. Prefer [unitigs](https://github.com/IndexThePlanet/Logan/blob/main/Unitigs.md) for DNA occurrence discovery. Their documented preservation rule concerns 31-mers observed more than twice in the source reads; it does not establish preservation of every low-count sequence. Use [contigs](https://github.com/IndexThePlanet/Logan/blob/main/Contigs.md) for longer coding and sequence context. Assembly can remove variation, so no assembly hit cannot establish absence from the original reads.
3. Use an existing global DNA index for an initial manageable query pilot, with its own release, threshold and coverage denominator recorded. Design a supported batch workflow for all extensions before scaling. Retrieve the implicated unitigs/contigs and retain accession, assembly sequence ID and source object provenance.
4. Keep protein, Dayhoff and HP searches independent of the DNA screen. Distant protein similarities can lack exact shared DNA31 words. Compare translated-contig sketches at scaled10/100 using the same k values as the current project. Unitig fragmentation and assembly errors can change translated sketches; measure those losses in the matched pilot.
5. Evaluate Logan's [protein collection](https://github.com/IndexThePlanet/Logan/blob/main/Proteins.md) as a separate reference: it uses Prodigal predictions from v1.0 contigs through December2023, and complete predicted proteins are clustered at50% identity. It is not automatically a full animal transcript/protein catalog. Retain cluster-member/accession mappings; representative matches alone do not establish per-run occurrence.

Both global sequence coverage and four-alphabet sensitivity remain unmeasured. Retain empty sketches and missing accessions as missing/unsearchable, never biological negatives. Public context is sequence occurrence evidence, not functional validation. No global SPLASH or DE-kupl pass is proposed.

Sources were checked on 2026-10-23. Pin source commits, object versions and dataset/index manifests before execution. A full local copy of the global assembly collection is not part of this plan; first establish a workable retrieval and search strategy from measured pilot costs.
