# Data sources

One section per notebook. Each dataset has its download URL, the date it was downloaded,
its size and a checksum. A field that is not known yet says so; nothing here is
estimated.

## Notebook 352: kmerseek index and query time against database size

### UniParc active sequences, UniProt release 2026_03

- URL: https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/fasta/active/
  (200 files, `uniparc_active_p1.fasta.gz` to `uniparc_active_p200.fasta.gz`)
- Checksums: the md5 of every file is in that folder's `RELEASE.metalink`. The copy read on
  2026-10-20 is committed as `notebooks/352_data/uniparc_active_RELEASE.metalink`. The
  setup job fetches the metalink again and stops if it differs from the committed copy.
- Metalink copy: 147_591 bytes, sha256 `cc1f9d09e20e53901cfc2d3a20eac17cb3bced398fae03e84a3fb9d9aea21fc2`,
  md5 `6879cce235aaaaede98907c016c6d845`.
- Size: 248_676_531_083 bytes in total over the 200 files, summed from the metalink's
  `<size>` entries (read 2026-10-20, not yet downloaded).
- Release: 2026_03, metalink `<version>` 2026_03. UniProt's release notes for 2026_03
  (https://ftp.uniprot.org/pub/databases/uniprot/current_release/relnotes.txt, read
  2026-10-20) give 1,128,904,654 active UniParc entries.
- Date downloaded: not yet. The setup job (`notebooks/352_setup.sbatch`) writes the date,
  the byte count and the md5 check of every file to `results/download_record.tsv` and
  `results/parts_md5.tsv`; they are copied into `notebooks/352_data/sherlock/results/` and
  this entry is updated from them.
- Residue count: not known until the setup job's first pass
  (`results/histogram_stats.json`).

### Not downloaded, read for the choice of source

- UniRef100 2026_03, https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz:
  63_187_449_732 bytes, md5 `003fda83e4f34ffc6e3d72b334466f51` (from its RELEASE.metalink,
  read 2026-10-20), 221,143,266 sequences (uniref100.release_note). Not used: its residue
  count is not published and may be under 100 billion.

### kmerseek image and the K used for E-values

- Image: `docker.io/olgabot/kmerseek:0.4.0-rc5`, the Sherlock copy at
  `$SCRATCH/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img` (227_278_848 bytes,
  pulled 2026-09-21). Its sha256 is written by each size job to `results/size_*/node.tsv`.
- `--ka-k 0.00020260135043837733`: column `k` of
  `$SCRATCH/invertebrate-dark-set/results-0.4/kmerseek_index/minus_Chromadorea.hp_pbotc_1st_ed2.k19.lcfalse.kmerseek.rocksdb/ka_survival.C1.59.csv`
  on Sherlock, read 2026-10-20 (fit on 500 database sequences, penalty 1.59, give-up
  margin 6.36).
