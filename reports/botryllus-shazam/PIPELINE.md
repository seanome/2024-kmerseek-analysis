# Programs, inputs, and outputs

Grey = input files. Blue = programs. Green = output files.
This diagram describes the intended full analysis, not completed full-depth results.
Current completion records are in README.md and HANDOFF.md.

```mermaid
flowchart TB
  subgraph READS[Study identity and full-depth preprocessing]
    P[Paper and Dataset S1 XLSX]:::input --> M[Python / openpyxl metadata join and design audit]:::program
    E[ENA run TSV and BioSample XML]:::input --> M
    M --> S[samples.tsv: genotype, treatment, exact time, inclusion, unresolved pairing]:::output
    E --> C[curl: resumable full paired-run downloads]:::program
    C --> R[R1.fastq.gz and R2.fastq.gz per run]:::output
    R --> V[MD5 and bytes; gzip; Rust paired-FASTQ validator]:::program
    V --> G[Per-run validated.json: pair and base totals]:::output
    G --> NF[nf-core/rnaseq 3.27.0 / Nextflow 25.10.4<br/>fq lint; FastQC; Trim Galore 2.3.0; Salmon strand estimation]:::program
    S --> NF
    NF --> T[Trimmed paired FASTQs; QC; strand estimates; execution trace]:::output
    T --> TV[Rust paired-FASTQ validation and SHA-256]:::program
    TV --> SH[shared-trimmed and validated input manifest]:::output
  end
  subgraph DISCOVERY[Separate discovery evidence]
    SH --> SP[Upstream SPLASH 2.11.9: DNA anchor31, target31, gap0]:::program
    SP --> A[All tested anchors, statistics, per-sample target counts]:::output
    A --> Q[Select every pval_opt_corrected less than 0.05; no cap]:::program
    Q --> CO[Upstream Compactors 2.11.9]:::program
    CO --> X[DNA extensions FASTA; parent/child TSV; support; terminal flags]:::output
    SH --> DK[Upstream DE-kupl: nomask and supported counting stages]:::program
    DK --> K[Original k-mer count matrix and normalization records]:::output
    S --> D[Experimental-unit and estimable-model check]:::program
    K --> DM[Design-appropriate abundance model only after design passes]:::program
    D --> DM
    DM --> DA[Abundance statistics and assembled DNA contigs<br/>External analysis distinguished from upstream DE-kupl]:::output
  end
  subgraph PROTEIN[Primary protein retrieval with Sourmash]
    X --> TR[Six-frame translation; split stops and ambiguity; minimum 20 aa]:::program
    DA --> TR
    TR --> PE[Peptide fragments FASTA and DNA-coordinate membership TSV]:::output
    PE --> SQ[sourmash 4.9.4 sketch protein<br/>Protein k7; Dayhoff k12; HP k30<br/>Scaled 5 then downsample to 10 and 100]:::program
    BP[Original new Botryllus proteome: all 45,339 records]:::input --> SR[sourmash 4.9.4 reference sketches and searchable indexes<br/>Matching encoding, k, seed, and scale]:::program
    SW[Checksummed reviewed Swiss-Prot FASTA]:::input --> SR
    SQ --> SS[sourmash search --containment<br/>Retain shared hashes, ties, empty sketches and denominators]:::program
    SR --> SS
    SS --> HIT[Protein candidates TSV/JSON; database, encoding, scale, ambiguity]:::output
    HIT -. follow-up validation .-> AL[DIAMOND alignment; HMMER 3.4 / Pfam domains<br/>eggNOG or structure search only when verified and justified]:::program
    AL --> EV[Separate sequence, family/domain, localization and reliability evidence]:::output
    HIT --> EV
  end
  subgraph REFERENCE[Reference-based comparison and SPLASH+]
    GEN[Version-matched genome FASTA and GTF; paper version unresolved]:::input --> PL[Actual SPLASH+ R scripts<br/>STAR; Bowtie2; SAMtools; BEDTools<br/>Matched index, exon and splice-site files required]:::program
    X --> PL
    PL --> CL[Classified anchors; alignments; junction evidence]:::output
    GEN --> RF[STAR; HTSeq-count; design-appropriate DESeq2 model]:::program
    SH --> RF
    D --> RF
    RF --> GE[Reference gene counts and abundance results]:::output
    P --> CP[Sequence-based comparison; published nominal and adjusted-P lists kept separate]:::program
    EV --> CP
    DA --> CP
    A --> CP
    CL --> CP
    GE --> CP
    CP --> CMP[Cross-method agreement, conflict, unassigned sequences, published-direction recovery]:::output
  end
  subgraph ARCHIVE[Public tunicate occurrence search]
    TI[ENA tunicate RNA-Seq manifest and checksums]:::input --> BD[Bounded downloads and input validation; measured pilot before scaling]:::program
    BD --> AS[sourmash sketch dna and sketch translate<br/>DNA k31; protein k7; Dayhoff k12; HP k30<br/>Scaled 5 to 10 and 100; mates remain separate records]:::program
    AS --> IX[sourmash index: searchable per-run collections and metadata]:::program
    IX --> AR[Validated archive signatures and indexes]:::output
    AR --> LS[sourmash search and sequence confirmation]:::program
    X --> LS
    PE --> LS
    LS --> CT[Run/study occurrence TSV; self-study flags; ties; missing coverage]:::output
    CI[Chordate metadata only]:::input --> COST[Inventory and compute/storage cost estimation; no chordate indexing]:::program
  end
  subgraph DELIVERY[Unknown-gene investigation and evidence explorer]
    EV --> CR[Select reliable nonredundant candidates and controls under frozen rules]:::program
    CMP --> CR
    CT --> CR
    CR --> CARDS[Blinded review cards; unknown-status diagnosis; claim confidence<br/>Proposed model changes and testable experiments]:::output
    CARDS --> WEB[Interactive evidence explorer: pilot/full-depth, sequence search, lengths, provenance, stage status]:::program
    WEB --> OUT[JSON; TSV; DNA and peptide FASTA; methods; logs; resource manifests; restart handoff]:::output
  end
  classDef input fill:#eeeeee,stroke:#666666,color:#111111;
  classDef program fill:#deebf7,stroke:#1e5a85,color:#111111;
  classDef output fill:#dcf0e6,stroke:#24704a,color:#111111;
```

All significant SPLASH anchors, not just the old top-300 subset, are extended in new runs.
SPLASH composition significance is not treatment-associated abundance.
Sourmash is primary protein retrieval; alignment and domains check candidate assignments.
Six-frame read translations are not complete predicted proteins.
Archive sequence occurrence is not independent biological validation or a function label.
