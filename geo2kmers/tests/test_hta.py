from geo2kmers.hta import genes_in, non_coding_names

CODING = "NM_003319 // TTN // titin // 2q31.2 // 7273 /// ENST00000589042 // TTN // titin // 2q31.2 // 7273"
LINC = "--- // --- // linc-BRD9-1 chr5:-:1040857-1048427|linc-BRD9-1 chr5:-:1041183-1045593 // --- // ---"
ENSEMBL = "--- // --- // cdna:known chromosome:GRCh37:15:102313010:102334505:1 gene:ENSG00000259660 gene_biotype:pseudogene // --- // ---"


def test_genes_in_reads_the_symbol_once():
    assert genes_in(CODING) == ["TTN"]


def test_genes_in_skips_missing_symbols():
    assert genes_in(LINC) == []


def test_non_coding_names_reads_lincrna_and_ensembl_ids():
    assert non_coding_names(LINC) == ["linc-BRD9-1"]
    assert non_coding_names(ENSEMBL) == ["ENSG00000259660"]
    assert non_coding_names("---") == []
