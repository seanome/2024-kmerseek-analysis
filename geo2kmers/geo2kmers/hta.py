"""Read gene names out of the gene_assignment column of an Affymetrix HTA 2.0 annotation (GPL17586)."""

import re


def genes_in(assignment):
    """The gene symbols in one gene_assignment cell, whose parts are split by /// and fields by //."""
    names = [part.split(" // ")[1].strip() for part in assignment.split(" /// ") if part.count(" // ") >= 1]
    return sorted({n for n in names if n and n != "---"})


def non_coding_names(assignment):
    """lincRNA names (linc-BRD9-1) and Ensembl gene ids in the third field, where a cluster with no gene symbol keeps them."""
    described = " ".join(part.split(" // ")[2] for part in assignment.split(" /// ") if part.count(" // ") >= 2)
    return sorted(set(re.findall(r"linc-[A-Za-z0-9.-]+[A-Za-z0-9]", described)) | set(re.findall(r"ENSG\d+", described)))
