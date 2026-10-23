import json
import random
import pytest
from Bio.Seq import Seq
from Bio.Data import CodonTable
from clam_context.core import (sketch, translate_segments, build_index, ProteinIndex,
    encoded_kmers, AA, DAYHOFF, HP, analyze)
from clam_context.demo import demo


def test_six_frame_coordinates_and_stops():
    dna='ATGGCTTGTTAA'+'ATGGCTTGTTAA'
    plus=[x for x in translate_segments(dna,3) if x['strand']=='+' and x['frame']==1]
    assert [x['peptide'] for x in plus]==['MAC','MAC']
    assert [(x['nt_start'],x['nt_end']) for x in plus]==[(0,9),(12,21)]
    for frag in translate_segments(str(Seq(dna).reverse_complement()),3):
        sub=str(Seq(dna).reverse_complement())[frag['nt_start']:frag['nt_end']]
        oriented=sub if frag['strand']=='+' else str(Seq(sub).reverse_complement())
        assert str(Seq(oriented).translate())==frag['peptide']
    assert not list(translate_segments('NNNNNNNNN',3))


@pytest.mark.parametrize('mode,mapping',[('dayhoff',DAYHOFF),('hp',HP)])
def test_evidence_encoding_matches_actual_sourmash(mode,mapping):
    # Exhaustively verify the equivalence classes used for highlighting against sourmash.
    for a in AA:
        for b in AA:
            same=set(sketch(a*3,mode,2,1).hashes)==set(sketch(b*3,mode,2,1).hashes)
            assert same==(mapping[a]==mapping[b]), (mode,a,b)


def test_scaled_downsampling():
    seq=''.join(random.Random(9).choices(sorted(AA),k=2000))
    for mode,k in [('protein',7),('dayhoff',12),('hp',30)]:
        ten=sketch(seq,mode,k,10);hundred=sketch(seq,mode,k,100)
        assert set(hundred.hashes)<=set(ten.hashes)
        assert ten.downsample(scaled=100)==hundred


def test_actual_search_and_no_upsampling(tmp_path):
    seq=''.join(random.Random(17).choices(sorted(AA),k=500))
    fasta=tmp_path/'db.fa';fasta.write_text('>p1 Known control\n'+seq+'\n')
    db=tmp_path/'db.sqlite';build_index(fasta,db)
    idx=ProteinIndex(db)
    try:
        for mode in ['protein','dayhoff','hp']:
            for scaled in [10,100]:
                hits=idx.search(seq,mode,min_shared=1,scaled=scaled)
                assert hits and hits[0]['containment']==1 and hits[0]['protein_id']=='p1'
                assert hits[0]['query_hashes']==hits[0]['shared_hashes']
        with pytest.raises(ValueError,match='finer'):idx.search(seq,'protein',scaled=1)
    finally:idx.close()


def test_duplicate_database_ids_fail_atomically(tmp_path):
    f=tmp_path/'db.fa';f.write_text('>p\nACDEFGHIKLMNPQ\n>p\nACDEFGHIKLMNPQ\n')
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):build_index(f,tmp_path/'bad.sqlite')
    assert not (tmp_path/'bad.sqlite').exists()
    assert not (tmp_path/'bad.sqlite.building').exists()


def test_demo_integration_and_sparse_signal(tmp_path):
    path=demo(tmp_path/'demo')
    report=json.loads((tmp_path/'demo/results/results.json').read_text())
    assert report['demo'] is True
    exact,reduced,unmatched,short=report['events']
    assert any(h['mode']=='protein' and h['protein_id']=='DEMO_EXACT' for h in exact['hits'])
    assert any(h['mode']=='dayhoff' and h['protein_id']=='DEMO_DAYHOFF' for h in reduced['hits'])
    assert not any(h['mode']=='protein' and h['protein_id']=='DEMO_DAYHOFF' for h in reduced['hits'])
    assert not unmatched['hits']
    assert short['status']=='no peptide above length cutoff'
    assert any(s['query_hashes']==0 for s in exact['sketch_signal'])
    assert '__REPORT_DATA__' not in open(path).read()


def test_ambiguous_residues_not_bridged():
    assert not sketch('ACDXEFG','protein',4,1).hashes
    assert not encoded_kmers('ACDXEFG','protein',4)
