"""Keep valid zero-match Branchwater output distinct from incomplete searches."""
import csv
import gzip
import re

FIELDS='query_name,query_md5,match_name,containment,intersect_hashes,ksize,scaled,moltype,match_md5,jaccard,max_containment,query_containment_ani'.split(',')


def save_matches(source,destination,accession,log,queries):
    # Accept the count with or without thousands separators.
    # Use the last such line, as replay_full_dna.py does.
    done=re.findall(r'DONE\. Processed ([\d,]+) search sigs',log.read_text())
    if not done or int(done[-1].replace(',',''))!=queries:
        raise ValueError('Search did not confirm all query signatures processed')
    names=set();count=0
    with source.open() as src,gzip.open(destination,'wt') as saved:
        reader=csv.DictReader(src)
        fields=reader.fieldnames
        if fields is not None and not {'query_name','match_name','intersect_hashes'} <= set(fields):
            raise ValueError('Unexpected manysearch result columns')
        writer=csv.DictWriter(saved,fieldnames=fields or FIELDS);writer.writeheader()
        for hit in reader:
            if hit['match_name']!=accession or float(hit['intersect_hashes'])<=0:
                raise ValueError('Invalid matched run or shared-hash count')
            if hit['query_name'] in names:raise ValueError('Duplicate query/run pair')
            names.add(hit['query_name']);count+=1;writer.writerow(hit)
    return names,count
