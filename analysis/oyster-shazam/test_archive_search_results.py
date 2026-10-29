import gzip
from pathlib import Path
import tempfile
import unittest
from archive_search_results import save_matches


class SearchResultsTest(unittest.TestCase):
    def test_empty_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'in').write_text('');(p/'log').write_text('DONE. Processed 2 search sigs')
            self.assertEqual(save_matches(p/'in',p/'out.gz','RUN',p/'log',2),(set(),0))
            with gzip.open(p/'out.gz','rt') as f:self.assertTrue(f.read().startswith('query_name,'))

    def test_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'in').write_text('');(p/'log').write_text('Processed 1 search sigs')
            with self.assertRaises(ValueError):save_matches(p/'in',p/'out.gz','RUN',p/'log',2)

    def test_last_done_line_used(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'in').write_text('');(p/'log').write_text('DONE. Processed 2 search sigs\nDONE. Processed 1 search sigs\n')
            with self.assertRaises(ValueError):save_matches(p/'in',p/'out.gz','RUN',p/'log',2)

    def test_positive_and_invalid_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'log').write_text('DONE. Processed 2 search sigs')
            header='query_name,match_name,intersect_hashes\n'
            (p/'in').write_text(header+'q,RUN,1\n')
            self.assertEqual(save_matches(p/'in',p/'out.gz','RUN',p/'log',2),({'q'},1))
            for rows in ['q,OTHER,1\n','q,RUN,0\n','q,RUN,1\nq,RUN,1\n']:
                (p/'in').write_text(header+rows)
                with self.assertRaises(ValueError):save_matches(p/'in',p/'out.gz','RUN',p/'log',2)


if __name__=='__main__':unittest.main()
