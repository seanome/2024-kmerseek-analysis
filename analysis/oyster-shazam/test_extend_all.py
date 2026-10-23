import tempfile
import json
from unittest.mock import patch
import unittest
from pathlib import Path
from extend_all_significant import main, select


class SelectionTests(unittest.TestCase):
    def selection(self, rows):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'scores.tsv'
            p.write_text('anchor\tpval_opt_corrected\n'+rows)
            return select(p)

    def test_strict_threshold_and_stable_order(self):
        self.assertEqual(self.selection('CCC\t0.049\nAAA\t0\nGGG\t0.05\nTTT\t0.9\n'), ['AAA','CCC'])

    def test_empty(self):
        self.assertEqual(self.selection(''), [])

    def test_invalid_and_duplicate(self):
        for rows in ['AAA\tnan\n','AAA\t-1\n','AAA\t1.1\n','AAA\t0\nAAA\t0.01\n', '\t0.01\n', 'XYZ\t0.01\n']:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.selection(rows)

    def test_incompatible_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            source = root / 'splash-trimmed'
            source.mkdir()
            (source / 'splash.after_correction.scores.tsv').write_text(
                'anchor\tpval_opt_corrected\nAAA\t0.01\n')
            out = root / 'splash-trimmed-all/batch-000'
            out.mkdir(parents=True)
            (out / 'validated.json').write_text(json.dumps({'threshold': 0.1}))
            with patch('sys.argv', ['extend', '--root', t, '--batch', '0']):
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    main()


if __name__ == '__main__':
    unittest.main()
