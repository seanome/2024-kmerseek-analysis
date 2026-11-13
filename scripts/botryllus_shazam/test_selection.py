import tempfile
import unittest
from pathlib import Path
from select_anchors import select

class SelectionTests(unittest.TestCase):
    def test_threshold_order_and_empty(self):
        for text, expected in [('CCC\t0.01\nAAA\t0.01\nGGG\t0.05\n', ['AAA', 'CCC']), ('', [])]:
            with tempfile.TemporaryDirectory() as d:
                source = Path(d) / 'input.tsv'
                source.write_text('anchor\tpval_opt_corrected\n' + text)
                out = Path(d) / 'out'
                report = select(source, out)
                self.assertEqual(report['selected_anchors'], len(expected))
                self.assertEqual((out / 'selected_anchors.tsv').read_text().splitlines()[1:], expected)

    def test_invalid_inputs(self):
        for row in ['\t0.01\n', 'XYZ\t0.01\n', 'AAA\tnan\n', 'AAA\t0.01\nAAA\t0.02\n']:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as d:
                source = Path(d) / 'input.tsv'
                source.write_text('anchor\tpval_opt_corrected\n' + row)
                with self.assertRaises(ValueError):
                    select(source, Path(d) / 'out')

if __name__ == '__main__':
    unittest.main()
