"""Bounded checks of search input validation; no index or search is launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from search_extended import main

class ValidationTests(unittest.TestCase):
    def test_changed_extension_refused(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'splash-trimmed-all/batch-000'
            source.mkdir(parents=True)
            (source / 'validated.json').write_text(json.dumps({'output_sha256': 'wrong'}))
            (source / 'compactors.tsv').write_text('anchor\tid\tcompactor\nAAA\tx\tAAAA\n')
            with patch('sys.argv', ['search', '--root', d, '--batch', '0']):
                with self.assertRaisesRegex(ValueError, 'hash'):
                    main()
            self.assertFalse((Path(d) / 'search-trimmed-all').exists())

if __name__ == '__main__':
    unittest.main()
