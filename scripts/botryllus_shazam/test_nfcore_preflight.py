"""Regression for the actual duplicate option that cancelled the full-depth run."""

import unittest
from nfcore_preflight import trim_tokens


class GeneratedCommandTests(unittest.TestCase):
    def test_generated_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Repeated singleton"):
            trim_tokens(
                "trim_galore --quality 20 --no_poly_g --no_poly_g --paired a.fq b.fq\n"
            )

    def test_corrected_command(self):
        argv = trim_tokens(
            "trim_galore \\\n --quality 20 --no_poly_g \\\n --paired --gzip a.fq b.fq\n"
        )
        self.assertEqual(argv.count("--no_poly_g"), 1)

    def test_missing_command_is_not_success(self):
        with self.assertRaises(ValueError):
            trim_tokens("echo stub\n")

    def test_other_singleton_duplicate_is_rejected(self):
        with self.assertRaises(ValueError):
            trim_tokens("trim_galore --no_poly_g --length 62 --length 20 a.fq\n")
