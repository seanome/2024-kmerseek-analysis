import unittest
from dekupl_translation import translate_context


class TranslationTests(unittest.TestCase):
    def test_boundaries(self):
        for length in (29, 30, 31):
            f, m, c = translate_context({"origin": "GCT" * length})
            plus_zero = [k for k in c if c[k]["strand"] == "+" and c[k]["frame"] == 0]
            self.assertEqual(len(plus_zero), int(length >= 30))
            if plus_zero:
                self.assertEqual(f[plus_zero[0]], "A" * length)
            self.assertTrue(all(v == "origin" for v in m.values()))

    def test_x_stops_six_frames(self):
        f, m, c = translate_context(
            {"x": "GCT" * 15 + "NNN" + "GCT" * 15 + "TAA" + "GCT" * 30}
        )
        plus_zero = [f[k] for k in c if c[k]["strand"] == "+" and c[k]["frame"] == 0]
        self.assertEqual(plus_zero, ["A" * 15 + "X" + "A" * 15, "A" * 30])
        f, m, c = translate_context({"all": "GCT" * 100})
        self.assertEqual(
            {(v["strand"], v["frame"]) for v in c.values()},
            {(s, i) for s in ("+", "-") for i in range(3)},
        )
        self.assertTrue(all(m[k] == c[k]["origin"] for k in f))
