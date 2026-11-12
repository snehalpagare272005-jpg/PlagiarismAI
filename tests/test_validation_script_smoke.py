"""
SMOKE TEST ONLY: checks that validate_real_model.py runs end to end and has the right shape.

It uses FakeEmbedder, so it proves nothing about the real MiniLM model or about detection quality.
The real validation is:  python validate_real_model.py   (needs sentence-transformers).
"""
import contextlib
import io
import unittest

from support import FakeEmbedder, app

import validate_real_model as v


class ValidationScriptSmoke(unittest.TestCase):
    def test_script_runs_and_reports_with_a_fake_model(self):
        results = v.evaluate(FakeEmbedder(), v.SCENARIOS)
        self.assertEqual(len(results), len(v.SCENARIOS))
        for scenario, matches, rows, stats, oks in results:
            self.assertEqual(len(rows), len(scenario["sentences"]))
            self.assertEqual(len(oks), len(scenario["expect"]))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            v.print_report(results)
        self.assertIn("NOT an accuracy measurement", buffer.getvalue())

    def test_expectation_rules(self):
        self.assertTrue(v.meets("counted", app.SEMANTIC))
        self.assertTrue(v.meets("exact", app.EXACT))
        self.assertFalse(v.meets("exact", app.NEAR))
        self.assertTrue(v.meets("not_counted", app.NONE))
        self.assertFalse(v.meets("not_counted", app.SEMANTIC))

    def test_script_never_substitutes_a_fake_model(self):
        import inspect
        source = inspect.getsource(v.main)
        self.assertIn("SentenceTransformer(app.MODEL_NAME)", source)
        self.assertNotIn("FakeEmbedder", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
