"""
Regression tests: semantic matches must be reported consistently everywhere.

The original bug: `result["matches"]` holds one entry per (sentence, source) pair, while the
summary keeps ONE status per sentence (strongest match wins).  A sentence that is an exact
copy in source A and only a paraphrase in B and C showed up as semantic rows in the detailed
comparison but contributed 0 to "Semantic matches" in the statistics.

These tests use FakeEmbedder (hand-set cosines).  They verify counting/reporting logic only,
NOT the real MiniLM model.  Run:  python -m unittest discover -s tests -v
"""
import json
import os
import sqlite3
import unittest
from unittest import mock

from support import REAL_FPDF, FakeEmbedder, app, of_type, run_pipeline

# Seven sentences on seven different subjects; each is paraphrased in sources B and C in the same
# order (so consecutive sentences line up) and the paraphrases reuse some subject vocabulary.
USER = [
    "Glaciers carve deep valleys as they slowly move downhill under their own weight.",
    "The Roman senate gradually lost authority after the republic collapsed into civil war.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
    "Central banks raise interest rates to slow inflation in overheating economies.",
    "Coral reefs bleach when ocean temperatures stay elevated for several weeks.",
    "Printing presses accelerated the spread of literacy across medieval Europe.",
    "Antibiotics lose effectiveness when bacteria evolve resistance mechanisms.",
]
PARA_B = [
    "Moving slowly downhill, glaciers cut deep valleys because of their enormous weight.",
    "After civil war ended the republic, the Roman senate steadily lost its authority.",
    "Plants turn sunlight, water and carbon dioxide into glucose, releasing oxygen.",
    "To curb inflation in overheating economies, central banks lift interest rates.",
    "Prolonged elevated ocean temperatures make coral reefs bleach within weeks.",
    "Literacy spread rapidly through medieval Europe once printing presses appeared.",
    "Bacteria that evolve resistance mechanisms make antibiotics lose their effectiveness.",
]
PARA_C = [
    "A glacier gradually carves a deep valley as its weight pushes it downhill.",
    "Following the republic's civil war collapse, senate authority in Rome faded.",
    "Glucose and oxygen are produced from carbon dioxide, water and sunlight by plants.",
    "Raising interest rates is how central banks slow inflation when economies overheat.",
    "Several weeks of elevated ocean temperatures cause coral reefs to bleach.",
    "The printing press, widespread across medieval Europe, accelerated literacy.",
    "Resistance mechanisms evolving in bacteria reduce antibiotics effectiveness.",
]
A = ("Source A", "https://a.example/page", list(USER))        # verbatim copies
B = ("Source B", "https://b.example/page", list(PARA_B))
C = ("Source C", "https://c.example/page", list(PARA_C))


def make_model(cos_b=0.85, cos_c=0.80):
    model = FakeEmbedder()
    for u, b, c in zip(USER, PARA_B, PARA_C):
        model.link(u, b, cos_b)
        model.link(u, c, cos_c)
    return model


def scan(sentences, pages, **kw):
    cos_b, cos_c = kw.pop("cos_b", 0.85), kw.pop("cos_c", 0.80)
    return run_pipeline(make_model(cos_b, cos_c), sentences, pages, **kw)


class RootCauseReproduction(unittest.TestCase):
    """Reproduce 'statistics say 0, detail lists many' and show it is now explained."""

    def setUp(self):
        self.matches, self.rows, self.stats = scan(USER, [A, B, C])

    def test_detail_has_many_semantic_entries_but_semantic_only_is_zero(self):
        self.assertEqual(len(of_type(self.matches, app.SEMANTIC)), 14)   # 7 sentences x 2 sources
        self.assertEqual(self.stats["semantic"], 0)                       # primary status is exact
        self.assertEqual(self.stats["exact"], 7)

    def test_entries_are_unique_sentence_source_pairs_not_duplicates(self):
        sem = of_type(self.matches, app.SEMANTIC)
        self.assertEqual(len({(m["index"], m["url"]) for m in sem}), len(sem))
        self.assertEqual(len({m["index"] for m in sem}), 7)

    def test_summary_accounts_for_every_semantic_entry(self):
        s = self.stats
        self.assertEqual(s["semantic_any"], 7)                # unique sentences, not 14
        self.assertEqual(s["semantic_also"], 7)               # all hidden behind an exact match
        self.assertEqual(s["semantic_pairs"], 14)
        self.assertEqual(s["semantic_pairs"], len(of_type(self.matches, app.SEMANTIC)))
        self.assertEqual(s["semantic_any"], s["semantic"] + s["semantic_also"])
        self.assertEqual(s["semantic_any_pct"], 100.0)


class CountingRules(unittest.TestCase):
    def test_many_sources_for_one_sentence_count_as_one_sentence(self):
        matches, rows, stats = scan(USER[:1], [B, C])
        self.assertEqual(len(of_type(matches, app.SEMANTIC)), 2)
        self.assertEqual(stats["semantic"], 1)                # NOT 2
        self.assertEqual(stats["semantic_any"], 1)
        self.assertEqual(stats["semantic_pairs"], 2)
        self.assertEqual(rows[0]["semantic_sources"], 2)
        self.assertEqual(stats["semantic_pct"], 100.0)        # 1 of 1 sentence, never 200%

    def test_statuses_always_add_up_to_total_sentences(self):
        extra = USER[:3] + ["Completely unrelated sentence about volcanoes and tectonic plates today."]
        _, _, s = scan(extra, [A, B, C])
        self.assertEqual(s["exact"] + s["near_exact"] + s["semantic"] + s["none"], s["total_sentences"])
        labels, values = app._donut_data(s)
        self.assertEqual(sum(values), s["total_sentences"])   # chart agrees with summary
        self.assertEqual(dict(zip(labels, values))["Semantic only"], s["semantic"])
        self.assertEqual(s["none"], 1)

    def test_semantic_only_sentence_is_counted_in_summary_and_chart(self):
        matches, rows, stats = scan(USER[:2], [B])
        self.assertEqual(stats["semantic"], 2)
        self.assertEqual(stats["semantic_also"], 0)
        self.assertEqual(stats["semantic_any"], 2)
        self.assertEqual(app._donut_data(stats)[1][2], 2)
        self.assertTrue(all(r["status"] == app.SEMANTIC for r in rows))

    def test_threshold_is_respected(self):
        _, _, below = scan(USER[:1], [B], cos_b=0.69, threshold=0.70)
        _, _, at = scan(USER[:1], [B], cos_b=0.70, threshold=0.70)
        self.assertEqual(below["semantic"], 0)
        self.assertEqual(below["semantic_any"], 0)
        self.assertEqual(below["topical_sentences"], 0)       # below threshold = not even a candidate
        self.assertEqual(at["semantic"], 1)

    def test_exact_never_leaks_into_semantic_and_near_stays_separate(self):
        near_sentence = "The committee number 9 reviewed the quarterly irrigation budget proposal carefully."
        near_source = ["The committee number 9 reviewed the quarterly irrigation budget proposals carefully."]
        sentences = [USER[0], near_sentence]
        model = FakeEmbedder()
        matches, rows, stats = run_pipeline(model, sentences,
                                            [("S", "https://s.example", [USER[0]] + near_source)])
        by_index = {m["index"]: m["type"] for m in matches}
        self.assertEqual(by_index[0], app.EXACT)
        self.assertEqual(by_index[1], app.NEAR)
        self.assertEqual(of_type(matches, app.SEMANTIC), [])
        self.assertEqual((stats["exact"], stats["near_exact"], stats["semantic"]), (1, 1, 0))
        self.assertEqual(stats["semantic_any"], 0)


class ReportingSurfaces(unittest.TestCase):
    """The PDF text, highlighted text and old saved scans agree with the summary."""

    def setUp(self):
        self.matches, self.rows, self.stats = scan(USER, [A, B, C])

    def test_pdf_statistics_lines_explain_semantic_numbers(self):
        text = "\n".join(app.report_stat_lines(self.stats))
        self.assertIn("Exact matches: 7", text)
        self.assertIn("Near-exact matches: 0", text)
        self.assertIn("Semantic-only matches: 0", text)
        self.assertIn("Sentences with any semantic match: 7 (100.0%)", text)
        self.assertIn("14 sentence-source pair(s)", text)
        self.assertIn("7 of these are already counted as exact/near-exact", text)
        self.assertNotIn("Semantic matches: 0 (0.0%)", text)
        self.assertNotIn("Topical", text)                     # nothing topical in this scan

    def test_pdf_lines_omit_breakdown_when_no_semantic_matches(self):
        _, _, stats = scan(USER[:2], [A])
        self.assertNotIn("any semantic", "\n".join(app.report_stat_lines(stats)))

    def test_highlight_uses_primary_status_and_mentions_hidden_semantic_matches(self):
        out = app._highlight_html(self.rows)
        self.assertIn(app.HIGHLIGHT_BG[app.EXACT], out)
        self.assertNotIn(app.HIGHLIGHT_BG[app.SEMANTIC], out)
        self.assertIn("also similar in meaning to 2 other source(s)", out)
        _, rows, _ = scan(USER[:1], [B])
        self.assertIn(app.HIGHLIGHT_BG[app.SEMANTIC], app._highlight_html(rows))

    def test_old_saved_scan_without_breakdown_is_normalized_from_its_matches(self):
        legacy = {"kind": "web_scan", "status": "matches", "document_name": "old",
                  "scanned_at": "2026-01-01 00:00:00", "sentences": [dict(r) for r in self.rows],
                  "matches": self.matches, "sources": [],
                  "stats": {k: v for k, v in self.stats.items()
                            if not k.startswith(("semantic_", "topical_"))}}
        for row in legacy["sentences"]:
            row.pop("semantic_sources", None)
        fixed = app._normalize_result(legacy)["stats"]
        self.assertEqual((fixed["semantic_any"], fixed["semantic_also"], fixed["semantic_pairs"]), (7, 7, 14))
        self.assertEqual(fixed["semantic"], 0)
        self.assertEqual(fixed["topical_pairs"], 0)

    @unittest.skipUnless(os.environ.get("PLAGIARISM_DB_PATH") and os.path.exists(os.environ["PLAGIARISM_DB_PATH"]),
                         "set PLAGIARISM_DB_PATH to a real plagiarism_history.db to run this check")
    def test_real_saved_scans_reconcile(self):
        with sqlite3.connect(os.environ["PLAGIARISM_DB_PATH"]) as conn:
            details = [json.loads(r[0]) for r in conn.execute("SELECT details FROM scans WHERE details IS NOT NULL")]
        for d in details:
            s = app._normalize_result(d)["stats"]
            entries = [m for m in d["matches"] if m["type"] == app.SEMANTIC]
            self.assertEqual(s["semantic_pairs"], len({(m["index"], m["url"]) for m in entries}))
            self.assertEqual(s["semantic_any"], len({m["index"] for m in entries}))
            self.assertLessEqual(s["semantic_any"], s["total_sentences"])

    @unittest.skipUnless(REAL_FPDF, "fpdf2 not installed - PDF rendering not exercised")
    def test_generated_pdf_contains_new_statistics_and_separate_sections(self):
        import io
        from pypdf import PdfReader
        result = {"kind": "web_scan", "status": "matches", "incomplete_reason": "", "document_name": "t",
                  "scanned_at": "2026-01-01 00:00:00", "sentences": self.rows, "matches": self.matches,
                  "sources": [], "stats": self.stats, "warnings": [], "pipeline": {}}
        data = app.generate_pdf_report(result)
        text = " ".join(" ".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages).split())
        for needle in ("Semantic-only matches: 0", "Sentences with any semantic match: 7",
                       "Exact matches (", "Near-exact matches (", "Semantic matches (BERT) (14",
                       "Topical similarity, not counted (0"):
            self.assertIn(needle, text)


class DocumentComparison(unittest.TestCase):
    def test_compare_documents_uses_the_same_unique_sentence_counting(self):
        model = make_model()
        with mock.patch.object(app, "extract_sentences",
                               side_effect=lambda text, *a, **k: text.split("|")):
            result = app.compare_documents(model, "|".join(USER[:3]), "|".join(PARA_B[:3]), "A", "B")
        s = result["stats"]
        self.assertEqual(s["semantic"], 3)
        self.assertEqual(s["semantic_any"], 3)
        self.assertEqual(s["exact"] + s["near_exact"] + s["semantic"] + s["none"], s["total_sentences"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
