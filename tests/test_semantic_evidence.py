"""
Tests for the semantic evidence gate (semantic similarity != copying).

READ THIS FIRST
---------------
These tests use FakeEmbedder with hand-set cosine scores.  Where a test says "0.963" it is
re-using the score YOU observed from the real MiniLM model, assigned to a fake embedding, to
check what the gate does with such a score.  They verify the DECISION LOGIC only.  They do
not show that MiniLM produces those scores, nor that the gate separates copying from topical
similarity on real text.  That needs validate_real_model.py run with the real model.

The database sentences are the ones from your report (plus one constructed fourth sentence).
The gate itself contains no database-specific rules; the other tests use history, biology,
economics, ecology and medicine to show it is not tied to a subject.
"""
import unittest
from unittest import mock

from support import FakeEmbedder, app, of_type, run_pipeline
from test_semantic_reporting import PARA_B, PARA_C, USER, make_model

# ---------------------------------------------------------------------------------------
# Common technical definitions and generic statements (your database example)
# ---------------------------------------------------------------------------------------
DB_USER = [
    "A database management system is software used to store, organize, retrieve, and manage data efficiently.",
    "Relational databases use tables consisting of rows and columns.",
    "SQL is commonly used to perform operations such as inserting, updating, deleting, and retrieving records.",
    "Indexes improve query performance by reducing the amount of data that must be scanned.",   # constructed
]
P1 = ("A database management system is software that enables users to create, store, retrieve, "
      "update, and manage complex data in a structured way.")
P1B = "A DBMS is a software package for defining, creating and maintaining databases of structured data."
P1C = "Database management systems are software tools that store, retrieve and manage data for users."
P1D = "In simple terms, a database management system is software used to organise and manage data."
P2 = "Data is stored in tables consisting of rows and columns."
P2B = "A relational database organises data into tables made of rows and columns."
P2C = "Tables, which have rows and columns, are the core structure of relational databases."
P3 = ("The SQL programming language allows you to apply create, read, update, and delete operations "
      "(CRUD operations) to data and database objects.")


def topical_page(tag, count=40):
    """Filler that imitates a long page on the SAME subject: topic words recur in many sentences."""
    systems = ["database management system software", "SQL database server", "relational database engine"]
    verbs = ["stores", "retrieves", "updates", "manages", "organizes", "deletes"]
    objects = ["data", "records", "tables", "rows", "columns", "queries"]
    techs = ["SQL operations", "index structures", "transactions", "schemas"]
    return [f"The {systems[k % 3]} {verbs[k % 6]} {objects[(k // 2) % 6]} with {techs[(k // 3) % 4]} "
            f"for {tag} users, step {k}." for k in range(count)]


def page_with(tag, inserts):
    """A topical page with passages inserted at fixed, well-separated positions (never adjacent)."""
    sentences = topical_page(tag)
    for position, passage in inserts:
        sentences.insert(position, passage)
    return (f"Source {tag}", f"https://{tag.lower()}.example/page", sentences)


def db_model_and_pages():
    model = FakeEmbedder()
    model.link(DB_USER[0], P1, 0.963)     # the scores you reported from the real model
    model.link(DB_USER[1], P2, 0.776)
    model.link(DB_USER[2], P3, 0.802)
    model.link(DB_USER[0], P1B, 0.88)     # other sources say the same generic things
    model.link(DB_USER[0], P1C, 0.84)
    model.link(DB_USER[0], P1D, 0.82)
    model.link(DB_USER[1], P2B, 0.80)
    model.link(DB_USER[1], P2C, 0.78)
    pages = [page_with("S1", [(5, P1), (17, P2), (29, P3)]),
             page_with("S2", [(4, P1B), (20, P2B)]),
             page_with("S3", [(8, P1C), (25, P2C)]),
             page_with("S4", [(12, P1D)])]
    return model, pages


class CommonTechnicalDefinitions(unittest.TestCase):
    def setUp(self):
        self.model, self.pages = db_model_and_pages()
        self.matches, self.rows, self.stats = run_pipeline(self.model, DB_USER, self.pages)

    def test_the_reported_pairs_are_no_longer_counted_as_plagiarism(self):
        self.assertEqual(self.stats["semantic"], 0)
        self.assertEqual(self.stats["semantic_any"], 0)
        self.assertEqual(self.stats["overall_pct"], 0.0)       # was 100% before the gate
        self.assertEqual(self.stats["none"], 4)
        self.assertEqual(of_type(self.matches, app.SEMANTIC), [])

    def test_they_remain_visible_to_the_reviewer_as_topical_similarity(self):
        topical = of_type(self.matches, app.TOPICAL)
        self.assertEqual(len(topical), 8)                      # 4 + 3 + 1 sentence-source pairs
        top = max(topical, key=lambda m: m["score"])
        self.assertEqual(top["score"], 96.3)                   # score and passage are still reported
        self.assertEqual(top["matched"], P1)
        self.assertIn("not counted", top["level"])

    def test_multiple_sources_do_not_multiply_sentences(self):
        self.assertEqual(self.stats["topical_sentences"], 3)   # sentences 1-3, not 8
        self.assertEqual(self.stats["topical_pairs"], 8)
        self.assertEqual(self.stats["exact"] + self.stats["near_exact"] + self.stats["semantic"]
                         + self.stats["none"], self.stats["total_sentences"])

    def test_standard_statement_wording_names_the_reason(self):
        rows_by_sentence = {m["index"]: m["level"] for m in of_type(self.matches, app.TOPICAL)}
        self.assertIn("Standard statement", rows_by_sentence[1])   # rows/columns: found in 3 sources
        self.assertIn("Related topic only", rows_by_sentence[2])   # SQL/CRUD: one source, no evidence

    def test_old_behaviour_would_have_flagged_everything(self):
        """Documents the problem: with the gate bypassed, every cosine >= 0.70 candidate counted."""
        user_emb = app.encode_sentences(self.model, DB_USER)
        raw = []
        for title, url, src in self.pages:
            raw += app.match_against_source(self.model, DB_USER, user_emb, src, title, url, 0.70, 0.90)
        _, stats = app.summarize_matches(DB_USER, raw)        # candidates are still typed SEMANTIC here
        self.assertEqual(stats["semantic"], 3)

    def test_pdf_statistics_explain_the_uncounted_similarity(self):
        text = "\n".join(app.report_stat_lines(self.stats))
        self.assertIn("Topical similarity (NOT counted above): 3 sentence(s)", text)
        self.assertIn("8 sentence-source pair(s)", text)
        self.assertIn("Overall similarity: 0.0%", text)


class GenericStatements(unittest.TestCase):
    def test_statement_found_in_many_sources_is_not_counted_even_with_shared_wording(self):
        """Commonness veto is domain-agnostic: a stock sentence about boiling water, 3 sources."""
        user = ["Distilled water boils at one hundred degrees Celsius under standard atmospheric pressure."]
        variants = ["Under standard atmospheric pressure, distilled water boils at one hundred degrees Celsius.",
                    "At standard atmospheric pressure distilled water reaches its boiling point at 100 degrees Celsius.",
                    "Distilled water boils at 100 degrees Celsius when atmospheric pressure is standard."]
        model = FakeEmbedder()
        for v in variants:
            model.link(user[0], v, 0.93)
        pages = [(f"P{k}", f"https://p{k}.example", [v]) for k, v in enumerate(variants)]
        _, _, three = run_pipeline(model, user, pages)
        self.assertEqual((three["semantic"], three["topical_sentences"]), (0, 1))
        _, _, two = run_pipeline(model, user, pages[:2])         # same wording, only 2 sources
        self.assertEqual((two["semantic"], two["topical_sentences"]), (1, 0))

    def test_veto_threshold_is_the_documented_constant(self):
        self.assertEqual(app.COMMON_SOURCE_MIN, 3)


class GenuineParaphrase(unittest.TestCase):
    def test_paraphrase_with_shared_distinctive_wording_is_counted_in_any_subject(self):
        matches, rows, stats = run_pipeline(make_model(), USER, [("B", "https://b.example", list(PARA_B))])
        self.assertEqual(stats["semantic"], 7)                  # history, biology, economics, ecology ...
        self.assertEqual(stats["topical_sentences"], 0)

    def test_single_sentence_with_shared_distinctive_wording_is_counted(self):
        matches, _, stats = run_pipeline(make_model(), USER[:1], [("B", "https://b.example", [PARA_B[0]])])
        self.assertEqual(stats["semantic"], 1)
        self.assertIn("shares distinctive wording", matches[0]["level"])

    def test_context_alone_is_enough_when_the_wording_is_completely_different(self):
        user, source = CTX_USER, CTX_SOURCE
        for u, s in zip(user, source):                           # guard: the fixture shares no distinctive words
            pool = user + source
            df, n = app.document_frequencies(pool)
            self.assertEqual(app.lexical_evidence(u, s, df, n)[0], [])
        model = FakeEmbedder()
        for u, s in zip(user, source):
            model.link(u, s, 0.80)
        pad = [f"Unrelated filler sentence number {k} about gardening and soil." for k in range(6)]
        matches, _, stats = run_pipeline(model, user, [("S", "https://s.example", pad + source)])
        self.assertEqual(stats["semantic"], 3)
        self.assertTrue(all("consecutive sentences" in m["level"] for m in of_type(matches, app.SEMANTIC)))

    def test_context_beats_the_commonness_veto(self):
        model = FakeEmbedder()
        for u, s in zip(CTX_USER, CTX_SOURCE):
            model.link(u, s, 0.80)
        pages = [(f"S{k}", f"https://s{k}.example", list(CTX_SOURCE)) for k in range(4)]   # found in 4 sources
        _, _, stats = run_pipeline(model, CTX_USER, pages)
        self.assertEqual(stats["semantic"], 3)
        self.assertEqual(stats["semantic_pairs"], 12)           # 12 rows, but 3 sentences counted

    def test_known_limitation_lone_reworded_sentence_without_evidence_is_not_counted(self):
        """Trade-off made deliberately: no neighbours + no shared distinctive wording = topical only.
        It is still listed for manual review, never silently dropped."""
        model = FakeEmbedder()
        model.link(CTX_USER[0], CTX_SOURCE[0], 0.90)
        matches, _, stats = run_pipeline(model, CTX_USER[:1], [("S", "https://s.example", [CTX_SOURCE[0]])])
        self.assertEqual(stats["semantic"], 0)
        self.assertEqual([m["type"] for m in matches], [app.TOPICAL])


CTX_USER = [
    "The harbor town prospered because merchants could unload cargo cheaply.",
    "Later, silting of the estuary forced ships to anchor farther offshore.",
    "Eventually the community shrank as trade moved to deeper ports.",
]
CTX_SOURCE = [
    "Inexpensive docking attracted traders, which made the seaside settlement wealthy.",
    "Sediment accumulating in the river mouth pushed vessels to moor further from land.",
    "Gradually inhabitants departed, since commerce shifted toward harbours with greater depth.",
]


class ExactCopying(unittest.TestCase):
    def test_verbatim_copy_is_exact_and_never_topical_or_semantic(self):
        model, pages = db_model_and_pages()
        sentences = [DB_USER[0], DB_USER[1]]
        pages = [page_with("X", [(7, DB_USER[1])]), pages[0]]   # sentence 2 copied verbatim into one source
        matches, rows, stats = run_pipeline(model, sentences, pages)
        self.assertEqual(rows[1]["status"], app.EXACT)
        self.assertEqual(stats["exact"], 1)
        copies = [m for m in matches if m["index"] == 1 and m["url"] == pages[0][1]]
        self.assertEqual([m["type"] for m in copies], [app.EXACT])   # exact stays separate from semantic

    def test_exact_and_near_exact_unchanged_by_the_gate(self):
        near_user = "The committee number 9 reviewed the quarterly irrigation budget proposal carefully."
        near_src = "The committee number 9 reviewed the quarterly irrigation budget proposals carefully."
        sentences = [USER[0], near_user]
        matches, _, stats = run_pipeline(FakeEmbedder(), sentences,
                                         [("S", "https://s.example", [USER[0], near_src])])
        self.assertEqual((stats["exact"], stats["near_exact"], stats["semantic"], stats["topical_sentences"]),
                         (1, 1, 0, 0))


class UnrelatedContent(unittest.TestCase):
    def test_unrelated_text_produces_no_candidates_at_all(self):
        matches, _, stats = run_pipeline(FakeEmbedder(), DB_USER, [page_with("U", [])])
        self.assertEqual(matches, [])
        self.assertEqual((stats["overall_pct"], stats["none"], stats["topical_sentences"]), (0.0, 4, 0))

    def test_just_below_threshold_is_not_even_topical(self):
        model = FakeEmbedder()
        model.link(DB_USER[1], P2, 0.69)
        matches, _, stats = run_pipeline(model, DB_USER[1:2], [("S", "https://s.example", [P2])])
        self.assertEqual(matches, [])


class EvidenceHelpers(unittest.TestCase):
    def test_topic_words_are_not_distinctive_but_rare_words_are(self):
        pool = topical_page("T") + [DB_USER[1], P2]
        df, n = app.document_frequencies(pool)
        shared, coverage = app.lexical_evidence(DB_USER[1], P2, df, n)
        for topic_word in ("table", "row", "column", "data", "databas", "relational"):
            self.assertNotIn(topic_word, shared)               # these run through the whole page
        self.assertLess(len(shared), app.LEXICAL_MIN_SHARED)   # only "consist" is rare: not enough alone
        rare_a = "The zebrafish genome encodes unusual transposon families."
        rare_b = "Unusual transposon families are encoded in the zebrafish genome."
        df, n = app.document_frequencies(pool + [rare_a, rare_b])
        shared, coverage = app.lexical_evidence(rare_a, rare_b, df, n)
        self.assertGreaterEqual(len(shared), 3)
        self.assertGreater(coverage, 0.5)

    def test_stemming_groups_word_forms_without_merging_unrelated_words(self):
        self.assertEqual(len({app._stem(w) for w in ("store", "stores", "storing", "stored")}), 1)
        self.assertEqual(len({app._stem(w) for w in ("row", "rows")}), 1)
        self.assertNotEqual(app._stem("data"), app._stem("date"))


class RunScanIntegration(unittest.TestCase):
    """Whole run_scan with search and downloads mocked; only the network boundary is fake."""

    def _scan(self, model, text, pages):
        hits = [{"title": title, "url": url, "snippet": "", "found_by": [app.QUERY_EXACT], "best_rank": k}
                for k, (title, url, _) in enumerate(pages)]
        search = {"results": hits, "errors": [], "notes": [], "engines": ["fake"], "provider": "fake",
                  "provider_label": "Fake search",
                  "queries": [{"query": "q", "type": app.QUERY_EXACT, "ok": True, "hits": len(hits), "engine": "fake"}]}
        texts = {url: " ".join(src) for _, url, src in pages}
        fetch = lambda url, *a, **k: {"ok": True, "text": texts[url], "page_title": "", "reason": "",
                                      "kind": "html", "pdf_url": "", "note": ""}
        settings = {"provider": "auto", "semantic_threshold": 0.70, "near_threshold": 0.90, "max_queries": 4,
                    "results_per_query": 5, "max_pages": 10, "max_sentences": 200}
        with mock.patch.object(app, "search_web_sources", return_value=search), \
             mock.patch.object(app, "fetch_webpage_content", side_effect=fetch):
            return app.run_scan(model, text, "test", settings, lambda fraction, message: None)

    def test_common_definitions_scan_reports_no_matches_but_keeps_topical_rows(self):
        model, pages = db_model_and_pages()
        result = self._scan(model, " ".join(DB_USER), pages)
        self.assertEqual(result["status"], "no_matches")
        s = result["stats"]
        self.assertEqual((s["overall_pct"], s["semantic"], s["topical_sentences"], s["topical_pairs"]),
                         (0.0, 0, 3, 8))
        self.assertTrue(all(src["matches"] == 0 for src in result["sources"] if src["verified"]))
        self.assertEqual(result["pipeline"]["sources_with_matches"], 0)
        self.assertEqual(len(of_type(result["matches"], app.TOPICAL)), 8)

    def test_genuine_paraphrase_scan_counts_sentences_once_and_per_source_matches(self):
        model = make_model()
        pages = [("B", "https://b.example/page", list(PARA_B)), ("C", "https://c.example/page", list(PARA_C))]
        result = self._scan(model, " ".join(USER), pages)
        s = result["stats"]
        self.assertEqual(result["status"], "matches")
        self.assertEqual((s["semantic"], s["total_sentences"], s["overall_pct"]), (7, 7, 100.0))
        self.assertEqual(s["semantic_pairs"], 14)               # 14 rows, 7 sentences
        self.assertEqual([src["matches"] for src in result["sources"]], [7, 7])
        self.assertEqual(result["pipeline"]["sources_with_matches"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
