"""
Shared helpers for the PlagiarismAI tests.

IMPORTANT: FakeEmbedder is NOT the SentenceTransformer model.  It returns hand-set cosine
similarities, so tests built on it verify the counting and evidence-gate LOGIC only.  They
say nothing about how well the real all-MiniLM-L6-v2 model separates copying from topical
similarity.  Use validate_real_model.py (real model, run on your machine) for that.
"""
import importlib
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stub_if_missing():
    """Real packages are always preferred; stubs are only used when an import fails."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        st = types.ModuleType("streamlit")
        st.cache_resource = lambda *a, **k: (lambda f: f)
        st.session_state = {}
        st.secrets = {}
        sys.modules["streamlit"] = st
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        plotly = types.ModuleType("plotly")
        go = types.ModuleType("plotly.graph_objects")
        plotly.graph_objects = go
        sys.modules["plotly"], sys.modules["plotly.graph_objects"] = plotly, go
    try:
        import fpdf  # noqa: F401
        return True
    except ImportError:
        fpdf = types.ModuleType("fpdf")
        fpdf.FPDF = type("FPDF", (), {})
        sys.modules["fpdf"] = fpdf
        return False


REAL_FPDF = _stub_if_missing()
app = importlib.import_module("app")


class FakeEmbedder:
    """encode() returns unit vectors; link(a, b, cos) sets the cosine between two sentences.
    Every pair that was not linked has cosine 0."""
    DIM = 4096

    def __init__(self):
        self.vectors, self._next = {}, 0

    def _fresh(self):
        v = np.zeros(self.DIM)
        v[self._next] = 1.0
        self._next += 1
        return v

    def vec(self, sentence):
        if sentence not in self.vectors:
            self.vectors[sentence] = self._fresh()
        return self.vectors[sentence]

    def link(self, original, other, cos):
        base = self.vec(original)
        self.vectors[other] = cos * base + np.sqrt(1 - cos ** 2) * self._fresh()

    def encode(self, sentences, **kwargs):
        return np.array([self.vec(s) for s in sentences])


def run_pipeline(model, sentences, pages, threshold=0.70, near=0.90):
    """
    The same steps run_scan performs after downloading: cascade per source, then
    finalize_semantic_matches over all sources, then summarize_matches.
    `pages` = [(title, url, [source sentences])].
    """
    user_embeddings = app.encode_sentences(model, sentences)
    matches = []
    for title, url, source_sentences in pages:
        matches += app.match_against_source(model, sentences, user_embeddings, source_sentences,
                                            title, url, threshold, near)
    app.finalize_semantic_matches(matches)
    rows, stats = app.summarize_matches(sentences, matches)
    return matches, rows, stats


def of_type(matches, match_type):
    return [m for m in matches if m["type"] == match_type]
