"""
READ-ONLY diagnostic: why was one submitted sentence not reported?

Run it from the folder that contains app.py and plagiarism_history.db (it changes nothing):

    python diagnose_sentence.py                 # latest saved scan, sentence index 2 (the 3rd sentence)
    python diagnose_sentence.py 10 2            # scan id 10, sentence index 2
    python diagnose_sentence.py 10 2 pmc.txt    # also compare a page you saved by hand as .txt
                                                # (e.g. the sources that returned HTTP 403)

It re-downloads the SAME sources the scan used (their URLs are in the saved scan), saves every page's
text to ./diag_pages/ for auditing, and for the chosen sentence prints, per source:
  * the 5 highest cosine scores (the saved scan only keeps scores >= the 0.70 threshold),
  * which gate rejected / accepted the best passage, using the app's own functions.
It uses the real SentenceTransformer model, never a fake one.
"""
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import app  # noqa: E402


def diagnose(model, sentences, idx, pages, threshold=0.70, near=0.90, top=5, out=print):
    """pages = [(title, url, [source sentences])]. Returns one summary dict per source."""
    target = sentences[idx]
    out(f"\nSENTENCE {idx}: {target}")
    out(f"  words={len(target.split())}  semantic-eligible={app._semantic_eligible(target)}  "
        f"exact-eligible={app._exact_eligible(target)}")
    user_emb = app.encode_sentences(model, sentences)
    all_matches, summaries = [], []
    for title, url, src in pages:
        out("\n" + "=" * 100 + f"\nSOURCE: {title}\n{url}\n  {len(src)} sentences after extraction")
        if not src:
            out("  -> nothing to compare")
            summaries.append({"url": url, "best": None})
            continue
        src_emb = app.encode_sentences(model, src)
        sim = app.calculate_similarity(user_emb, src_emb)
        order = np.argsort(-sim[idx])[:top]
        for rank, col in enumerate(order, 1):
            flag = "  <-- best" if rank == 1 else ""
            out(f"  #{rank} cos={sim[idx][col]:.3f} src[{col}] {src[col][:150]}{flag}")
        best_cols = np.argmax(sim, axis=1)
        best = int(best_cols[idx])
        score = float(sim[idx][best])
        out(f"  best cosine for this sentence = {score:.3f}  (threshold {threshold:.2f})")
        for i in range(len(sentences)):
            out(f"    sentence {i}: best src[{best_cols[i]}] cos={sim[i][best_cols[i]]:.3f}")
        matches = app.match_against_source(model, sentences, user_emb, src, title, url, threshold, near)
        all_matches += matches
        mine = [m for m in matches if m["index"] == idx]
        if score < threshold:
            out(f"  GATE 1 (cosine >= {threshold:.2f}): REJECTED - {score:.3f} < {threshold:.2f}; "
                "context and wording checks never ran for this source")
        summaries.append({"url": url, "best": score, "matches": mine})
    app.finalize_semantic_matches(all_matches)
    out("\n" + "=" * 100 + f"\nFINAL CLASSIFICATION of sentence {idx} after finalize_semantic_matches():")
    rows = [m for m in all_matches if m["index"] == idx]
    if not rows:
        out("  no candidate in any source (every best cosine was below the threshold)")
    for m in rows:
        out(f"  {m['type']:<20} cos={m['score']:.1f}  context={m.get('context')}  "
            f"shared={m.get('shared_words')}  coverage={m.get('coverage')}\n    {m['level']}\n    {m['url']}")
    return summaries


def _load_scan(conn, scan_id):
    if scan_id is None:
        row = conn.execute("SELECT id, details FROM scans WHERE details IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
    else:
        row = conn.execute("SELECT id, details FROM scans WHERE id=?", (scan_id,)).fetchone()
    if not row:
        sys.exit("scan not found")
    return row[0], json.loads(row[1])


def main():
    args = sys.argv[1:]
    scan_id = int(args[0]) if args else None
    idx = int(args[1]) if len(args) > 1 else 2
    extra = args[2:]
    with sqlite3.connect(app.DB_PATH) as conn:
        scan_id, details = _load_scan(conn, scan_id)
    sentences = [r["text"] for r in details["sentences"]]
    threshold = details["settings"]["semantic_threshold"]
    near = details["settings"]["near_threshold"]
    print(f"scan {scan_id} ({details['scanned_at']}): {len(sentences)} sentences, threshold={threshold}")

    os.makedirs("diag_pages", exist_ok=True)
    pages = []
    for k, source in enumerate(details["sources"]):
        page = app.fetch_webpage_content(source["url"])
        if not page["ok"]:
            print(f"UNVERIFIED (no text): {source['url']}  reason: {page['reason']}")
            continue
        path = os.path.join("diag_pages", f"{k:02d}_" + re.sub(r"\W+", "_", source["title"])[:40] + ".txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(page["text"])
        src = app.rank_source_sentences(app.extract_sentences(page["text"]), sentences, app.MAX_SOURCE_SENTENCES)
        pages.append((source["title"], source["url"], src))
    for path in extra:
        with open(path, encoding="utf-8") as fh:
            pages.append((os.path.basename(path), "file://" + os.path.abspath(path), app.extract_sentences(fh.read())))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(app.MODEL_NAME)
    diagnose(model, sentences, idx, pages, threshold, near)


if __name__ == "__main__":
    main()
