# PlagiarismAI

A Streamlit app for comparing documents and checking text against web sources
with exact, near-exact, and Sentence-BERT semantic matching. Similarity is not
proof of plagiarism; review results and sources yourself.

## Run locally

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Add a Tavily or Brave API key to .env
python -m streamlit run app.py
```

The Sentence-BERT model is downloaded the first time it is needed. Search
requires a Tavily or Brave API key. Keep `.env` private; it is excluded from
version control.

## Tests

```powershell
python -m unittest discover -s tests -v
```
