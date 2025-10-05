# PE Buyer Shortlist Prototype

Input a company URL. The app fetches public page text, asks an LLM to extract structured company facts, then matches to a small PE fund catalog and explains why the funds fit.

## Run on Streamlit Community Cloud
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app pointing to `app.py`.
3. In the app's Settings → Secrets add:
```
OPENAI_API_KEY = "sk-..."
```
4. Paste a company URL and click Analyze.

## Local run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

## Notes
Public web content only.
Strict JSON schemas with Pydantic validation.
Deterministic score wrapper for explainability.
