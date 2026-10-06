# University Admissions Chatbot (RAG)

A retrieval-augmented chatbot that helps Pakistani students figure out which
field suits them and which universities they're eligible for, based on your
own PDF data (eligibility, test type, fees, deadlines, programs).

## 1. Setup

```bash
cd uni-admission-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then add your GOOGLE_API_KEY
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## 2. Add PDFs

Put each university's PDFs in its own folder under `data/pdfs/`, and name each
file after what it contains so the bot can tag it correctly:

```
data/pdfs/
  NUST/
    eligibility.pdf
    test_type.pdf
    fees.pdf
    deadlines.pdf
    programs.pdf
  FAST/
    eligibility.pdf
    test_type.pdf
    ...
```

File names don't have to be exact — the ingest script checks if a category
word (eligibility, test_type, fees, deadlines, programs) appears in the
filename. Anything else is tagged "general".

## 3. Build the knowledge base

Run this once, and again any time you add/update PDFs:

```bash
python -m app.ingest
```

This reads every PDF, splits it into chunks, tags each chunk with
`university` + `category` metadata, and saves embeddings into `chroma_db/`.

## 4. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I did pre-engineering with 85% aggregate, interested in coding. What field and university fits me?"}'
```

## 5. Add the widget to your portfolio

Copy the contents of `static/widget.html` into your portfolio's HTML, right
before `</body>`. Update `API_URL` inside the script to point at your
deployed backend once it's live.

## 6. Deploy

Free options that work well for this:
- **Render** or **Railway** — deploy the FastAPI app directly from GitHub
- Keep `chroma_db/` out of git if it gets large — instead run `python -m app.ingest`
  as a build step, or commit it if it's small enough (a few MB is fine)

## Notes on accuracy

- The bot only answers from what's in your PDFs — it's instructed not to
  invent deadlines, percentages, or test names.
- If a student asks about a university you haven't added data for, it will
  say so instead of guessing.
- Re-run `python -m app.ingest` every admission cycle when merit lists,
  deadlines, or criteria change.

## Next steps to extend

- Add a `university` filter dropdown in the widget so students can narrow
  the conversation to one school
- Log conversations (anonymized) to see what students actually ask, and use
  that to fill gaps in your PDF data
- Add simple analytics: track which universities/fields get asked about most
