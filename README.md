---
title: MedDRA Coding Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# 🩺 MedDRA Coding Assistant

Turn natural language into MedDRA codes. Type a symptom, product complaint,
abbreviation, or even a misspelling — get the closest **LLT** and **PT** terms,
ranked with confidence scores. No narratives, no reports — just fast, accurate
coding.

- **Hybrid search**: semantic AI (ONNX embeddings) + fuzzy/lexical (RapidFuzz) + exact.
- **Optional AI accuracy layer**: interprets lay idioms and re-ranks — grounded, never invents codes.
- **In-memory & fast** (~a few ms/query after warmup). No OpenSearch, no database, no PyTorch.
- **Three ways to use it**: web page, REST API, Telegram bot.
- **Deploy-friendly**: every file is under GitHub's 25 MB limit; light Docker image.

### Examples

| You type | Top MedDRA suggestion |
|---|---|
| `bleeding` | Haemorrhage |
| `SOB` | Dyspnoea |
| `drug not working` | Drug ineffective (family) |
| `tablet is hard` | Product physical issue |
| `hedache` (typo) | Headache |
| `high bp` | Blood pressure increased / Hypertension |

---

## Project layout
```
app/            FastAPI app, search engine, Telegram bot
data/           prebuilt index (terms + embeddings) — shipped, ready to run
static/         web search UI
scripts/        build_index.py — regenerate the index from MEDDRA.xlsx (offline)
```
The running service needs only `app/`, `data/`, and `static/`.
**MEDDRA.xlsx is NOT needed at runtime** — only to rebuild the index.

---

## Run locally
```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000

- Web UI: `/`
- API: `GET /api/code?q=bleeding` or `POST /api/code {"text": "...", "top_k": 8}`
- Docs: `/docs`
- Health: `/health`

---

## Deploy to Render (GitHub → Render)
1. Push this repo to GitHub. **Do not commit your `.env`** — it's already
   git-ignored, so real secrets stay local.
2. Render → **New** → **Blueprint** → connect the GitHub repo (uses `render.yaml`).
3. In the Render dashboard → your service → **Environment**, set `AI_API_KEY`
   (and `TELEGRAM_BOT_TOKEN` if you want the bot). `AI_API_BASE_URL` and
   `AI_MODEL` already default correctly from `render.yaml`.
4. Deploy. Render calls `GET /health` to confirm the service is up, then the
   web UI is live at your `*.onrender.com` URL.

---

## Deploy to Railway
1. Push this repo to GitHub (all files are < 25 MB, so web upload works).
2. Railway → **New Project** → **Deploy from GitHub repo** → select it.
   Railway auto-detects the `Dockerfile`.
3. (Optional) add `TELEGRAM_BOT_TOKEN` in **Variables** to enable the bot.
4. Open the generated domain → the web UI loads. Done.

Boot is fast: the embedding model is baked into the image and the index loads in ~1 s.

---

## AI accuracy layer (optional)
For hard cases — idioms, vague phrasing, regional slang — enable the AI layer, an
OpenAI-compatible `/chat/completions` provider (default: `https://api.hcnsec.cn/v1`,
model `auto`, which lets the provider route to its best model, e.g. Kimi K2):

1. Set `AI_API_KEY` in your env / Render / Railway variables (**never commit it to
   git** — `.env` is already git-ignored).
2. Optionally set `AI_API_BASE_URL` and `AI_MODEL` to point at a different
   OpenAI-compatible provider/model.

How it stays accurate (and never hallucinates codes):
1. Hybrid search finds real candidate MedDRA terms.
2. The AI **interprets** the phrase and either **re-ranks** those candidates or
   **suggests clinical search terms** we then search ourselves.
3. Every returned code comes from the real MedDRA index — the AI only chooses.

Example: `nerve dancing` → (pure search) Neuralgia 75% → (with AI) it interprets
"involuntary twitching/tingling" and returns **Paraesthesia / Muscle twitching** at ~99%.

Toggle per request with `?ai=true|false`; the web UI has an **AI Suggest** side tab
with a live status signal (🟢 live / 🔴 dead or quota exceeded / configured or not).
Without a key, everything falls back to pure hybrid search automatically.

⚠️ **Key hygiene**: if an API key was ever pasted into a chat, doc, or committed to
git, treat it as compromised and rotate it with the provider.

## Telegram bot
1. Message **@BotFather** → `/newbot` → copy the token.
2. Set `TELEGRAM_BOT_TOKEN` in your env / Railway variables.
3. Open your bot, send any term. Default polling mode needs no public URL.

For multiple replicas, set `TELEGRAM_MODE=webhook` (uses `/telegram/webhook`).

---

## Rebuild the index (only if MEDDRA.xlsx changes)
```bash
pip install -r requirements.txt -r requirements-build.txt
python scripts/build_index.py --xlsx /path/to/MEDDRA.xlsx
```
This regenerates `data/meddra_terms.jsonl.gz` and `data/meddra_vectors.npz`.

---

## How ranking works
Each candidate term is scored by blending:
- **semantic** cosine similarity (bge-small ONNX embeddings) — meaning & paraphrases,
- **lexical** WRatio (length-damped so tiny generic terms can't hijack) — word overlap,
- **edit-distance** boost — misspellings,
- **exact/substring** boost, with a brevity tiebreak so canonical terms rank first.

Results are de-duplicated by Preferred Term and returned with a 0–100 confidence.
