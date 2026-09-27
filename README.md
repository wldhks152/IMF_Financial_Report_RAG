# 📈 IMF World Economic Outlook 2026 RAG Chatbot

**English** | [한국어](README.ko.md)

A RAG (Retrieval-Augmented Generation) chatbot that answers questions based on the IMF **World Economic Outlook 2026** reports (January Update · April full report · July Update).
Built with LangChain + OpenAI + Chroma and served on the web with Gradio. The chatbot answers in Korean.

![Chatbot screenshot: comparing Japan's and Korea's 2026 growth projections by report release date, with sources](docs/demo.png)

- Shows the **source (report · page)** for every answer
- Does not make things up: answers **"not found in the provided reports"** when the reports don't cover it
- When reports give different numbers, **compares them in order of release date**
- **Conversation memory**: supports follow-up questions like "What about Japan?"
- **Rewrites Korean questions into English search queries** to improve retrieval over English reports
- **Table handling**: converts each table row into a sentence with its title and column names, so country-level figures can be retrieved

## Architecture

```
[Indexing: build_index.py — run only when the PDFs change]
data/*.pdf ─ loader.py ─┬─ text pages ── splitter.py (1000-char chunks) ─┐
                        └─ table pages ─ tables.py (GPT: row → sentence) ─┴─ vectorstore.py ─▶ chroma_db/

[Querying: app.py — on every question]
question + chat history ─▶ ⓪ rewrite into English query ─▶ ① Chroma search (top 4) ─▶ ② build prompt ─▶ ③ GPT answer + sources
```

| File | Role |
|---|---|
| `config.py` | Settings: paths, model names, chunk size, number of results, etc. |
| `loader.py` | Loads the PDFs in `data/` as page-level Documents |
| `splitter.py` | Splits pages into 1000-character chunks (200-character overlap) |
| `tables.py` | Detects table pages → converts rows into sentences with GPT → verifies numbers → caches results |
| `vectorstore.py` | OpenAI embeddings + creating/loading the Chroma DB |
| `build_index.py` | Runs the steps above in one go to build the vector DB |
| `rag_chain.py` | Connects query rewriting + retrieval + prompt + GPT as an LCEL chain |
| `app.py` | Gradio chat UI (streaming, sources, report selector) |
| `compare_search.py` | Experiment comparing Korean vs. English-rewritten search |
| `requirements.txt` | Dependencies (pinned to tested versions) |
| `.env.example` | Template for the API key file |
| `docs/` | Images for the README |

Files created at runtime (not in the repository):

| File | Description |
|---|---|
| `.env` | Your OpenAI API key |
| `chroma_db/` | Vector DB (created by `build_index.py`) |
| `table_cache.json` | Cache of table conversions (reused on later `build_index.py` runs without calling GPT again) |

## Getting started

Tested on Python 3.14.2 / Windows 11.

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

> If PowerShell says scripts cannot be run when you call `activate`, you can skip activation and call the virtual environment's Python directly.
> e.g. `.venv\Scripts\python.exe -m pip install -r requirements.txt`, `.venv\Scripts\python.exe app.py`

<details>
<summary>macOS / Linux</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```
(Not tested on macOS / Linux.)
</details>

### 2. Set your API key

```bash
copy .env.example .env
```
Open `.env` and put your key in `OPENAI_API_KEY`.

### 3. Add the PDFs

The PDFs are not included in this repository. Download the reports below from the [IMF World Economic Outlook](https://www.imf.org/en/Publications/WEO) page and put them in the `data/` folder (any file name works).

- World Economic Outlook Update, January 2026
- World Economic Outlook, April 2026
- World Economic Outlook Update, July 2026

### 4. Build the vector DB (only when the PDFs change)

```bash
python build_index.py
```
The first run takes about 4 minutes because the table pages (43 pages) are converted with GPT. The results are saved to `table_cache.json`, so later runs are fast.

### 5. Run the app

```bash
python app.py
```
Open http://127.0.0.1:7860 in your browser.

## Models and API usage

- Embeddings: `text-embedding-3-small`
- Answers · query rewriting · table conversion: `gpt-5-mini`

Token usage measured with the three PDFs above (208 pages):

| Task | Tokens | Approx. cost |
|---|---|---|
| First indexing — table conversion (43 pages) | ~100K input / ~180K output | ~$0.40 |
| First indexing — embeddings (2,376 documents) | ~390K | < $0.01 |
| Later indexing (table cache reused) | embeddings only | < $0.01 |
| One question | ~1,500 (more for answers based on body text) | < $0.01 |

Costs are estimates based on prices at the time of writing. Check the [OpenAI pricing page](https://openai.com/api/pricing/) for current prices.

## Design decisions

### 1. Vector DB: Chroma instead of FAISS
I first planned to use FAISS as the simplest option, but switched to Chroma after deciding to **support multiple PDFs**.
- **Metadata filters**: `filter={"file_name": ...}` restricts search to one report, which made the "report to search" menu easy to build.
- **Upsert with unique IDs**: each chunk gets a `file:page:start_index` ID, so rebuilding the index doesn't create duplicates.
- LangChain's FAISS integration lives in `langchain-community`, which is being sunset, while Chroma is maintained as a standalone package (`langchain-chroma`).

### 2. Chunk size: 1000 characters / 200 overlap
Measuring the PDFs showed an average sentence length of about 159 characters. 1000 characters is about six sentences, i.e. **one or two paragraphs**, which matches the "one claim + its supporting evidence" unit of IMF reports. The 200-character overlap (a bit more than one sentence) makes sure a sentence cut at a chunk boundary survives intact in one of the neighboring chunks.

### 3. Different numbers for the same question
"2026 global growth" is 3.3% in January, 3.1% in April and 3.0% in July. If only the chunks are passed, GPT can't tell which projection is the latest. So I **prepend the report title (release month) from the PDF metadata to each chunk** and added a prompt rule to compare figures in order of release. I also confirmed that renaming the files has no effect on retrieval (only the body text is embedded); it only matters when passed to GPT.

### 4. Korean questions → English search queries
The reports are in English and the questions are in Korean, so the search scores barely differed. For example, the top 4 distances for "What is the 2026 global growth outlook?" (asked in Korean) were 0.377–0.388, practically identical. So I added a step that uses GPT to **write an English search query** before retrieval.

| Question (asked in Korean) | Korean as is | English rewrite |
|---|---|---|
| What is the outlook for the Korean economy? | 0.413 | **0.345** |
| What happens if the AI investment boom ends? | 0.542 | **0.348** |
| How should central banks set interest rates? | 0.741 | **0.421** |
| What is the 2026 global growth outlook? | 0.377 | 0.380 (no difference) |

(Cosine distance of the top chunk; lower means closer to the question. Measured with `compare_search.py`. Changing the query sentence also shifts the baseline slightly, so this is a rough comparison.)

I revised the prompt three times along the way.
1. **"Use IMF terminology"** → the model stuffed in close to 50 words of keywords, including topics the user never asked about, which blurred the focus of the query. → Added a rule: "stay within the question's scope and keep it short".
2. **"What about Japan?" got worse** → the cause was "IMF World Economic Outlook" in the query. Every document is an IMF report, so these words are equally similar to every chunk and diluted the weight of the word that mattered, "Japan". The correct chunk fell to 9th place, and moved up to 1st once those words were removed. → Added a rule: **"do not add words common to every document"**.
3. **Made-up details** → for the interest-rate question, the model added "Korea", which the user never mentioned. Raising `reasoning_effort` from `minimal` (~0.9s) to `low` (~1.7s) fixed it. A wrong search query breaks every step after it, so I chose accuracy over speed.

### 5. Conversation memory is needed in two places
A regular chatbot only needs to pass the chat history to GPT, but RAG has a separate **retrieval step**. Retrieval embeds only the single question sentence, so searching "explain the first one in more detail" as is returns meaningless results. So ① before retrieval, the question is **rewritten** with the conversation context filled in, and ② the chat history is also passed when answering. To keep tokens from growing, only the last 3 turns are passed and the source lists shown in the UI are stripped.

### 6. Table handling and verifying LLM output
Splitting tables into 1000-character chunks separated number lines like `Japan –0.2 1.1 0.6 0.7` from their column headers (years), so the embeddings were meaningless and GPT couldn't tell which year a value belonged to. So during indexing, only the table pages (43 pages that have a table title and at least 15% numeric tokens) are selected and **converted by GPT into "one row = one sentence with the table title and column names"**.

Instead of trusting the numbers GPT transcribed, a verification step **drops, in code, any row containing a number that doesn't appear on the original page** (3 of 1,563 rows removed). Conversions are cached so rebuilding the index doesn't call the API again. As a result, the chatbot can now answer questions about information that **exists only in the statistical appendix tables**, such as German inflation or Korea's current account balance, and I checked several rows against the original PDF to confirm the numbers.

## Known limitations

- Table conversion is done by GPT, so the verification step may not catch a number **attached to the wrong column**. (Numbers not in the original are filtered out.)
- Some table rows may mislabel actual values vs. projections.
- Page headers are mixed into chunks, and sentences are cut at page boundaries.
- `langchain-community` is being sunset, so a deprecation warning is shown at runtime. (It still works.)

## License and attribution

- Code: [MIT License](LICENSE)
- Original reports © International Monetary Fund. This repository is a RAG example for learning purposes and is not affiliated with the IMF.
