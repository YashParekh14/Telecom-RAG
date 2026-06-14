Live Demo : https://telecom-rag-elh9qedf8cwdgdxjpcfdva.streamlit.app/

# RAG Telecom Chatbot

A customer care chatbot built with Retrieval-Augmented Generation (RAG) for telecom support. It answers questions about mobile connectivity, billing, SIM issues, and roaming by pulling relevant context from three knowledge sources and generating responses with GPT-4o-mini.

I built this to get hands-on experience with the full RAG stack — not just wiring up LangChain, but actually measuring what works and why.

## What it does

- Answers telecom support questions using real FAQ data, resolved support tickets, and a user guide PDF
- Remembers context across a conversation — follow-up questions like "how do I dispute that?" work without repeating yourself
- Shows which documents grounded each answer (FAQ entry, ticket ID, guide chunk) with relevance scores
- Reranks retrieved candidates with a cross-encoder instead of blindly returning top-k per collection

## Architecture


Chat history + new question
     │
     ▼
1. Condense (LLM) — rewrites follow-ups into standalone questions
     │
     ▼
2. Merged Retriever
  ├── ChromaDB · faq        FAQ entries (CSV)
  ├── ChromaDB · tickets    Resolved support tickets (SQLite)
  └── ChromaDB · guides     PDF guide chunks
     │
     score threshold filter → dedup → cross-encoder rerank → top-6
     │
     ▼
3. GPT-4o-mini → streamed answer + source list


**Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally)  
**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally)  
**LLM:** `gpt-4o-mini` via OpenAI

## Evaluation

I built a small eval harness (`evaluate.py`) that runs 12 fixed questions and measures two things:

1. **Retrieval hit rate** — did the retriever find at least one relevant document?
2. **Answer quality** — an LLM-as-judge grades each answer 1–5 for faithfulness (no hallucination) and relevance (actually answers the question)

### Baseline results

Starting with the original 25 FAQ entries:

| Metric | Score |
|---|---|
| Retrieval hit rate | 10/12 (83%) |
| Avg faithfulness | 4.83 / 5 |
| Avg relevance | 4.33 / 5 |

Two questions missed completely:
- "I can't send text messages but calls work fine" — no SMSC content in the knowledge base
- "What happens to my number if I cancel?" — nothing about PAC codes or number porting anywhere

### What I tried

Rather than immediately tuning hyperparameters, I first ran the experiments to understand where the problem actually was:

| What I changed | Hit rate | Observation |
|---|---|---|
| Baseline (reranker ON, threshold=0.25) | 83% | 2 knowledge gaps |
| Reranker OFF (threshold=0.25) | 83% | No change — ranking wasn't the issue |
| Lower threshold to 0.15 (reranker ON) | 83% | No change — the docs just didn't exist |
| Added 5 FAQ entries covering the gaps | **100%** | Fixed both misses immediately |

The experiments made it clear the bottleneck was missing content, not algorithm tuning. Once I added targeted FAQ entries for SMS troubleshooting, number porting, and voicemail setup, hit rate jumped to 100% and stayed there regardless of reranker or threshold settings.

### After data augmentation

Added 5 FAQ entries:
- Number porting / PAC code request process
- What happens to your number if you cancel without porting
- SMS not sending / SMSC number fix
- Text messages troubleshooting steps
- Voicemail setup and PIN reset

| Metric | Score |
|---|---|
| Retrieval hit rate | **12/12 (100%)** |
| Avg faithfulness | 4.83 / 5 |
| Avg relevance | 4.33 / 5 |

Faithfulness stayed high throughout — the model rarely invented details outside the retrieved context.

## Project structure

```
rag-telecom-chatbot/
├── app.py              Streamlit web UI
├── main.py             CLI entry point
├── config.py           All settings in one place
├── helpers.py          Formatting and history utilities (unit-tested)
├── rag_chain.py        Condense → retrieve → answer pipeline
├── retriever.py        Merged retriever with threshold, dedup, reranking
├── evaluate.py         Eval harness
├── eval/
│   └── eval_set.json   12 test questions with expected keywords
├── tests/
│   ├── test_helpers.py
│   └── test_ingest_loaders.py
├── ingest_faq.py
├── ingest_tickets.py
├── ingest_pdf.py
├── data/
│   ├── faq.csv             30 FAQ entries
│   ├── tickets.db          SQLite database of resolved tickets
│   ├── telecom_guide.pdf   Reference guide (chunked at ingest)
│   ├── seed_tickets.py
│   └── generate_pdf.py
├── chroma_store/       Persisted vector store (created at ingest)
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## Setup

**1. Install**
```bash
git clone <repo-url>
cd rag-telecom-chatbot
uv sync
uv add langchain-openai
```

**2. Configure `.env`**
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

**3. Ingest data**
```bash
uv run python ingest_faq.py
uv run python ingest_tickets.py
uv run python ingest_pdf.py
```

**4. Run**
```bash
uv run streamlit run app.py    # web UI → localhost:8501
uv run python main.py          # CLI
```

## Evaluation

```bash
uv run python evaluate.py                    # retrieval + LLM-as-judge scoring
uv run python evaluate.py --retrieval-only   # retrieval only, no API calls
```

## Tests

```bash
uv run pytest
```

10 tests covering document formatting, source labels, history truncation, and both ingestion loaders.

## Docker

```bash
docker build -t telecom-rag .
docker run -p 8501:8501 --env-file .env \
  -v $(pwd)/chroma_store:/app/chroma_store \
  telecom-rag
```

## Config knobs

Everything lives in `config.py`:

| Setting | Default | What it does |
|---|---|---|
| `K_PER_COLLECTION` | 4 | Candidates fetched per collection before reranking |
| `TOP_N_RESULTS` | 6 | Documents passed to the LLM after reranking |
| `SCORE_THRESHOLD` | 0.25 | Minimum relevance score to be considered |
| `USE_RERANKER` | true | Enable/disable cross-encoder reranking |
| `MAX_HISTORY_TURNS` | 6 | How many past turns to include in the prompt |
| `CHUNK_SIZE` | 600 | PDF chunk size at ingest |
| `CHUNK_OVERLAP` | 100 | PDF chunk overlap at ingest |
