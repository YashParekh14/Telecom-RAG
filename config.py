
import os

DATA_DIR   = "data"
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")

FAQ_CSV    = os.path.join(DATA_DIR, "faq.csv")
TICKETS_DB = os.path.join(DATA_DIR, "tickets.db")
GUIDE_PDF  = os.path.join(DATA_DIR, "telecom_guide.pdf")

COLLECTION_FAQ     = "faq"
COLLECTION_TICKETS = "tickets"
COLLECTION_GUIDES  = "guides"
ALL_COLLECTIONS    = (COLLECTION_FAQ, COLLECTION_TICKETS, COLLECTION_GUIDES)


EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Chunking (PDF guide) ──
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100

# ── Retrieval ───
# Candidates fetched per collection before merging/reranking.
K_PER_COLLECTION = 4
# Final number of context documents passed to the LLM after reranking.
TOP_N_RESULTS = 6
# Minimum cosine relevance score (0–1) for a candidate to be considered.
# Filters out documents that are retrieved only because k must be filled.
SCORE_THRESHOLD = 0.25
# Rerank merged candidates with a cross-encoder. Set to False to fall back
# to pure vector-similarity ordering (faster startup, slightly worse ranking).
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# ── Conversation ──
# Number of past (user, assistant) turns included in the prompt.
MAX_HISTORY_TURNS = 6
