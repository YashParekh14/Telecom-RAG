
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sqlite3
import sys

from langchain_core.documents import Document

from config import CHROMA_DIR, COLLECTION_TICKETS, EMBED_MODEL, TICKETS_DB


def load_ticket_documents(db_path: str) -> list[Document]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM tickets WHERE status = 'resolved'"
        ).fetchall()
    finally:
        conn.close()

    docs = []
    for row in rows:
        # Combine issue description + resolution into a single searchable text block
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source":    "ticket",
                "ticket_id": row["ticket_id"],
                "category":  row["category"],
                "status":    row["status"],
            },
        ))
    return docs


def main():
    if not os.path.exists(TICKETS_DB):
        print(f"  ✗ {TICKETS_DB} not found. Run: python data/seed_tickets.py")
        sys.exit(1)

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Loading ticket documents from SQLite...")
    docs = load_ticket_documents(TICKETS_DB)
    print(f"  {len(docs)} resolved tickets loaded.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{COLLECTION_TICKETS}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_TICKETS,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()
