
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys

import pandas as pd
from langchain_core.documents import Document

from config import CHROMA_DIR, COLLECTION_FAQ, EMBED_MODEL, FAQ_CSV


def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    required = {"id", "question", "answer", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"faq.csv is missing required columns: {sorted(missing)}")

    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(Document(
            page_content=content,
            metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])},
        ))
    return docs


def main():
    if not os.path.exists(FAQ_CSV):
        print(f"  ✗ {FAQ_CSV} not found. Generate or add the FAQ data first.")
        sys.exit(1)

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Loading FAQ documents...")
    docs = load_faq_documents(FAQ_CSV)
    print(f"  {len(docs)} FAQ entries loaded.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{COLLECTION_FAQ}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_FAQ,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()
