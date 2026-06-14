
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_GUIDES,
    EMBED_MODEL,
    GUIDE_PDF,
)


def main():
    if not os.path.exists(GUIDE_PDF):
        print(f"  ✗ {GUIDE_PDF} not found. Run: python data/generate_pdf.py")
        sys.exit(1)

    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("Loading PDF...")
    loader = PyPDFLoader(GUIDE_PDF)
    pages = loader.load()
    print(f"  {len(pages)} pages loaded.")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

   
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{COLLECTION_GUIDES}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_GUIDES,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()
