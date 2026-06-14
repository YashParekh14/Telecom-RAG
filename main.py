
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys

from dotenv import load_dotenv

from config import CHROMA_DIR
from helpers import describe_source, to_langchain_messages

load_dotenv()


def check_setup() -> None:
    problems = []
    if not os.getenv("GROQ_API_KEY"):
        problems.append("GROQ_API_KEY is not set (copy .env.example to .env).")
    if not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        problems.append(
            f"Vector store '{CHROMA_DIR}/' is missing or empty. Run the "
            "ingestion scripts first (ingest_faq.py, ingest_tickets.py, "
            "ingest_pdf.py)."
        )
    if problems:
        print("Setup problems found:\n")
        for p in problems:
            print(f"  ✗ {p}")
        sys.exit(1)


def main():
    check_setup()

    from rag_chain import build_rag  # heavy import deferred until checks pass

    print("=== Telecom Customer Care Chatbot (RAG) ===")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("Follow-up questions are supported.\n")

    rag = build_rag()
    history: list[dict] = []

    while True:
        try:
            question = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        chat_history = to_langchain_messages(history)

        print("\nAssistant: ", end="", flush=True)
        answer_parts = []
        try:
            for chunk in rag.stream(question, chat_history):
                print(chunk, end="", flush=True)
                answer_parts.append(chunk)
        except Exception as exc:
            print(f"\n[error] Failed to generate a response: {exc}")
            continue
        print()

        if rag.last_sources:
            print("\nSources:")
            for doc in rag.last_sources:
                print(f"  - {describe_source(doc)}")
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "".join(answer_parts)})


if __name__ == "__main__":
    main()
