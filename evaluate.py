"""
Evaluation harness for the telecom RAG chatbot.

Measures two things over eval/eval_set.json:

  1. Retrieval hit rate — for each question, did at least one retrieved
     document contain any of the expected keywords?

  2. Answer quality (LLM-as-judge) — grades each answer on faithfulness
     and relevance, each 1-5.

Usage:
    uv run python evaluate.py                # full eval
    uv run python evaluate.py --retrieval-only   # no LLM calls, fast
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import json
import re
import statistics
import sys

from dotenv import load_dotenv
load_dotenv()

EVAL_SET_PATH = os.path.join("eval", "eval_set.json")

JUDGE_PROMPT = """You are grading a telecom customer-support answer.

Question:
{question}

Retrieved context the assistant was given:
{context}

Assistant's answer:
{answer}

Grade the answer on two axes, each an integer from 1 (terrible) to 5 (excellent):
- faithfulness: every claim in the answer is supported by the context, with no invented details
- relevance: the answer directly addresses the customer's question

Reply with ONLY a JSON object, no markdown fences, in the form:
{{"faithfulness": <int>, "relevance": <int>, "comment": "<one short sentence>"}}
"""


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def retrieval_hit(docs, keywords: list[str]) -> bool:
    blob = " ".join(d.page_content.lower() for d in docs)
    return any(kw.lower() in blob for kw in keywords)


def parse_judge_response(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return {
            "faithfulness": int(data["faithfulness"]),
            "relevance": int(data["relevance"]),
            "comment": str(data.get("comment", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation and judging (no LLM calls).",
    )
    args = parser.parse_args()

    if not args.retrieval_only and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — run with --retrieval-only or add it to .env")
        sys.exit(1)

    from helpers import format_docs
    from retriever import build_retriever

    eval_set = load_eval_set()
    retriever = build_retriever()

    rag = None
    judge = None
    if not args.retrieval_only:
        from langchain_openai import ChatOpenAI
        from rag_chain import TelecomRAG

        rag = TelecomRAG(retriever=retriever)
        judge = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

    hits = 0
    faithfulness_scores: list[int] = []
    relevance_scores: list[int] = []

    print(f"Running eval on {len(eval_set)} questions...\n")

    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        docs = retriever.retrieve(question)
        hit = retrieval_hit(docs, item["expected_keywords"])
        hits += hit

        line = f"[{i:>2}/{len(eval_set)}] retrieval {'HIT ' if hit else 'MISS'} ({len(docs)} docs)"

        if rag is not None:
            answer = rag.invoke(question)
            judge_raw = judge.invoke(
                JUDGE_PROMPT.format(
                    question=question,
                    context=format_docs(docs),
                    answer=answer,
                )
            ).content
            scores = parse_judge_response(judge_raw)
            if scores:
                faithfulness_scores.append(scores["faithfulness"])
                relevance_scores.append(scores["relevance"])
                line += (
                    f" | faith {scores['faithfulness']}/5"
                    f" | rel {scores['relevance']}/5 — {scores['comment']}"
                )
            else:
                line += " | judge response unparseable"

        print(f"{line}\n        {question}")

    print("\n" + "=" * 60)
    print(f"Retrieval hit rate : {hits}/{len(eval_set)} ({hits / len(eval_set):.0%})")
    if faithfulness_scores:
        print(f"Avg faithfulness   : {statistics.mean(faithfulness_scores):.2f}/5")
        print(f"Avg relevance      : {statistics.mean(relevance_scores):.2f}/5")
    print("=" * 60)


if __name__ == "__main__":
    main()
