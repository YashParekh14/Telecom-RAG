"""
The RAG pipeline with conversation memory.
Uses OpenAI gpt-4o-mini as the LLM provider.
"""
import logging
import os
from collections.abc import Iterator

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from helpers import format_docs
from retriever import MergedRetriever, build_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from three sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)
- The official telecom user guide (reference documentation, split into chunks)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest customer message, rewrite the "
        "message as a single standalone question that contains all context "
        "needed to search a knowledge base. Do NOT answer it. If the message "
        "is already standalone, return it unchanged. Reply with the rewritten "
        "question only — no preamble, no quotes.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


class TelecomRAG:
    """Conversational RAG pipeline with retrieval transparency."""

    def __init__(self, retriever: MergedRetriever | None = None):
        self.retriever = retriever or build_retriever()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            max_retries=2,
        )
        self.last_sources = []
        self.last_standalone_question = None

    def condense_question(self, question: str, chat_history: list[BaseMessage]) -> str:
        if not chat_history:
            return question
        try:
            chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
            rewritten = chain.invoke(
                {"chat_history": chat_history, "question": question}
            ).strip()
            return rewritten or question
        except Exception as exc:
            logger.warning("Question condensation failed: %s", exc)
            return question

    def stream(self, question: str, chat_history: list[BaseMessage] | None = None) -> Iterator[str]:
        chat_history = chat_history or []
        standalone = self.condense_question(question, chat_history)
        self.last_standalone_question = standalone
        docs = self.retriever.retrieve(standalone)
        self.last_sources = docs
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        yield from chain.stream({
            "context": format_docs(docs),
            "chat_history": chat_history,
            "question": question,
        })

    def invoke(self, question: str, chat_history: list[BaseMessage] | None = None) -> str:
        return "".join(self.stream(question, chat_history))


def build_rag(**kwargs) -> TelecomRAG:
    return TelecomRAG(**kwargs)
