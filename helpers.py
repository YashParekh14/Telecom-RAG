
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import MAX_HISTORY_TURNS


def format_docs(docs: list[Document]) -> str:
    """Render retrieved documents into the context block for the prompt."""
    if not docs:
        return "(no relevant documents found)"
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def describe_source(doc: Document) -> str:

    meta = doc.metadata
    source = meta.get("source", "unknown")
    if source == "faq":
        label = f"FAQ #{meta.get('faq_id', '?')} · {meta.get('category', 'general')}"
    elif source == "ticket":
        label = f"Ticket {meta.get('ticket_id', '?')} · {meta.get('category', 'general')}"
    elif source == "guide":
        label = f"Guide · chunk {meta.get('chunk_index', '?')} (p.{meta.get('page', '?')})"
    else:
        label = source
    score = meta.get("rerank_score", meta.get("retrieval_score"))
    if score is not None:
        label += f" · score {score}"
    return label


def to_langchain_messages(
    history: list[dict],
    max_turns: int = MAX_HISTORY_TURNS,
) -> list[BaseMessage]:
    
    messages: list[BaseMessage] = []
    for msg in history:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    # one turn ≈ two messages (user + assistant)
    return messages[-(2 * max_turns):]
