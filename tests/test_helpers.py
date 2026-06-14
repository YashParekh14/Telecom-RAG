from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from helpers import describe_source, format_docs, to_langchain_messages


def test_format_docs_labels_sources():
    docs = [
        Document(page_content="Q: a\nA: b", metadata={"source": "faq"}),
        Document(page_content="Issue: x", metadata={"source": "ticket"}),
        Document(page_content="Guide text", metadata={"source": "guide"}),
    ]
    out = format_docs(docs)
    assert "[FAQ]" in out
    assert "[TICKET]" in out
    assert "[GUIDE]" in out
    assert out.count("---") == 2


def test_format_docs_empty():
    assert "no relevant documents" in format_docs([])


def test_format_docs_unknown_source():
    out = format_docs([Document(page_content="x", metadata={})])
    assert "[UNKNOWN]" in out


def test_describe_source_faq():
    doc = Document(
        page_content="x",
        metadata={"source": "faq", "faq_id": "12", "category": "billing",
                  "rerank_score": 4.21},
    )
    label = describe_source(doc)
    assert "FAQ #12" in label
    assert "billing" in label
    assert "4.21" in label


def test_describe_source_ticket():
    doc = Document(
        page_content="x",
        metadata={"source": "ticket", "ticket_id": "T-3041", "category": "sim"},
    )
    assert "Ticket T-3041" in describe_source(doc)


def test_to_langchain_messages_roles():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    msgs = to_langchain_messages(history)
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[0].content == "hi"


def test_to_langchain_messages_truncates():
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    msgs = to_langchain_messages(history, max_turns=3)
    assert len(msgs) == 6
    assert msgs[0].content == "q17"  