import sqlite3

import pandas as pd
import pytest

from ingest_faq import load_faq_documents
from ingest_tickets import load_ticket_documents


@pytest.fixture
def faq_csv(tmp_path):
    path = tmp_path / "faq.csv"
    pd.DataFrame([
        {"id": 1, "question": "How do I top up?",
         "answer": "Use the app.", "category": "billing"},
        {"id": 2, "question": "How do I roam?",
         "answer": "Enable roaming.", "category": "roaming"},
    ]).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tickets_db(tmp_path):
    path = tmp_path / "tickets.db"
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE tickets (
            ticket_id TEXT, issue_type TEXT, description TEXT,
            resolution TEXT, category TEXT, status TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("T-1", "slow data", "Data is slow", "Reset APN", "connectivity", "resolved"),
            ("T-2", "billing", "Overcharged", "Refund issued", "billing", "resolved"),
            ("T-3", "sim", "SIM broken", "", "sim", "open"),  
        ],
    )
    conn.commit()
    conn.close()
    return str(path)


def test_load_faq_documents(faq_csv):
    docs = load_faq_documents(faq_csv)
    assert len(docs) == 2
    assert docs[0].page_content.startswith("Q: How do I top up?")
    assert docs[0].metadata == {"source": "faq", "category": "billing", "faq_id": "1"}


def test_load_faq_documents_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"id": 1, "question": "x"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_faq_documents(str(path))


def test_load_ticket_documents_only_resolved(tickets_db):
    docs = load_ticket_documents(tickets_db)
    assert len(docs) == 2
    ids = {d.metadata["ticket_id"] for d in docs}
    assert ids == {"T-1", "T-2"}
    assert "Resolution: Reset APN" in docs[0].page_content
