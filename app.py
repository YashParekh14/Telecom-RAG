import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv

from config import CHROMA_DIR
from helpers import describe_source, to_langchain_messages

load_dotenv()

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)


def run_startup_checks() -> list[str]:
    problems = []
    if not os.getenv("OPENAI_API_KEY"):
        problems.append(
            "`OPENAI_API_KEY` is not set. Add it to your .env file."
        )
    if not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        problems.append(
            f"Vector store `{CHROMA_DIR}/` not found or empty. "
            "Run: python ingest_faq.py && python ingest_tickets.py && python ingest_pdf.py"
        )
    return problems


startup_problems = run_startup_checks()
if startup_problems:
    st.title("📡 Telecom Support — setup required")
    for p in startup_problems:
        st.error(p)
    st.stop()


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_rag():
    from rag_chain import build_rag
    return build_rag()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("Powered by RAG · GPT-4o-mini")
    st.divider()
    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("Customer Care Assistant")
st.caption(
    "Ask me anything about your mobile service — connectivity, billing, "
    "SIM, roaming, and more. Follow-up questions are supported."
)


def render_sources(sources: list[str]):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for label in sources:
            st.markdown(f"- {label}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

question = st.chat_input("Describe your issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    chat_history = to_langchain_messages(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        rag = get_rag()
        try:
            response = st.write_stream(rag.stream(question, chat_history))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()
        source_labels = [describe_source(d) for d in rag.last_sources]
        render_sources(source_labels)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": source_labels,
    })
