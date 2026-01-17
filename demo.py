import streamlit as st
from src.rag_pipeline import run_rag
from src.retriever import get_relevant_events

st.set_page_config(
    page_title="Event Intelligence RAG",
    layout="centered"
)

st.title("🚨 Event Intelligence RAG System")
st.markdown(
    """
Ask natural language questions about operational events, alarms, and incidents.
Answers are **strictly grounded in data**.
"""
)

# --- User Input ---
query = st.text_input(
    "Enter your question:",
    placeholder="Why are there so many critical alarms from component 103?"
)

show_context = st.checkbox("Show retrieved event context (debug)")

# --- Run RAG ---
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing events..."):
            answer = run_rag(query)

        st.subheader("🧠 Answer")
        st.write(answer)

        # Optional: show retrieved docs
        if show_context:
            docs = get_relevant_events(query)

            st.subheader("📄 Retrieved Events")
            if not docs:
                st.info("No relevant events retrieved.")
            else:
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Event {i} — {doc.metadata.get('event_id')}"):
                        st.write(doc.page_content)
                        st.json(doc.metadata)

# --- Footer ---
st.markdown("---")
st.caption("Event Intelligence RAG • Built with Streamlit, Chroma, and LLMs")
