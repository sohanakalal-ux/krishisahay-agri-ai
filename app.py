import streamlit as st
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="KCC Farmer Support System",
    page_icon="🌾",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌾 KCC Support System")
st.sidebar.markdown(
    """
    **AI-powered platform for farmer-related queries**

    **Tech Stack**
    - Sentence Transformers  
    - FAISS  
    - Streamlit  

    **Features**
    - Semantic Search  
    - Offline Knowledge Base  
    - Fast Retrieval
    """
)

top_k = st.sidebar.slider(
    "Number of related answers",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.markdown("---")
st.sidebar.info("Developed as an AI prototype for farmer welfare")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🌾 KCC Farmer Support System</h1>
    <p style="text-align:center; font-size:18px;">
    Ask agriculture-related questions and get AI-powered answers
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD RESOURCES ----------------
@st.cache_resource
def load_resources():
    with open("data/kcc_qa_pairs.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    with open("data/kcc_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    index = faiss.read_index("data/faiss_index.bin")

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

    return qa_pairs, index, model

qa_pairs, index, model = load_resources()

# ---------------- SEARCH FUNCTION ----------------
def semantic_search(query, top_k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        results.append(qa_pairs[idx]["answer"])

    return results

# ---------------- TABS (HORIZONTAL TASK BAR) ----------------
tab1, tab2, tab3 = st.tabs(
    ["💬 Ask Question", "🕘 History", "ℹ️ About Project"]
)

# ---------------- TAB 1: ASK QUESTION ----------------
with tab1:
    st.subheader("Ask a Question")

    user_question = st.text_input(
        "Enter your question below",
        placeholder="e.g. What are the major problems faced by farmers?"
    )

    if st.button("🔍 Get Answer"):
        if user_question.strip() == "":
            st.warning("⚠️ Please enter a question.")
        else:
            answers = semantic_search(user_question, top_k=top_k)

            st.markdown("### ✅ Most Relevant Answer")
            st.success(answers[0])

            # Save to history
            st.session_state.history.append({
                "question": user_question,
                "answer": answers[0]
            })

            if len(answers) > 1:
                st.markdown("### 📌 Related Information")
                for ans in answers[1:]:
                    st.info(ans)

# ---------------- TAB 2: HISTORY ----------------
with tab2:
    st.subheader("Question History")

    if not st.session_state.history:
        st.info("No questions asked yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{i}: {item['question']}**")
            st.write(item["answer"])
            st.markdown("---")

        if st.button("🗑 Clear History"):
            st.session_state.history.clear()
            st.success("History cleared successfully.")

# ---------------- TAB 3: ABOUT ----------------
with tab3:
    st.subheader("About This Project")

    st.markdown(
        """
        ### 📌 Project Overview
        This system helps retrieve answers related to farmer welfare
        using **semantic search**.

        ### 🧠 How It Works
        - Converts questions into vector embeddings
        - Uses FAISS for similarity search
        - Returns answers based on meaning, not keywords

        ### 🚀 Future Enhancements
        - LLM integration
        - Multilingual support
        - Voice-based queries

        ### 🎓 Academic Use
        Demonstrates NLP, vector databases, and AI-based retrieval
        """
    )

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px; color:gray;">
    © 2026 | KCC Farmer Support System | AI Prototype
    </p>
    """,
    unsafe_allow_html=True
) 