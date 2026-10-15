import os
import shutil

import streamlit as st

from dotenv import load_dotenv

load_dotenv()


# Load the API key from Streamlit Cloud Secrets
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

from rag_pipeline import (
    load_documents,
    create_chunks,
    create_vector_database,
    create_rag_chain
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 AI Research & Decision Assistant")

st.write(
    "An AI-powered RAG system for researching "
    "and analyzing business documents."
)


# --------------------------------------------------
# CREATE DOCUMENT FOLDER
# --------------------------------------------------

os.makedirs(
    "./data/documents",
    exist_ok=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📚 Document Management")


uploaded_files = st.sidebar.file_uploader(
    "Upload TXT documents",
    type=["txt"],
    accept_multiple_files=True
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            "./data/documents",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

    st.sidebar.success(
        f"{len(uploaded_files)} files uploaded!"
    )


# --------------------------------------------------
# BUILD KNOWLEDGE BASE
# --------------------------------------------------

if st.button("🔨 Build Knowledge Base"):

    with st.spinner(
        "Loading and processing documents..."
    ):

        documents = load_documents()

        if not documents:

            st.error(
                "No TXT documents found."
            )

            st.stop()

        chunks = create_chunks(
            documents
        )

        vectordb = create_vector_database(
            chunks
        )

        rag_chain, retriever = create_rag_chain(
            vectordb
        )

        st.session_state.rag_chain = rag_chain

        st.session_state.retriever = retriever

        st.session_state.num_documents = len(
            documents
        )

        st.session_state.num_chunks = len(
            chunks
        )


    st.success(
        "✅ Knowledge base created successfully!"
    )


# --------------------------------------------------
# DISPLAY STATISTICS
# --------------------------------------------------

if "num_documents" in st.session_state:

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Documents",
            st.session_state.num_documents
        )

    with col2:

        st.metric(
            "Chunks",
            st.session_state.num_chunks
        )


# --------------------------------------------------
# QUESTION
# --------------------------------------------------

st.header("💬 Ask a Research Question")


question = st.text_input(
    "Enter your question:"
)


if st.button("🔍 Ask"):

    if "rag_chain" not in st.session_state:

        st.warning(
            "⚠️ Please build the knowledge base first."
        )

    elif not question:

        st.warning(
            "⚠️ Please enter a question."
        )

    else:

        with st.spinner(
            "🔎 Searching documents..."
        ):

            response = (
                st.session_state
                .rag_chain
                .invoke(question)
            )


        st.subheader("🧠 Answer")

        st.write(response)