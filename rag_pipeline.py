import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


DOCUMENT_FOLDER = "./data/documents"
CHROMA_FOLDER = "./chroma_db"


def load_documents():

    loader = DirectoryLoader(
        DOCUMENT_FOLDER,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    return documents


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


def create_vector_database(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER
    )

    return vectordb


def create_rag_chain(vectordb):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI Research Assistant.

Answer the question using ONLY the
provided context.

If the answer is not available in the
context, say:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever