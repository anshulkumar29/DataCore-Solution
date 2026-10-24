"""
Phase 5 - RAG chatbot (LangChain).
It answers questions about our project using ONLY two files:
    README.md          (all phase notes)
    model_metrics.json (final model scores)

RAG steps:  load files -> split into chunks -> embed -> store -> retrieve top chunks -> LLM answers from them
"""
import json
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = Path(__file__).parent
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

PROMPT = ChatPromptTemplate.from_template(
    """You are a friendly AI assistant for our customer churn prediction project (a college project).
You have two sources of help:
1. The project context below (from README.md and model_metrics.json).
2. Your own general knowledge.

Rules:
- If the question is about the project, answer from the context and quote its exact numbers.
- If the question is a greeting, about you, or a general question (churn, machine learning, statistics), answer normally in your own words. You may use the context to make the answer more specific.
- Never invent project numbers or results. If a project detail is not in the context, say it is not in the project files.
- Keep answers short and clear.

Earlier conversation:
{history}

Project context:
{context}

Question: {question}
Answer:"""
)


@st.cache_resource(show_spinner=False)
def build_retriever():
    # 1. Load the two project files as documents
    readme = (BASE / "README.md").read_text(encoding="utf-8")
    metrics = (BASE / "model_metrics.json").read_text(encoding="utf-8")
    docs = [
        Document(page_content=readme, metadata={"source": "README.md"}),
        Document(page_content="Final model metrics (model_metrics.json):\n" + metrics,
                 metadata={"source": "model_metrics.json"}),
    ]

    # 2. Split into small chunks (try headings first, then paragraphs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed the chunks and keep them in an in-memory vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=st.secrets["GOOGLE_API_KEY"])
    store = InMemoryVectorStore.from_documents(chunks, embeddings)

    # 4. Retriever = finds the 5 chunks most similar to the question
    return store.as_retriever(search_kwargs={"k": 5})


def show_chatbot():
    st.title("Ask the Project Chatbot")
    st.caption("RAG with LangChain: the LLM answers using README.md and model_metrics.json as its knowledge.")

    if "chat" not in st.session_state:
        st.session_state.chat = []          # list of {"role": ..., "text": ...}

    # show old messages
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    question = st.chat_input("Ask anything about the project, e.g. Why is Payment Delay important?")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the project files..."):
                retriever = build_retriever()
                found = retriever.invoke(question)                 # retrieve
            context = "\n---\n".join(d.page_content for d in found)
            history = "\n".join(f"{m['role']}: {m['text']}" for m in st.session_state.chat[-4:]) or "(none)"

            llm = ChatGoogleGenerativeAI(
                model=CHAT_MODEL, temperature=0.2, timeout=30, max_retries=1,
                google_api_key=st.secrets["GOOGLE_API_KEY"])
            chain = PROMPT | llm | StrOutputParser()               # prompt -> model -> text
            answer = st.write_stream(chain.stream(
                {"history": history, "context": context, "question": question}))

            with st.expander("Chunks used for this answer"):
                for d in found:
                    st.caption(d.metadata["source"])
                    st.text(d.page_content[:400])
        except Exception as e:
            answer = "Sorry, the chatbot could not answer right now. Check the API key and internet. (" + str(e)[:150] + ")"
            st.error(answer)

    st.session_state.chat.append({"role": "user", "text": question})
    st.session_state.chat.append({"role": "assistant", "text": answer})
