"""RAG version of the chatbot. Reuses chatbot.py (unchanged) for facts, prompt and model."""
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chatbot import MODEL, PROJECT_INFO, live_facts, prompt

KB_FILE = "knowledge.md"          # the saved knowledge file that gets chunked + indexed


@st.cache_resource(show_spinner=False)
def get_retriever(df, _train):
    text = PROJECT_INFO + live_facts(df, _train)
    open(KB_FILE, "w", encoding="utf-8").write(text)                       # 1. save to file
    text = open(KB_FILE, encoding="utf-8").read()                          # 2. load it back
    docs = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60,
                                          separators=["\n\n", "\n", " "]
                                          ).create_documents([text])        # 3. chunk
    emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",
                                       google_api_key=st.secrets["GOOGLE_API_KEY"])
    store = InMemoryVectorStore.from_documents(docs, emb)                   # 4. embed + index
    return store.as_retriever(search_kwargs={"k": 5})                       # 5. retriever


def render_chatbot(df, train_model):
    with st.sidebar.expander("\U0001F50E Ask the chatbot (RAG)"):
        with st.form("rag_form", clear_on_submit=True):
            question = st.text_input("Your question")
            ask = st.form_submit_button("Ask")
        hist = st.session_state.setdefault("rag_hist", [])
        if ask and question:
            with st.spinner("Building knowledge base (first time only)..."):
                retriever = get_retriever(df, train_model)
            chunks = retriever.invoke(question)                             # retrieve top-k
            context = "\n---\n".join(d.page_content for d in chunks)
            llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.2, max_retries=1,
                                         timeout=30, google_api_key=st.secrets["GOOGLE_API_KEY"])
            chain = prompt | llm | StrOutputParser()
            history = "\n".join(f"Q: {q}\nA: {a}" for q, a, _ in hist[-3:]) or "(none)"
            ans = st.write_stream(chain.stream(
                {"info": context, "history": history, "question": question}))
            hist.append((question, ans, context))
        elif hist:
            st.info(hist[-1][1])
        if hist:
            with st.expander("Retrieved chunks used for the last answer"):
                st.text(hist[-1][2])