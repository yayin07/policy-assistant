import streamlit as st
from utils import load_pdf_text, split_text, embed_texts, init_qdrant, upload_to_qdrant, query_qdrant
from langchain.chat_models import ChatOpenAI
from qdrant_client import QdrantClient

st.title("📘 Company Policy Assistant (RAG-Powered)")

openai_api_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📂 Upload Company Policy PDF", type="pdf")

question = st.text_input("❓ Ask a question about company policy")

if uploaded_file and openai_api_key:
    text = load_pdf_text(uploaded_file)
    chunks = split_text(text)
    embeddings, _ = embed_texts(chunks, openai_api_key)

    # Connect to Qdrant Cloud (you can replace with your actual API key and host)
    client = QdrantClient(
        url="https://<your-qdrant-cluster>.qdrant.cloud",
        api_key="<your-qdrant-api-key>"
    )

    collection = "company-policy"
    init_qdrant(client, collection)
    upload_to_qdrant(client, collection, chunks, embeddings)

    if question:
        retrieved_docs = query_qdrant(client, collection, question, embeddings)
        context = "\n".join(retrieved_docs)

        # Generate answer using OpenAI
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        response = llm.predict(f"Answer the question using this policy info:\n{context}\n\nQ: {question}")
        st.markdown("### 📄 Answer")
        st.write(response)
