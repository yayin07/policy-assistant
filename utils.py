import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from PyPDF2 import PdfReader

def load_pdf_text(pdf):
    reader = PdfReader(pdf)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def embed_texts(texts, api_key):
    openai.api_key = api_key
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings, texts

def init_qdrant(client: QdrantClient, collection_name: str, embedding_size: int = 1536):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )

def upload_to_qdrant(client, collection_name, texts, embeddings):
    client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": i,
                "vector": embeddings.embed_query(t),
                "payload": {"text": t}
            }
            for i, t in enumerate(texts)
        ]
    )

def query_qdrant(client, collection_name, question, embeddings):
    query_vector = embeddings.embed_query(question)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    return [hit.payload["text"] for hit in search_result]
