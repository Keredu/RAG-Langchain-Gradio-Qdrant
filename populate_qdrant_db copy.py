import glob

import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Qdrant
from pdfminer.high_level import extract_text
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from rag.chatbot import Chatbot


def create_collection():
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "potter20"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = Qdrant(client, collection_name, embedding_function)

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Example PDF extraction
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text


if __name__ == '__main__':
    datadir='data/potter20/*'
    # https://github.com/lucifertrj/Awesome-RAG/blob/main/apps/Langchain_Streaming/ingest.py
    client = QdrantClient(url="http://localhost:6333")
    collection_name = 'potter20'
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    create_collection(client=client,
                      collection_name=collection_name,
                      encoder=encoder)
    i=0
    for path in glob.glob(datadir):
        if path.endswith('.pdf'):
            text = extract_text_from_pdf(path)
            chunks = get_chunks(text)
            embeddings = encoder.encode(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=i,
                            vector=embedding.tolist(),
                            payload={'path':path,
                                     'name':chunk[:100]
                            }
                        )
                    ],
                )
                i+=1