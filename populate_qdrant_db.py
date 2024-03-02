import glob

import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from pdfminer.high_level import extract_text
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


def create_collection(client, collection_name, encoder):
    size = encoder.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=size,  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=i,
                vector=np.random.rand(size).tolist(),
            )
            for i in range(100)
        ],
    )


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
    print(i)
    # vectorstore = Qdrant(
    #     client=client,
    #     collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    #     embeddings=embeddings
    # )