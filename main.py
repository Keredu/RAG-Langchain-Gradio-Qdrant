from qdrant_client import QdrantClient
from rag.logger import get_logger
from rag.conf import load_conf

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import json
from rag.qdrant import load_collection


if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.info("Loading configuration")
    conf = load_conf()
    collection_conf = conf.get("collection", None)
    if collection_conf is None:
        logger.critical("Collection configuration not found in configuration")
        exit(1)

    embeddings_conf = conf.get("embeddings", None)
    if embeddings_conf is None:
        logger.critical("Embeddings configuration not found in configuration")
        exit(1)

    load_collection(collection_conf=collection_conf,
                    embeddings_conf=embeddings_conf)

    client = QdrantClient(url="http://localhost:6333")
    retriever = Qdrant(client,
                       collection_conf['name'],
                       embeddings=OpenAIEmbeddings()).as_retriever()
    llm = OpenAI()
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=False,
    )

    with open("data/questions.json", "r") as f:
        questions = json.load(f)[:5]

    for question in questions:
        try:
            d = qa.invoke(question)
            q, r = d['query'], d['result']
            print(f'> {q}\n{r}', end="\n\n")
        except Exception as e:
            print(f"Error: {e}", end="\n\n")
            continue