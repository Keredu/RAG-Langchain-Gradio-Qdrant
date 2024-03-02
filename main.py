from rag.chatbot import Chatbot
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub


def retrieve_documents(client, collection_name, query, top_k=5):
    # This is a simplified example. You'll need to adjust it according to your collection schema.
    # Perform a search query in your collection
    search_results = client.search(
        collection_name=collection_name,
        query=query,
        top=top_k
    )
    documents = [result.payload for result in search_results["hits"]]
    return documents

if __name__ == "__main__":
    # chatbot = Chatbot()
    # chatbot.run()
    # Extract documents from the search results

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
    # Create the RAG model

    retriever = db.as_retriever()

    query = "Describe Harry Potter's first Quidditch match."
    relevant_documents = retriever.get_relevant_documents(query)

    print(relevant_documents)
