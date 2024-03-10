from qdrant_client import QdrantClient
from rag.logger import get_logger
from rag.conf import load_conf

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from rag.qdrant import load_collection
from langchain import hub


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
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt,
                           "verbose": True}
    )

    question = "3 characters with a surname longer than 5 characters"
    question = "Who is Harry Potter?"

    try:
        if False:
            # Task identified as complex; modify to actually use RAG for retrieval
            d = qa.invoke(question)  # Use RAG to retrieve relevant documents
            response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system","content": "You are a helpful assistant who takes the output from a RAG and generates python code to be executed directly without any changes with ."},
                {"role": "user", "content": f"Given the question {question} and this answer from RAG: {d}\n\n Generate python code to answer the question. The python code will be executed directly without any changes with the following statement: result = eval(python_code). Hence, answer only with the code, nothing else."},
            ]
            )
            
            # Execute the dynamically generated code
            result = eval(None)
            print(f'> {question}\n{result}', end="\n\n")
        else:
            # Task is not complex; proceed with RAG as usual
            d = qa.invoke(question)
            q, r = d['query'], d['result']
            print(f'> {q}\n{r}')
    except Exception as e:
        logger.error(f"Error: {e}")