import json
import glob

from langchain.text_splitter import CharacterTextSplitter
from pdfminer.high_level import extract_text
from qdrant_client import QdrantClient, models
from rag.enums import Datatype
from rag.logger import get_logger
import random
import openai
import os
from qdrant_client.http.models import PointStruct

logger = get_logger(__name__)


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

def upsert_json(qdrant_client, openai_client, collection_name, texts, source_name, embeddings_conf):
    """Inserts data into the collection using OpenAI API."""

    result = openai_client.embeddings.create(input=texts, 
                                             model=embeddings_conf['embedding_model'])
    points = []
    for (data, text) in enumerate(zip(result.data, texts)):
        point = PointStruct(id=random.random(9999999),
                            vector=data.embedding,
                            payload={"text": text,
                                     "source": source_name})
        points.append(point)
    qdrant_client.upsert(collection_name, points)
    

def populate_collection(qdrant_client, collection_name, openai_client, datadir, datatype, source_name, embeddings_conf):
    if datatype == Datatype.JSON.value:
        for path in glob.glob(datadir):
            with open(path, "r") as fp:
                texts = json.load(fp)
            upsert_json(qdrant_client=qdrant_client,
                        openai_client=openai_client,
                        collection_name=collection_name,
                        texts=texts,
                        source_name=source_name,
                        embeddings_conf=embeddings_conf)

    elif datatype == Datatype.PDF.value:
        pass

    else:
        raise ValueError(f"Datatype {datatype} not supported")


def check_sources(sources):
    """Check that datadir, type and name are in the sources configuration."""
    for source in sources:
        datadir = source.get("datadir", None)
        type = source.get("type", None)
        name = source.get("name", None)
        if datadir is None:
            logger.critical("datadir not found in source configuration")
            exit(1)
        if type is None:
            logger.critical("type not found in source configuration")
            exit(1)
        if name is None:
            logger.critical("name not found in source configuration")
            exit(1)

def check_embeddings_conf(embeddings_conf):
    """Check that model and size are in the embeddings configuration."""
    model = embeddings_conf.get("model", None)
    if model is None:
        logger.critical("model not found in embeddings configuration")
        exit(1)
    size = embeddings_conf.get("size", None)
    if size is None:
        logger.critical("size not found in embeddings configuration")
        exit(1)

def load_collection(collection_conf, embeddings_conf):
    """Load the collection into the Qdrant database."""

    collection_name = collection_conf.get("name", None)
    sources = collection_conf.get("sources", None)
    check_sources(sources)
    check_embeddings_conf(embeddings_conf)

    qdrant_client = QdrantClient(url="http://localhost:6333")
    qdrant_client.get_collections()
    if collection_name in [c.name for c in qdrant_client.get_collections().collections]:
        logger.info(f"Collection {collection_name} already exists. Skipping creation.")
    else:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embeddings_conf["size"],
                distance=models.Distance.COSINE,
            ),
        )
        openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        for source in sources:    
            datadir = source["datadir"]
            datatype = source["type"]
            source_name = source["name"]
            populate_collection(qdrant_client=qdrant_client,
                                collection_name=collection_name,
                                openai_client=openai_client,
                                datadir=datadir,
                                datatype=datatype,
                                source_name=source_name,
                                embeddings_conf=embeddings_conf)
            logger.info(f'Source {source_name} loaded successfully.')
        logger.info('Collection loaded successfully.')

