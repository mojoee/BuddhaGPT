import os
import sys
sys.path.insert(1, '/Users/ma012/Documents/Morit/10_ResearchProjects/08_ESGReader/Prototype')
from dotenv import load_dotenv
from config import EMBEDDING_MODEL, PG_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

loader = DirectoryLoader(
    os.path.abspath('./source_docs'),
    glob="**/*.pdf", 
    # use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
    sample_size=1
)

docs = loader.load()


if EMBEDDING_MODEL_NAME == "BAAI/bge-small-en":
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

elif EMBEDDING_MODEL_NAME == "openAI":
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

text_splitter = SemanticChunker(
    embeddings=embeddings
)

chunks = text_splitter.split_documents(docs)

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=PG_COLLECTION_NAME,
    connection_string="postgresql+psycopg://postgres:qwer1234@localhost:5433/vector_rag",
    pre_delete_collection=True
)
