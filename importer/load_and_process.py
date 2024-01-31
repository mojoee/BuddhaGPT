import os
import sys
sys.path.insert(1, '/Users/ma012/Documents/Morit/10_ResearchProjects/08_ESGReader/Prototype')
from dotenv import load_dotenv
from config import EMBEDDING_MODEL, PG_COLLECTION_NAME
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import PGVector
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
    sample_size=4
)

docs = loader.load()


embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings()
)

chunks = text_splitter.split_documents(docs)

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=PG_COLLECTION_NAME,
    connection_string="postgresql+psycopg://ma012@localhost:5432/pdf_rag_vectors",
    pre_delete_collection=True
)
