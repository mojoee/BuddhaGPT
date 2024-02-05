from dotenv import load_dotenv
import sys
from typing import TypedDict
from operator import itemgetter
sys.path.insert(1, '/Users/ma012/Documents/Morit/10_ResearchProjects/08_ESGReader/Prototype')
from config import PG_COLLECTION_NAME
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

vector_store = PGVector(
    collection_name=PG_COLLECTION_NAME,
    connection_string="postgresql+psycopg://postgres:qwer1234@localhost:5433/vector_rag",
    embedding_function=OpenAIEmbeddings()
)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", streaming=True)

class RagInput(TypedDict):
    question: str

final_chain  = (
    {
        "context": itemgetter("question") | vector_store.as_retriever(), 
        "question": itemgetter("question")
    }
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=RagInput)
# FINAL_CHAIN_INVOKE = final_chain.astream_log({"question": "Why did Epic sue Apple?"})
