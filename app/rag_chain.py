from dotenv import load_dotenv
import sys
import os
from typing import TypedDict
from operator import itemgetter
sys.path.insert(1, '/Users/ma012/Documents/Morit/10_ResearchProjects/08_ESGReader/Prototype')
from config import PG_COLLECTION_NAME, CHAT_LLM, HF_REPO_ID, POSTGRES_CONNECTION
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from importer.load_and_process import get_embeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


load_dotenv()

embeddings = get_embeddings()
vector_store = PGVector(
    collection_name=PG_COLLECTION_NAME,
    connection_string=POSTGRES_CONNECTION,
    embedding_function=embeddings
)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

if CHAT_LLM=="openAI":
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", streaming=True)

elif CHAT_LLM=="HF":
    # llm = HuggingFaceHub(
    #     repo_id=HF_REPO_ID, model_kwargs={"temperature": 0.0, "max_length": 64}
    # )

    model_id = HF_REPO_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation",
                    model=model, 
                    tokenizer=tokenizer, 
                    pad_token_id=19,
                    max_new_tokens=20
                    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=HF_REPO_ID,
    #     task="text-generation",
    #     token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    # )

class RagInput(TypedDict):
    question: str

final_chain  = (
    {
        "context": itemgetter("question") | vector_store.as_retriever(), 
        "question": itemgetter("question")
    }
    | ANSWER_PROMPT
    | llm
).with_types(input_type=RagInput)
#final_chain = vector_store.as_retriever(itemgetter("question"))
# FINAL_CHAIN_INVOKE = final_chain.astream_log({"question": "Why did Epic sue Apple?"})
