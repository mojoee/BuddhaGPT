from dotenv import load_dotenv
import sys
import os
from typing import TypedDict
from operator import itemgetter
cwd = os.getcwd()
sys.path.insert(1, cwd)
from config import PG_COLLECTION_NAME, CHAT_LLM, HF_REPO_ID, POSTGRES_CONNECTION
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from importer.load_and_process import get_embeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain_experimental.llms.ollama_functions import OllamaFunctions

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

template2 = """
[INST] Answer given the following context:
{context}

Question: {question}
[/INST]
"""


def get_llm(CHAT_LLM=CHAT_LLM):
    if CHAT_LLM=="openAI":
        llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4-1106-preview",
            streaming=True)
        answer_prompt = ChatPromptTemplate.from_template(template)

    elif CHAT_LLM=="ollama":
        model = OllamaFunctions(model="mistral")
        llm = ChatOpenAI(
            temperature=0,
            model=model,
            streaming=True)
        answer_prompt = ChatPromptTemplate.from_template(template)

    elif CHAT_LLM=="HF":
        # llm = HuggingFaceHub(
        #     repo_id=HF_REPO_ID, model_kwargs={"temperature": 0.0, "max_length": 64}
        # )
        from langchain import PromptTemplate, LLMChain
        import torch
        import transformers
        from transformers import AutoTokenizer
        model = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_length=200,
            max_new_tokens=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        llm = HuggingFacePipeline(pipeline=pipeline)
        template = """Given this context: {context}
        ***
        answer the following Question: {question}
        ***
        Answer: Let's think step by step."""
        answer_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        #llm_chain = LLMChain(prompt=prompt, llm=llm)

        # model_id = HF_REPO_ID
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = AutoModelForCausalLM.from_pretrained(model_id)
        # pipe = pipeline("text-generation",
        #                 model=model, 
        #                 tokenizer=tokenizer, 
        #                 max_new_tokens=10
        #                 )
        # llm = HuggingFacePipeline(pipeline=pipe)
        # answer_prompt = PromptTemplate.from_template(template2)


        # llm = HuggingFacePipeline.from_model_id(
        #     model_id=HF_REPO_ID,
        #     task="text-generation",
        #     token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        # )
    return llm, answer_prompt

class RagInput(TypedDict):
    question: str

llm, answer_prompt = get_llm()

final_chain = (
    {
        "context": itemgetter("question") | vector_store.as_retriever(), 
        "question": itemgetter("question")
    }
    | answer_prompt 
    | llm
).with_types(input_type=RagInput)

# final_chain  = (
#     {
#         "context": itemgetter("question") | vector_store.as_retriever(), 
#         "question": itemgetter("question")
#     }
#     | answer_prompt
#     | llm
# ).with_types(input_type=RagInput)
#final_chain = vector_store.as_retriever(itemgetter("question"))
# FINAL_CHAIN_INVOKE = final_chain.astream_log({"question": "Why did Epic sue Apple?"})


if __name__=="__main__":
    llm, answer_prompt = get_llm()
    print(llm(answer_prompt))