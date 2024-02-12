import tru_shared

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import glob
import os

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS_2")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS_2")

framework = tru_shared.Framework.LANG_CHAIN

chatModel = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.get_azure_embeddings_model(framework)

pdf_datasets = ["history_of_alexnet"]

collection_names = [
    "PyPDFLoader",
    "PDFMinerLoader_by_page"
    "LayoutPDFReader_base",
    "UnstructuredFileLoader_single"
]

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

for collection_name in collection_names:
    vector_store = tru_shared.get_astra_vector_store(framework, collection_name)
    pipeline = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | chatModel
        | StrOutputParser()
    )

    tru_shared.execute_experiment(
        framework, pipeline, collection_name, pdf_datasets)
