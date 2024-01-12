
from e2e_tests.langchain.rag_application import (
    BASIC_QA_PROMPT,
    SAMPLE_DATA,
    format_docs,
)

from langchain.schema.vectorstore import VectorStore
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


def _config(engine, model) -> str:
    return f"""
    models:
      - type: main
        engine: {engine}
        model: {model}
    
    rails:
      input:
        flows:
          - self check input

    prompts:
      - task: self_check_input
        content: |
          Your task if to check if the user message below complies with the policy below.
                
          Policy for user messages:
          - should not contain any fruits

          User message: "{{ user_input }}"

          Question: Should the user message be blocked (Yes or No)?
          Answer:
    """


def _create_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    prompt = PromptTemplate.from_template(BASIC_QA_PROMPT)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def _try_runnable_rails(
    config: RailsConfig, llm: BaseLanguageModel, retriever: BaseRetriever
) -> None:
    guardrails = RunnableRails(config)
    chain = _create_chain(llm, retriever)
    chain_with_rails = guardrails | chain

    response = chain_with_rails.invoke(
        {"input": "When was MyFakeProductForTesting released for the first time?"}
    )
    assert "2020" in response

    response = chain_with_rails.invoke({"input": "What color is an apple?"})
    assert "I'm sorry, I can't respond to that" in response


def run_nemo_guardrails(
    vector_store: VectorStore, llm: BaseLanguageModel, config: dict
) -> None:
    retriever = vector_store.as_retriever()
    vector_store.add_texts(SAMPLE_DATA)

    model_config = _config(config["engine"], config["model"])
    rails_config = RailsConfig.from_content(yaml_content=model_config)
    _try_runnable_rails(config=rails_config, llm=llm, retriever=retriever)
