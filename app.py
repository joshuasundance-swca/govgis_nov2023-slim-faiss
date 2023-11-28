import os
from operator import itemgetter
from typing import Optional

import streamlit as st
import yaml
from huggingface_hub import hf_hub_download
from langchain.chat_models import ChatAnthropic
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import FAISS

DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 512
DEFAULT_SEARCH_RESULT_LIMIT = 3
default_hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
HF_HOME = os.environ.get("HF_HOME", default_hf_home)

if "chain" not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.session_state.search_result_limit = st.slider(
        "Search Result Limit",
        min_value=1,
        max_value=10,
        value=DEFAULT_SEARCH_RESULT_LIMIT,
        step=1,
    )

    st.session_state.anthropic_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
    )

    st.session_state.max_tokens = st.slider(
        "Max Tokens",
        min_value=512,
        max_value=12800,
        value=DEFAULT_MAX_TOKENS,
        step=256,
    )

    st.session_state.use_instant_for_rephrase = st.checkbox(
        "Use `claude-instant-v1` to generate search query",
        value=True,
    )


@st.cache_resource
def get_embedding_model(device: str = "cpu", **kwargs) -> HuggingFaceBgeEmbeddings:
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=HF_HOME,
        **kwargs,
    )


@st.cache_data
def download_data_from_hub(**kwargs) -> str:
    repo_id = "joshuasundance/govgis_nov2023-slim-spatial"
    filename = "govgis_nov2023-slim-nospatial.faiss.bytes"
    repo_type = "dataset"
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        cache_dir=HF_HOME,
        **kwargs,
    )


@st.cache_resource
def get_faiss(
    serialized_bytes_path: Optional[str] = None,
    embeddings: Optional[HuggingFaceBgeEmbeddings] = None,
) -> FAISS:
    serialized_bytes_path = serialized_bytes_path or download_data_from_hub()
    with open(serialized_bytes_path, "rb") as infile:
        return FAISS.deserialize_from_bytes(
            embeddings=embeddings or get_embedding_model(),
            serialized=infile.read(),
        )


def _combine_documents(
    docs: list[Document],
    document_separator: str = "\n\n",
) -> str:
    return document_separator.join(f"```yaml\n{doc.page_content}\n```" for doc in docs)


rephrase_template = """Given the User Input, return an English natural language Search Query that will return the most relevant documents.
Remember, you are working with a semantic search engine. It is not based solely on keywords or Google-Fu.
Be creative with your search query.
Your entire response will be fed directly into the search engine. Omit any text that is not part of the search query.

User Input: {question}"""
REPHRASE_QUESTION_PROMPT = PromptTemplate.from_template(rephrase_template)


answer_template = """The following search results were found for the given user query.
Provide a description of the relevant search results, providing relevant URLs and details.
Describing the search results in the context of the query is more important than answering the query.
Do not answer without referring to the search results; the search results are the most important part of the answer.
Base your response on the search results.
Always provide a URL when referencing a specific service, dataset, or API.
If multiple search results are relevant to the user's query, describe each result separately.
Describe what sets each result apart from the others.
Be detailed and specific, so the user can find the information they need.
Format your response as markdown as appropriate.
----------------
Search Results:
{context}
----------------
Question: {question}"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)


def get_chain(rephrase_llm, answer_llm, retriever):
    """
    Return a chain that rephrases, retrieves, and responds.

    Output keys:
    - search_query: str
    - docs: list[Document]
    - answer: str
    """
    return (
        # rephrase
        REPHRASE_QUESTION_PROMPT
        | rephrase_llm
        | {"search_query": StrOutputParser()}
        # retrieve
        | {
            "search_query": itemgetter("search_query"),
            "docs": itemgetter("search_query") | retriever,
            "question": itemgetter("search_query"),
        }
        # respond
        | {
            "search_query": itemgetter("search_query"),
            "docs": itemgetter("docs"),
            "answer": (
                {
                    "context": (lambda x: _combine_documents(x["docs"])),
                    "question": itemgetter("question"),
                }
                | ANSWER_PROMPT
                | answer_llm
                | StrOutputParser()
            ),
        }
    )


db = get_faiss()
retriever = db.as_retriever(
    search_kwargs={"k": st.session_state.search_result_limit},
)

if st.session_state.anthropic_api_key:
    rephrase_llm = ChatAnthropic(
        model="claude-instant-v1"
        if st.session_state.use_instant_for_rephrase
        else "claude-2.1",
        temperature=st.session_state.temperature,
        max_tokens_to_sample=512,
        anthropic_api_key=st.session_state.anthropic_api_key,
    )

    answer_llm = ChatAnthropic(
        model="claude-2.1",
        temperature=st.session_state.temperature,
        max_tokens_to_sample=st.session_state.max_tokens,
        anthropic_api_key=st.session_state.anthropic_api_key,
    )

    st.session_state.chain = get_chain(rephrase_llm, answer_llm, retriever)


user_input = st.text_input(
    "What are you looking for?",
    value="",
)

doc_md = """## [{name}]({url})

### Type
{type}

### Description
{description}

### Parent Service Description
{parent_service_description}

### Fields
{fields}
"""


def display_docs(docs: list[Document]) -> None:
    missing_value = ""
    for doc in docs:
        data = yaml.safe_load(doc.page_content)
        st.markdown(f"## [{data['name']}]({data['url']})")
        st.markdown(f"### Type\n{data['type']}")
        st.markdown("### Description")
        st.components.v1.html(data.get("description", missing_value))
        st.markdown("### Parent Service Description")
        st.components.v1.html(data.get("parent_service_description", missing_value))
        if data.get("fields", None):
            st.markdown("### Fields")
            for field in data["fields"]:
                st.markdown(f"- {field}")


if user_input:
    if st.session_state.chain is not None:
        result = st.session_state.chain.invoke(dict(question=user_input))
        st.markdown("# Query")
        st.markdown(result["search_query"])
        st.markdown("# Answer")
        st.markdown(result["answer"])
        st.markdown("# Documents")
        display_docs(result["docs"])
    else:
        results = retriever.invoke(user_input)
        display_docs(results)
