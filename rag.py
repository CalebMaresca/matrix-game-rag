from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from typing import Callable, List, Sequence, Annotated
from langchain_core.documents import Document
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class RagState(TypedDict):
  messages: Annotated[list, add_messages]
  context: list[Document]

RAG_PROMPT = """\
Given a provided context and a question, you must answer the question. If you do not know the answer, you must state that you do not know.

Context:
{context}

Question:
{question}

Answer:
"""

rag_prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)


def create_retriever_node(vector_store: VectorStore, search_kwargs: dict = {"k": 5}) -> Callable:
  def retriever_node(state: RagState) -> RagState:
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(state["messages"][-1].content)
    return {"context" : retrieved_docs}
  return retriever_node


def create_generator_node(model: BaseChatModel, template: ChatPromptTemplate = rag_prompt_template) -> Callable:
  generation_chain = template | model
  def generator_node(state: RagState) -> RagState:
    response = generation_chain.invoke({"query" : state["messages"][-1].content, "context" : state["context"]})
    return {"messages" : response}
  return generator_node


def make_rag_graph(model: BaseChatModel, vector_store: VectorStore, template: ChatPromptTemplate = rag_prompt_template, search_kwargs: dict = {"k": 5}) -> StateGraph:
  retriever_node = create_retriever_node(vector_store, search_kwargs)
  generator_node = create_generator_node(model, template)

  rag_graph = StateGraph(RagState)

  rag_graph.add_node("retriever", retriever_node)
  rag_graph.add_node("generator", generator_node)

  rag_graph.set_entry_point("retriever")

  rag_graph.add_edge("retriever", "generator")
  rag_graph.add_edge("generator", END)

  return rag_graph.compile()


# For the ReAct agent, the state is typically managed by the prebuilt agent itself,
# focusing on the 'messages' list. If a specific state object like RagState is needed
# for integration, the graph's input/output would need to be adapted.
# For now, we assume the agent operates on a message-based state.

def create_vector_search_tool(vector_store: VectorStore, search_kwargs: dict) -> Callable:
  @tool("vector-search")
  def vector_search_tool(query: str) -> List[str]:
    """Searches a vector database for the given query and returns relevant document contents."""
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(query)
    return [doc.page_content for doc in retrieved_docs]
  return vector_search_tool

