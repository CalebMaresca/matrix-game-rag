import chainlit as cl
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from rag import create_vector_search_tool
from game_designer_tool import GameDesignerTool
from prompts import RULES_SUMMARY
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langgraph.prebuilt import create_react_agent
from wikipedia_tool import WikipediaToolkit
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

# load_dotenv()

SYSTEM_PROMPT = '''
You are a helpful assistant that can aid users with tasks related to matrix games (the type of wargame).

When you require information regarding recent or historical events, do not rely only on your own knowledge, but use the Wikipedia tool.

Today's date is {current_date}.

Below is some basic information about matrix games. For more information, use the vector search tool.  
{rules_summary}
'''

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT.format(current_date=datetime.now().strftime("%B %d, %Y"), rules_summary=RULES_SUMMARY)),
    MessagesPlaceholder(variable_name="messages"),
])


@cl.on_chat_start
async def start_chat():
    settings = { # TODO: These settings might need to be passed to the Langchain model differently
        "model": "gpt-4.1-mini",
        "temperature": 0.5,
        "max_tokens": 2000,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    # Initialize Langchain components
    model = ChatOpenAI(
        model=settings["model"],
        temperature=settings["temperature"]
    )
    
    def tiktoken_len(text):
        tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(
            text,
        )
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 0,
        length_function = tiktoken_len,
    )

    # Pushing PDFs to HF-space is causing issues.
    # loader = PyMuPDFLoader("rag-data/PracticalAdviceOnMatrixGames.pdf")

    docs = []
    loader = DirectoryLoader("rag-data/", glob="*.html")
    docs.extend(loader.load())
    loader = DirectoryLoader("rag-data/", glob="*.txt")
    docs.extend(loader.load())

    docs = loader.load()

    split_chunks = text_splitter.split_documents(docs)
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = QdrantVectorStore.from_documents(
        split_chunks,
        embedding_function,
        location=":memory:",
        collection_name="matrix_game_docs",
    )

    # Initialize Wikipedia toolkit
    wikipedia_toolkit = WikipediaToolkit()
    wikipedia_tools = wikipedia_toolkit.get_tools()

    # Create the ReAct agent graph
    agent_graph = create_react_agent(
        model=model,
        prompt=prompt,
        tools=[create_vector_search_tool(vector_store, {"k": 5})] + wikipedia_tools + [GameDesignerTool()]
    )
    
    cl.user_session.set("agent_graph", agent_graph)


@cl.on_message
async def main(message: cl.Message):
    agent_graph = cl.user_session.get("agent_graph")

    if not agent_graph:
        await cl.Message(content="The agent is not initialized. Please restart the chat.").send()
        return

    # Convert OpenAI format messages to LangChain format
    chat_history = cl.chat_context.to_openai()
    
    msg = cl.Message(content="")
    # msg.content will be built by streaming tokens.
    # stream_token will call send() on the first token.
    async for token, metadata in agent_graph.astream(
                {'messages': chat_history},
                stream_mode="messages"
            ):
                
                if metadata['langgraph_node'] == 'agent':
                    await msg.stream_token(token.content)
                    
    if msg.streaming:
        await msg.update()
    elif not msg.content:
        pass
