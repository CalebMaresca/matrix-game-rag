import chainlit as cl
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
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
from dotenv import load_dotenv
import asyncio

load_dotenv()

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

split_chunks = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(model="CalebMaresca/matrix-game-embeddings-ft-v1")
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


@cl.on_chat_start
async def start_chat():
     pass


@cl.on_message
async def main(message: cl.Message):

    # Convert OpenAI format messages to LangChain format
    chat_history = cl.chat_context.to_openai()
    
    msg = cl.Message(content="")
    await msg.send()
    # msg.content will be built by streaming tokens.
    # stream_token will call send() on the first token.
    try:
        print(f"Entering agent_graph.astream loop for message: '{message.content}'")
        # Assuming agent_graph.astream yields (chunk, metadata_dict)
        async for token_chunk, metadata in agent_graph.astream(
                    {'messages': chat_history},
                    stream_mode="messages" # This mode should yield AIMessageChunk for create_react_agent
                ):
            # More detailed print for debugging what astream yields
            chunk_content_for_log = getattr(token_chunk, 'content', 'N/A (no content attr)')
            metadata_keys_for_log = list(metadata.keys()) if isinstance(metadata, dict) else 'N/A (metadata not a dict)'
            print(f"Yielded from astream: chunk_content='{chunk_content_for_log}', metadata_keys='{metadata_keys_for_log}'")
            
            # Check if 'langgraph_node' is in metadata (and metadata is a dict) and equals 'agent'
            # Also ensure token_chunk has a 'content' attribute and it's not empty
            if isinstance(metadata, dict) and metadata.get('langgraph_node') == 'agent' and \
               hasattr(token_chunk, 'content') and token_chunk.content:
                await msg.stream_token(token_chunk.content)
            
            # Yield control to the event loop.
            # This can help ensure other tasks (like WebSocket sending) get a chance to run.
            await asyncio.sleep(0)
    except Exception as e:
        # THIS WILL CAPTURE THE ERROR IN YOUR APPLICATION
        print(f"ERROR during agent_graph.astream for message '{message.content}': {type(e).__name__} - {e}")
                    
    if msg.streaming:
        await msg.update()
    elif not msg.content:
        pass
