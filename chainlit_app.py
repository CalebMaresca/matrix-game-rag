import chainlit as cl
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from rag import create_vector_search_tool
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langgraph.prebuilt import create_react_agent

# load_dotenv()


@cl.on_chat_start
async def start_chat():
    settings = { # TODO: These settings might need to be passed to the Langchain model differently
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 2000,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    # Initialize Langchain components
    model = ChatOpenAI(
        model=settings["model"],
        temperature=settings["temperature"],
        # max_tokens=settings["max_tokens"] # ChatOpenAI might not take max_tokens directly here
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

    loader = PyMuPDFLoader("data/PracticalAdviceOnMatrixGames.pdf")
    docs = loader.load()

    split_chunks = text_splitter.split_documents(docs)
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    # Create a dummy collection. You'll need to populate this with actual documents for RAG to work.
    vector_store = QdrantVectorStore.from_documents(
        split_chunks,
        embedding_function,
        location=":memory:",
        collection_name="matrix_game_docs",
    )
    # You might want to add some documents here if you have any, e.g.:
    # vector_store.add_texts(["Some initial context for the agent"])

    # Create the ReAct agent graph
    # The search_kwargs for the vector store can be customized if needed
    agent_graph = create_react_agent(
        model=model,
        tools=[create_vector_search_tool(vector_store, {"k": 5})]#,
        #checkpointer=checkpointer
    )
    
    cl.user_session.set("agent_graph", agent_graph)


@cl.on_message
async def main(message: cl.Message):
    agent_graph = cl.user_session.get("agent_graph")

    if not agent_graph:
        await cl.Message(content="The agent is not initialized. Please restart the chat.").send()
        return

    conversation_history = cl.chat_context.to_openai()

    msg = cl.Message(content="")
    # msg.content will be built by streaming tokens.
    # stream_token will call send() on the first token.
    async for token, metadata in agent_graph.astream(
                {'messages': conversation_history},
                stream_mode="messages"
            ):
                
                if metadata['langgraph_node'] == 'agent':
                    await msg.stream_token(token.content)

    # try:
    #     # Use stream_mode="messages" to get LLM tokens as MessageChunk objects
    #     async for chunk in agent_graph.astream(
    #         agent_input, {"stream_mode": "messages"}
    #     ):
    #         # chunk is expected to be a MessageChunk (e.g., AIMessageChunk)
    #         if hasattr(chunk, 'content'):
    #             token = chunk.content
    #             if token:  # Ensure there's content in the chunk
    #                 # msg.stream_token will handle sending the message shell on the first call
    #                 await msg.stream_token(token)
    #         # else:
    #             # Handle cases where the chunk might not be a MessageChunk as expected,
    #             # or if other types of events are streamed in this mode (though less likely for "messages" mode).
    #             # print(f"Received chunk without content: {chunk}")
    # except Exception as e:
    #     print(f"Error during agent stream: {e}")
    #     await cl.Message(content=f"An error occurred: {str(e)}").send()
    #     return

    # After the loop, if tokens were streamed, the message content is populated.
    # A final update might be necessary if other properties of msg need to be set,
    # or to ensure the stream is properly closed from Chainlit's perspective.
    if msg.streaming: # msg.streaming is True if stream_token was called
        await msg.update()
    elif not msg.content: # If no tokens were streamed and message is still empty
        # This case might occur if the agent produces no response or an error happened before streaming
        # (though the try-except should catch errors in astream itself).
        # Send a default message or handle as an error.
        # For now, if no content, we don't send an empty message unless explicitly handled.
        # If msg.send() was never called (because no tokens came), we might need to send something.
        # However, if there was genuinely no response, sending nothing might be intended.
        # Let's ensure an empty message isn't sent if no tokens were ever produced.
        # If an error occurred, it's handled by the except block.
        # If the agent genuinely returns no content, this will result in no message being sent
        # beyond the initial empty shell (if it were sent prior to token checks).
        # Since msg.send() is implicitly called by the first stream_token, if no tokens, no send.
        pass
