import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from psycopg import connect, Connection
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_chroma import Chroma

from agent_comps.nodes import get_sources
# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

CHECKPOINT_DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

def initialize_new_thread():
    return str(uuid.uuid4())

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = initialize_new_thread()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar to start a new chat
if st.sidebar.button("Start New Chat", key="start_new_chat"):
    st.session_state.thread_id = initialize_new_thread()
    st.session_state.chat_history = []
    st.sidebar.success("Started a new chat session!")

# Main application UI
st.title("Art Rag App")
st.subheader("Chat with your AI assistant")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input and response handling
if user_input := st.chat_input("Type your message..."):
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

    try:
        # Import workflow dynamically to avoid circular dependencies
        from agent_comps.agent import workflow

        # Connect to the database
        with connect(CHECKPOINT_DB_URI, **connection_kwargs) as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()

            # Compile the workflow with the checkpointer
            app = workflow.compile(checkpointer=checkpointer)

            # Prepare config with the thread ID
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Invoke the app with the user input
            ans = app.invoke({"original_query": user_input}, config=config)

            print(ans)
            assistant_response = ans.get("generation", "Sorry, I couldn't process that.")

            if ans.get('source') == 'retrieval':
                sources = get_sources(ans)
                source_links = "\n".join([f"- [{link}]({link})" for link in sources])
                assistant_response += f"\n\n**Sources:**\n{source_links}"
            elif ans.get('source') == 'wiki':
                assistant_response += "\n\n**Source:** Wiki"

            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            response_placeholder.markdown(assistant_response)

    except Exception as e:
        # Handle errors gracefully
        error_message = f"An error occurred: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        response_placeholder.markdown(error_message)
