import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from psycopg import connect
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_chroma import Chroma

from src.agent_comps.agent import get_sources, Agent

load_dotenv()

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

if "thread_id" not in st.session_state:
    st.session_state.thread_id = initialize_new_thread()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-2.0-flash"


st.sidebar.markdown(
    "Documents are sourced from [Boise State Art History](https://boisestate.pressbooks.pub/arthistory/)."
)

st.sidebar.title("Settings")


model_choices = ["gemini-2.0-flash", "gemini-1.5-pro", "phoenix-lite"]
selected_model = st.sidebar.selectbox(
    "Select Model:",
    options=model_choices,
    index=model_choices.index(st.session_state.get("model_name", "gemini-2.0-flash")),
    key="model_name",
)

if "model_name" not in st.session_state or st.session_state.model_name != selected_model:
    st.session_state.model_name = selected_model

st.sidebar.text_input(
    "Enter API Key:",
    value=st.session_state.api_key,
    key="api_key",
    type="password",
    help="Your API key to access the AI models."
)

if st.sidebar.button("Start New Chat", key="start_new_chat"):
    st.session_state.thread_id = initialize_new_thread()
    st.session_state.chat_history = []
    st.sidebar.success("Started a new chat session!")


st.subheader("Chat with your Art AI assistant")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

    try:
        agent = Agent(st.session_state.model_name, st.session_state.api_key)
        workflow = agent.create_agent()

        with connect(CHECKPOINT_DB_URI, **connection_kwargs) as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()

            app = workflow.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": st.session_state.thread_id}}

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
        error_message = f"An error occurred: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        response_placeholder.markdown(error_message)
