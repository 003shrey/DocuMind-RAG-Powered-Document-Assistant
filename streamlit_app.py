import os
import uuid
import streamlit as st
from dotenv import load_dotenv

if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from rag_utils import (
    load_doc_to_db, 
    load_url_to_db, 
    stream_llm_response, 
    stream_llm_rag_response
)

load_dotenv()
os.environ["USER_AGENT"] = "documind-agent"

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    .stTitle {
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        color: #6c757d;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=unsafe_allow_html)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_sidebar():
    with st.sidebar:
        st.markdown("### Model Configuration")
        
        provider = st.selectbox(
            "Provider",
            ["OpenAI", "GROQ", "Anthropic"],
            index=0
        )
        
        if provider == "OpenAI":
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "o3-mini"], index=0)
            api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "GROQ":
            model = st.selectbox("Model", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "qwen-2.5-32b"], index=0)
            api_key = st.text_input("GROQ API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
        elif provider == "Anthropic":
            model = st.selectbox("Model", ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"], index=0)
            api_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key

        st.divider()
        st.markdown("### Document Sources")
        
        st.file_uploader(
            "Upload Files", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )
        
        st.text_input(
            "Add Web URL", 
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )
        
        is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
        
        if st.session_state.rag_sources:
            with st.expander(f"Loaded Sources ({len(st.session_state.rag_sources)})"):
                for src in st.session_state.rag_sources:
                    st.text(f"• {src}")
                    
        st.divider()
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.toggle("Enable RAG", value=is_vector_db_loaded, key="use_rag", disabled=not is_vector_db_loaded)
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    return {"provider": provider, "model": model}

selection = render_sidebar()

st.title("DocuMind")
st.markdown("<div class='sub-caption'>Interactive Document Intelligence & Contextual Assistant</div>", unsafe_allow_html=True)

if selection["provider"] == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
    st.info("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()
elif selection["provider"] == "GROQ" and not os.environ.get("GROQ_API_KEY"):
    st.info("Enter your GROQ API key in the sidebar to begin.")
    st.stop()
elif selection["provider"] == "Anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
    st.info("Enter your Anthropic API key in the sidebar to begin.")
    st.stop()

if selection["provider"] == "OpenAI":
    llm_stream = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model_name=selection["model"], temperature=0.2, streaming=True)
elif selection["provider"] == "GROQ":
    llm_stream = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model=selection["model"], temperature=0.2, streaming=True)
elif selection["provider"] == "Anthropic":
    llm_stream = ChatAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), model=selection["model"], temperature=0.2, streaming=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]
        
        if not st.session_state.get("use_rag", False):
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
