# app_main.py
import os
import tempfile
import time
import json
import streamlit as st
from streamlit_chat import message
from rag_sentence_transformer_api import ChatPDFSentenceTransformerAPI
from document_summarizer import DocumentSummarizer
from chat_history import ChatHistoryManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Enhanced RAG with Sentence Transformers", 
    page_icon="📚",
    layout="wide"
)

def load_config():
    """Load configuration from config.json."""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("config.json not found. Please create configuration file.")
        return {}
    except json.JSONDecodeError:
        st.error("Error decoding config.json. Please check its format.")
        return {}

def display_messages():
    """Display the chat history."""
    st.subheader("💬 Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        
        with st.session_state["thinking_spinner"], st.spinner("🤔 Thinking..."):
            try:
                agent_text, sources = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
                
                # Add source information if available
                if sources:
                    source_text = "\n\n**Sources:** " + ", ".join(sources)
                    agent_text += source_text
                    
            except ValueError as e:
                agent_text = f"❌ Error: {str(e)}"
            except Exception as e:
                agent_text = f"❌ An error occurred: {str(e)}"
                logger.error(f"Error processing input: {e}")

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state["document_summary"] = ""

    file_info = []
    temp_files = []

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tf:
            tf.write(file.getbuffer())
            file_info.append((tf.name, file.name))
            temp_files.append(tf.name)

    if file_info:
        with st.session_state["ingestion_spinner"], st.spinner(f"📄 Ingesting {len(file_info)} files..."):
            try:
                t0 = time.time()
                st.session_state["assistant"].ingest(file_info)
                t1 = time.time()
                
                st.success(f"✅ Successfully ingested {len(file_info)} files in {t1 - t0:.2f} seconds")
                
                # Generate document summary
                with st.spinner("📝 Generating document summary..."):
                    summary = st.session_state["assistant"].get_document_summary()
                    st.session_state["document_summary"] = summary
                
            except Exception as e:
                st.error(f"❌ Error ingesting files: {e}")
                logger.error(f"Ingestion error: {e}")
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_file}: {e}")

def main():
    """Main application function."""
    # Initialize session state
    if "assistant" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["document_summary"] = ""
        
        try:
            config = load_config()
            embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
            st.session_state["assistant"] = ChatPDFSentenceTransformerAPI(embedding_model=embedding_model)
            st.session_state["summarizer"] = DocumentSummarizer()
            st.session_state["history_manager"] = ChatHistoryManager()
            
        except Exception as e:
            st.error(f"❌ Failed to initialize the application: {e}")
            st.stop()

    # Header
    st.title("📚 Enhanced RAG with Sentence Transformers")
    st.markdown("*Supports PDF, DOCX, TXT, MD, PY, HTML files with advanced RAG capabilities*")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, MD, PY, HTML)",
            type=["pdf", "doc", "docx", "txt", "md", "py", "html", "css", "js"],
            key="file_uploader",
            on_change=read_and_save_file,
            accept_multiple_files=True,
            help="Support for multiple file formats including code files"
        )

        st.session_state["ingestion_spinner"] = st.empty()

        st.header("⚙️ Settings")
        st.session_state["retrieval_k"] = st.slider(
            "Number of Retrieved Results (k)", 
            min_value=1, max_value=15, value=5,
            help="More results provide broader context but may include less relevant information"
        )
        st.session_state["retrieval_threshold"] = st.slider(
            "Similarity Score Threshold", 
            min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="Higher threshold = more relevant but fewer results"
        )

        # Chat History Management
        st.header("💾 Chat History")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Session", use_container_width=True):
                try:
                    session_data = st.session_state["assistant"].memory.export_conversation()
                    st.session_state["history_manager"].save_session(session_data)
                    st.success("✅ Session saved!")
                except Exception as e:
                    st.error(f"❌ Error saving session: {e}")
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["document_summary"] = ""
                st.session_state["assistant"].clear()
                st.rerun()

        # Session History
        sessions = st.session_state["history_manager"].list_sessions()
        if sessions:
            st.subheader("📜 Previous Sessions")
            for session in sessions[:5]:  # Show last 5 sessions
                with st.expander(f"💬 {session['session_id'][:20]}... ({session['message_count']} msgs)"):
                    st.write(f"📅 **Created:** {session['created_at'][:19]}")
                    st.write(f"💬 **Preview:** {session['preview'][:100]}...")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📂 Load", key=f"load_{session['session_id']}"):
                            try:
                                loaded_session = st.session_state["history_manager"].load_session(session['session_id'])
                                st.session_state["messages"] = [
                                    (msg['content'], msg['type'] == 'human') 
                                    for msg in loaded_session['messages']
                                ]
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error loading session: {e}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{session['session_id']}"):
                            try:
                                st.session_state["history_manager"].delete_session(session['session_id'])
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting session: {e}")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Document summary
        if st.session_state["document_summary"]:
            with st.expander("📊 Document Summary", expanded=True):
                st.markdown(st.session_state["document_summary"])

        # Chat interface
        display_messages()
        
        # Input
        st.text_input(
            "💬 Ask a question about your documents...", 
            key="user_input", 
            on_change=process_input,
            placeholder="Type your question here and press Enter"
        )
    
    with col2:
        # Statistics and info
        st.header("📈 Statistics")
        if hasattr(st.session_state["assistant"], 'documents') and st.session_state["assistant"].documents:
            st.metric("📄 Documents", len(set(doc.metadata.get('source_file', 'unknown') for doc in st.session_state["assistant"].documents)))
            st.metric("📝 Text Sections", len(st.session_state["assistant"].documents))
            total_chars = sum(len(doc.page_content) for doc in st.session_state["assistant"].documents)
            st.metric("🔤 Total Characters", f"{total_chars:,}")
        
        st.header("ℹ️ Supported Files")
        st.markdown("""
        - **📄 PDF** - Portable documents
        - **📝 DOCX** - Word documents  
        - **📋 TXT** - Plain text files
        - **📑 MD** - Markdown files
        - **🐍 PY** - Python source code
        - **🌐 HTML** - Web pages
        - **🎨 CSS** - Stylesheets
        - **⚡ JS** - JavaScript files
        """)

if __name__ == "__main__":
    main()