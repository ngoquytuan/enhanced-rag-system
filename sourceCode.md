Tôi sẽ giúp bạn xây dựng lại ứng dụng RAG với các yêu cầu đã nêu. Dựa trên code hiện tại, tôi sẽ tạo một phiên bản cải tiến hỗ trợ nhiều loại file hơn và sử dụng API cho LLM.Tôi sẽ xây dựng lại ứng dụng RAG với các cải tiến theo yêu cầu của bạn. Dưới đây là các file chính được cải tiến:

## 1. Enhanced Document Loader (document_loader.py)

```python
# document_loader.py
import os
import mimetypes
from typing import List, Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

def detect_file_type(file_path: str, original_filename: str = None) -> str:
    """
    Detect file type from extension or MIME type.
    Falls back to original filename if available.
    """
    # First try to get extension from the file path
    _, file_extension = os.path.splitext(file_path)
    
    # If no extension and we have original filename, use that
    if not file_extension and original_filename:
        _, file_extension = os.path.splitext(original_filename)
    
    # If still no extension, try MIME type detection
    if not file_extension:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            extension_map = {
                'application/pdf': '.pdf',
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'text/plain': '.txt',
                'text/markdown': '.md',
                'text/x-python': '.py',
                'text/html': '.html',
                'text/css': '.css',
                'application/javascript': '.js',
            }
            file_extension = extension_map.get(mime_type, '.txt')
    
    return file_extension.lower()

def get_document_loader(file_path: str, original_filename: str = None):
    """
    Selects the appropriate document loader based on the file extension.
    Supports: PDF, DOC/DOCX, TXT, MD, PY, HTML, and other text files.
    """
    file_extension = detect_file_type(file_path, original_filename)
    
    logger.info(f"Loading file with extension: {file_extension}")
    
    try:
        if file_extension == ".pdf":
            return PyPDFLoader(file_path)
        elif file_extension in [".doc", ".docx"]:
            return UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".html":
            return UnstructuredHTMLLoader(file_path)
        elif file_extension == ".md":
            return UnstructuredMarkdownLoader(file_path)
        elif file_extension in [".txt", ".py", ".js", ".css", ".json", ".xml"]:
            # Handle code files and text files with UTF-8 encoding
            return TextLoader(file_path, encoding="utf-8")
        else:
            # Fallback: try to read as text file
            logger.warning(f"Unknown file extension {file_extension}, treating as text file")
            return TextLoader(file_path, encoding="utf-8")
            
    except Exception as e:
        logger.error(f"Error creating loader for {file_path}: {e}")
        # Final fallback
        return TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)

def load_documents(file_paths: List[tuple]) -> List[Document]:
    """
    Loads documents from multiple files using the appropriate loaders.
    
    Args:
        file_paths: List of tuples (file_path, original_filename)
        
    Returns:
        List of loaded documents
    """
    all_documents = []
    
    for file_path, original_filename in file_paths:
        try:
            logger.info(f"Loading document: {original_filename or file_path}")
            loader = get_document_loader(file_path, original_filename)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source_file'] = original_filename or os.path.basename(file_path)
                doc.metadata['file_type'] = detect_file_type(file_path, original_filename)
            
            all_documents.extend(documents)
            logger.info(f"Successfully loaded {len(documents)} documents from {original_filename}")
            
        except Exception as e:
            logger.error(f"Error loading {original_filename}: {e}")
            continue
    
    return all_documents
```

## 2. Enhanced RAG with Sentence Transformer (rag_sentence_transformer_api.py)

```python
# rag_sentence_transformer_api.py
import json
import os
import shutil
from typing import List, Tuple, Optional
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from document_loader import load_documents
from chat_memory import ChatMemoryManager
from llm_providers import get_llm_from_provider
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDFSentenceTransformerAPI:
    """Enhanced RAG system using SentenceTransformer embeddings and external LLM API."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with SentenceTransformer embeddings and API-based LLM.
        
        Args:
            embedding_model: Name of the SentenceTransformer model to use
        """
        try:
            # Initialize LLM from API provider
            self.model = get_llm_from_provider()
            logger.info("LLM API provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise

        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        logger.info(f"Using SentenceTransformer model: {embedding_model}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize text splitter
        chunk_size = self.config.get("chunk_size", 1024)
        chunk_overlap = self.config.get("chunk_overlap", 100)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Initialize prompt
        self.prompt = self._create_prompt()
        
        # Initialize components
        self.vector_store = None
        self.retriever = None
        self.documents = []
        self.memory = ChatMemoryManager()
        
        logger.info("RAG system initialized successfully")

    def _load_config(self) -> dict:
        """Load configuration from config.json or use defaults."""
        try:
            with open("config.json") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("config.json not found, using default configuration")
            return {
                "chunk_size": 1024,
                "chunk_overlap": 100
            }

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template."""
        try:
            with open("prompt.json") as f:
                prompt_config = json.load(f)
                template = prompt_config.get("prompt_template")
        except FileNotFoundError:
            logger.warning("prompt.json not found, using default prompt")
            template = """You are a helpful AI assistant. Use the following context to answer the user's question accurately and comprehensively.

Context:
{context}

Previous conversation:
{chat_history}

Question: {question}

Please provide a detailed and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question completely, please state that clearly."""

        return ChatPromptTemplate.from_template(template)

    def ingest(self, file_info: List[tuple]):
        """
        Ingest multiple files, split their contents, and store embeddings.
        
        Args:
            file_info: List of tuples (file_path, original_filename)
        """
        logger.info(f"Starting ingestion for {len(file_info)} files")
        
        try:
            # Load all documents
            self.documents = load_documents(file_info)
            
            if not self.documents:
                raise ValueError("No documents were successfully loaded")
            
            logger.info(f"Loaded {len(self.documents)} document sections")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(self.documents)
            chunks = filter_complex_metadata(chunks)
            
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="chroma_db_sentence_transformer_api",
            )
            
            # Reset retriever to use new vector store
            self.retriever = None
            
            logger.info("Document ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            raise

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2) -> Tuple[str, List[str]]:
        """
        Answer a query using the RAG pipeline with conversation memory.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            Tuple of (answer, source_references)
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest documents first.")

        logger.info(f"Processing query: {query}")
        
        try:
            # Initialize retriever if not exists
            if not self.retriever:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k},
                )

            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(query)
            
            if not retrieved_docs:
                response = "I couldn't find relevant information in the documents to answer your question."
                self.memory.add_user_message(query)
                self.memory.add_ai_message(response)
                return response, []

            # Format context and get chat history
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            chat_history = self.memory.get_memory_context()
            
            # Prepare input for the LLM
            formatted_input = {
                "context": context,
                "chat_history": chat_history,
                "question": query,
            }

            # Build and execute the RAG chain
            chain = (
                RunnablePassthrough()
                | self.prompt
                | self.model
                | StrOutputParser()
            )

            logger.info("Generating response using LLM API")
            response = chain.invoke(formatted_input)
            
            # Update conversation memory
            self.memory.add_user_message(query)
            self.memory.add_ai_message(response)
            
            # Extract source references
            sources = list(set([
                doc.metadata.get('source_file', 'Unknown') 
                for doc in retrieved_docs
            ]))
            
            logger.info("Response generated successfully")
            return response, sources
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = f"An error occurred while processing your question: {str(e)}"
            return error_msg, []

    def get_document_summary(self) -> str:
        """Get a summary of ingested documents."""
        if not self.documents:
            return "No documents have been ingested yet."
        
        file_types = {}
        total_chars = 0
        
        for doc in self.documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            source_file = doc.metadata.get('source_file', 'unknown')
            
            if source_file not in file_types:
                file_types[source_file] = {
                    'type': file_type,
                    'chars': 0
                }
            
            file_types[source_file]['chars'] += len(doc.page_content)
            total_chars += len(doc.page_content)
        
        summary = f"**Document Summary:**\n"
        summary += f"- Total files: {len(file_types)}\n"
        summary += f"- Total content: {total_chars:,} characters\n\n"
        summary += "**Files:**\n"
        
        for filename, info in file_types.items():
            summary += f"- {filename} ({info['type']}): {info['chars']:,} chars\n"
        
        return summary

    def clear(self):
        """Reset the vector store, retriever, and memory."""
        logger.info("Clearing RAG system")
        
        # Clear vector store and retriever
        if self.vector_store:
            del self.vector_store
        if self.retriever:
            del self.retriever
            
        self.vector_store = None
        self.retriever = None
        self.documents = []
        
        # Clear memory
        self.memory.clear_memory()
        
        # Remove persistent storage
        if os.path.exists("chroma_db_sentence_transformer_api"):
            try:
                shutil.rmtree("chroma_db_sentence_transformer_api")
                logger.info("Chroma DB cleared successfully")
            except PermissionError as e:
                logger.warning(f"Could not clear Chroma DB: {e}")
```

## 3. Updated Main Application (app_main.py)

```python
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
```

## 4. Requirements.txt

```txt
streamlit
streamlit-chat
langchain
langchain-community
langchain-core
langchain-groq
langchain-google-genai
langchain-openai
chromadb
sentence-transformers
python-dotenv
pypdf
python-docx
unstructured[html,md]
ollama
```

## 5. Default Configuration Files

**config.json:**
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 1024,
  "chunk_overlap": 100,
  "pipeline": "sentence_transformer_api"
}
```

**prompt.json:**
```json
{
  "prompt_template": "You are a helpful AI assistant with expertise in analyzing documents. Use the following context to answer the user's question accurately and comprehensively.\n\nContext from documents:\n{context}\n\nPrevious conversation:\n{chat_history}\n\nUser Question: {question}\n\nPlease provide a detailed and accurate answer based on the context provided. If the context doesn't contain enough information to answer the question completely, please state that clearly and suggest what additional information might be needed."
}
```

## Các cải tiến chính:

1. **Hỗ trợ nhiều loại file**: PDF, DOCX, TXT, MD, PY, HTML, CSS, JS
2. **Sử dụng API LLM** thay vì local model
3. **Enhanced document loading** với metadata tracking
4. **Improved error handling** và logging
5. **Better UI/UX** với icons và status indicators
6. **Document statistics** và preview
7. **Conversation memory** với chat history
8. **Source tracking** để biết thông tin đến từ file nào

Hệ thống sử dụng python-docx để xử lý file DOCX và các loader khác nhau từ LangChain để hỗ trợ đa dạng loại file. Ứng dụng này sẽ cung cấp trải nghiệm RAG mạnh mẽ và linh hoạt hơn so với phiên bản trước.