# rag_sentence_transformer_api.py
import json
import os
import shutil
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
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

class CustomSentenceTransformerEmbeddings:
    """Custom wrapper for SentenceTransformer with better error handling."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            self.model_name = model_name
            logger.info(f"Successfully loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to default model: all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model_name = "all-MiniLM-L6-v2"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

class ChatPDFSentenceTransformerAPI:
    """Enhanced RAG system using SentenceTransformer embeddings and external LLM API."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with SentenceTransformer embeddings and API-based LLM.
        
        Args:
            embedding_model: Name of the SentenceTransformer model to use
        """
        logger.info(f"Initializing RAG system with embedding model: {embedding_model}")
        
        try:
            # Initialize LLM from API provider
            self.model = get_llm_from_provider()
            logger.info("LLM API provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise

        # Initialize embeddings with better error handling
        try:
            self.embeddings = CustomSentenceTransformerEmbeddings(embedding_model)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            logger.info("Falling back to default embedding model")
            self.embeddings = CustomSentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        
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
            with open("config.json", encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("config.json not found, using default configuration")
            return {
                "chunk_size": 1024,
                "chunk_overlap": 100
            }
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading config.json: {e}")
            return {
                "chunk_size": 1024,
                "chunk_overlap": 100
            }

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template."""
        try:
            with open("prompt.json", encoding='utf-8') as f:
                prompt_config = json.load(f)
                template = prompt_config.get("prompt_template")
                #template = 
        except (FileNotFoundError, UnicodeDecodeError) as e:
            logger.warning(f"Error loading prompt.json: {e}, using default prompt")
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
            
            # Filter out error documents for processing
            valid_documents = [doc for doc in self.documents if not doc.metadata.get('error', False)]
            
            if not valid_documents:
                raise ValueError("No valid documents found after filtering errors")
            
            logger.info(f"Loaded {len(valid_documents)} valid document sections (out of {len(self.documents)} total)")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(valid_documents)
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
        error_count = 0
        
        for doc in self.documents:
            if doc.metadata.get('error', False):
                error_count += 1
                continue
                
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
        summary += f"- Valid files: {len(file_types)}\n"
        if error_count > 0:
            summary += f"- Files with errors: {error_count}\n"
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