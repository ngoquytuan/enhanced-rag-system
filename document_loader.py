# document_loader.py
import os
import mimetypes
import chardet
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

class SafeTextLoader(TextLoader):
    """Enhanced TextLoader with better encoding detection and handling."""
    
    def __init__(self, file_path: str, encoding: str = None, autodetect_encoding: bool = True):
        self.autodetect_encoding = autodetect_encoding
        if encoding is None and autodetect_encoding:
            encoding = self._detect_encoding(file_path)
        super().__init__(file_path, encoding=encoding)
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                logger.info(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
                
                # Fallback to utf-8 if confidence is too low
                if confidence < 0.7:
                    logger.warning(f"Low confidence in encoding detection, using utf-8")
                    return 'utf-8'
                    
                return encoding
        except Exception as e:
            logger.warning(f"Error detecting encoding for {file_path}: {e}, using utf-8")
            return 'utf-8'
    
    def load(self) -> List[Document]:
        """Load with enhanced error handling."""
        try:
            return super().load()
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode error with {self.encoding}, trying alternatives: {e}")
            
            # Try common encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            if self.encoding in encodings_to_try:
                encodings_to_try.remove(self.encoding)
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Trying encoding: {encoding}")
                    temp_loader = TextLoader(self.file_path, encoding=encoding)
                    return temp_loader.load()
                except UnicodeDecodeError:
                    continue
            
            # Final fallback: read with errors='replace'
            logger.warning(f"All encodings failed, using utf-8 with error replacement")
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            metadata = {"source": self.file_path}
            return [Document(page_content=content, metadata=metadata)]

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
            try:
                return UnstructuredMarkdownLoader(file_path)
            except Exception as e:
                logger.warning(f"UnstructuredMarkdownLoader failed, using SafeTextLoader: {e}")
                return SafeTextLoader(file_path)
        elif file_extension in [".txt", ".py", ".js", ".css", ".json", ".xml", ".log"]:
            return SafeTextLoader(file_path)
        else:
            # Fallback: try to read as text file
            logger.warning(f"Unknown file extension {file_extension}, treating as text file")
            return SafeTextLoader(file_path)
            
    except Exception as e:
        logger.error(f"Error creating loader for {file_path}: {e}")
        # Final fallback
        return SafeTextLoader(file_path)

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
            # Try to create a placeholder document with error info
            error_doc = Document(
                page_content=f"Error loading file {original_filename}: {str(e)}",
                metadata={
                    'source_file': original_filename or os.path.basename(file_path),
                    'file_type': detect_file_type(file_path, original_filename),
                    'error': True
                }
            )
            all_documents.append(error_doc)
            continue
    
    return all_documents