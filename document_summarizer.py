# document_summarizer.py
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from llm_providers import get_llm_from_provider
import logging

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """
    A class to handle document summarization using an external LLM.
    """

    def __init__(self):
        """
        Initialize the summarizer.
        """
        try:
            self.model = get_llm_from_provider()
            logger.info("LLM provider for summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider for summarizer: {e}")
            raise
            
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in summarizing documents. 
            Based on the following text, provide a concise summary (around 150-200 words) 
            that captures the main points, key arguments, and important conclusions.

            Text:
            {document_text}

            Concise Summary:
            """
        )
        
        self.chain = self.prompt | self.model | StrOutputParser()

    def summarize(self, document_text: str) -> str:
        """
        Generate a summary for the given document text.

        Args:
            document_text: The text of the document to summarize.

        Returns:
            The generated summary.
        """
        if not document_text or not document_text.strip():
            logger.warning("Document text is empty, cannot generate summary.")
            return "The document is empty or could not be read."
            
        logger.info(f"Generating summary for document of length {len(document_text)}...")
        
        try:
            summary = self.chain.invoke({"document_text": document_text})
            logger.info("Successfully generated summary.")
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return f"An error occurred during summarization: {e}"
