# chat_memory.py
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatMemoryManager:
    """
    Manages conversation memory for the RAG chatbot.
    Keeps track of conversation history and context.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize the memory manager.
        
        Args:
            window_size: Number of recent message pairs to remember
        """
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="chat_history"
        )
        self.conversation_context = []
        self.current_session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_user_message(self, message: str):
        """Add a user message to memory."""
        self.memory.chat_memory.add_user_message(message)
        self.conversation_context.append({
            "type": "human",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_ai_message(self, message: str):
        """Add an AI message to memory."""
        self.memory.chat_memory.add_ai_message(message)
        self.conversation_context.append({
            "type": "ai",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.conversation_context
    
    def get_memory_context(self) -> str:
        """Get conversation context as formatted string for the prompt."""
        if not self.conversation_context:
            return ""
        
        context_parts = []
        # Get last few exchanges for context
        recent_messages = self.conversation_context[-6:]  # Last 3 exchanges
        
        for msg in recent_messages:
            if msg["type"] == "human":
                context_parts.append(f"User: {msg['content']}")
            else:
                context_parts.append(f"Assistant: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def clear_memory(self):
        """Clear all conversation memory."""
        self.memory.clear()
        self.conversation_context = []
        self.current_session_id = self._generate_session_id()
        logger.info("Conversation memory cleared")
    
    def get_session_id(self) -> str:
        """Get current session ID."""
        return self.current_session_id
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export conversation for saving to history."""
        return {
            "session_id": self.current_session_id,
            "messages": self.conversation_context,
            "created_at": datetime.now().isoformat()
        }