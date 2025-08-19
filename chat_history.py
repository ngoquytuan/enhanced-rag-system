# chat_history.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """
    Manages persistent storage of chat sessions.
    Saves and loads chat history to/from JSON files.
    """
    
    def __init__(self, history_dir: str = "chat_history"):
        """
        Initialize the chat history manager.
        
        Args:
            history_dir: Directory to store chat history files
        """
        self.history_dir = history_dir
        self.ensure_history_directory()
    
    def ensure_history_directory(self):
        """Create history directory if it doesn't exist."""
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            logger.info(f"Created chat history directory: {self.history_dir}")
    
    def save_session(self, session_data: Dict[str, Any]) -> str:
        """
        Save a chat session to file.
        
        Args:
            session_data: Dictionary containing session information
            
        Returns:
            str: Path to the saved file
        """
        try:
            session_id = session_data.get("session_id", "unknown")
            filename = f"{session_id}.json"
            filepath = os.path.join(self.history_dir, filename)
            
            # Add metadata
            session_data.update({
                "saved_at": datetime.now().isoformat(),
                "message_count": len(session_data.get("messages", [])),
                "version": "1.0"
            })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved chat session: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving chat session: {e}")
            raise
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific chat session.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            Dict containing session data, or None if not found
        """
        try:
            filename = f"{session_id}.json"
            filepath = os.path.join(self.history_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Session file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.info(f"Loaded chat session: {session_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Error loading chat session {session_id}: {e}")
            return None
    
    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List available chat sessions with metadata.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metadata dictionaries
        """
        try:
            sessions = []
            
            if not os.path.exists(self.history_dir):
                return sessions
            
            # Get all JSON files in history directory
            files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.history_dir, x)), reverse=True)
            
            for filename in files[:limit]:
                filepath = os.path.join(self.history_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # Extract metadata
                    session_info = {
                        "session_id": session_data.get("session_id", filename.replace('.json', '')),
                        "created_at": session_data.get("created_at", "Unknown"),
                        "saved_at": session_data.get("saved_at", "Unknown"),
                        "message_count": session_data.get("message_count", len(session_data.get("messages", []))),
                        "file_size": os.path.getsize(filepath),
                        "preview": self._get_session_preview(session_data)
                    }
                    sessions.append(session_info)
                    
                except Exception as e:
                    logger.warning(f"Error reading session file {filename}: {e}")
                    continue
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error listing chat sessions: {e}")
            return []
    
    def _get_session_preview(self, session_data: Dict[str, Any]) -> str:
        """
        Generate a preview text for a session.
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            Preview string
        """
        messages = session_data.get("messages", [])
        if not messages:
            return "Empty session"
        
        # Find first user message
        for msg in messages:
            if msg.get("type") == "human":
                content = msg.get("content", "")
                if len(content) > 100:
                    return content[:100] + "..."
                return content
        
        return "No user messages"
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            filename = f"{session_id}.json"
            filepath = os.path.join(self.history_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted chat session: {session_id}")
                return True
            else:
                logger.warning(f"Session file not found for deletion: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored chat sessions.
        
        Returns:
            Dictionary with statistics
        """
        try:
            if not os.path.exists(self.history_dir):
                return {"total_sessions": 0, "total_size": 0}
            
            files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
            total_size = sum(os.path.getsize(os.path.join(self.history_dir, f)) for f in files)
            
            total_messages = 0
            oldest_session = None
            newest_session = None
            
            for filename in files:
                filepath = os.path.join(self.history_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    total_messages += len(session_data.get("messages", []))
                    
                    created_at = session_data.get("created_at")
                    if created_at:
                        if not oldest_session or created_at < oldest_session:
                            oldest_session = created_at
                        if not newest_session or created_at > newest_session:
                            newest_session = created_at
                            
                except Exception:
                    continue
            
            return {
                "total_sessions": len(files),
                "total_messages": total_messages,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "oldest_session": oldest_session,
                "newest_session": newest_session
            }
            
        except Exception as e:
            logger.error(f"Error getting chat statistics: {e}")
            return {"error": str(e)}