"""
Memory system for AI Assistant
Handles conversation history, user preferences, and context retention
Built to easily integrate with future audio/environmental features
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from config import config

class ConversationMemory:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.memory.embedding_model)
        self.client = chromadb.PersistentClient(path=config.data_dir + "/chroma_db")
        
        # Create or get collections
        self.conversations = self.client.get_or_create_collection("conversations")
        self.user_facts = self.client.get_or_create_collection("user_facts")
        
        # In-memory conversation for current session
        self.current_conversation = []
        
        print("✅ Memory system initialized")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to current conversation and long-term memory"""
        timestamp = datetime.now().isoformat()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        # Add to current conversation
        self.current_conversation.append(message)
        
        # Add to persistent memory
        if config.memory.memory_persistence:
            self._store_in_vector_db(message)
        
        # Keep current conversation manageable
        if len(self.current_conversation) > config.memory.max_conversation_history:
            self._summarize_old_conversation()
    
    def _store_in_vector_db(self, message: Dict):
        """Store message in vector database for semantic search"""
        try:
            # Create searchable text
            searchable_text = f"{message['role']}: {message['content']}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(searchable_text).tolist()
            
            # Store in ChromaDB
            self.conversations.add(
                embeddings=[embedding],
                documents=[searchable_text],
                metadatas=[{
                    "role": message["role"],
                    "timestamp": message["timestamp"],
                    **message.get("metadata", {})
                }],
                ids=[f"msg_{datetime.now().timestamp()}"]
            )
        except Exception as e:
            print(f"⚠️  Error storing message in vector DB: {e}")
    
    def search_memories(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search through conversation history semantically"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.conversations.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            if results['documents'][0]:
                return [
                    {
                        "content": doc,
                        "metadata": meta,
                        "distance": dist
                    }
                    for doc, meta, dist in zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )
                ]
            return []
        except Exception as e:
            print(f"⚠️  Error searching memories: {e}")
            return []
    
    def get_context_for_response(self, current_message: str) -> str:
        """Get relevant context for generating a response"""
        context_parts = []
        
        # Add recent conversation
        recent_messages = self.current_conversation[-5:]  # Last 5 messages
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")
        
        # Search for relevant memories
        relevant_memories = self.search_memories(current_message, n_results=3)
        if relevant_memories:
            context_parts.append("\nRelevant past conversations:")
            for memory in relevant_memories:
                if memory['distance'] < 0.7:  # Only include similar enough memories
                    context_parts.append(f"- {memory['content']}")
        
        return "\n".join(context_parts)
    
    def store_user_fact(self, fact: str, category: str = "general"):
        """Store a fact about the user for future reference"""
        try:
            fact_embedding = self.embedding_model.encode(fact).tolist()
            
            self.user_facts.add(
                embeddings=[fact_embedding],
                documents=[fact],
                metadatas=[{
                    "category": category,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[f"fact_{datetime.now().timestamp()}"]
            )
            print(f"✅ Stored user fact: {fact}")
        except Exception as e:
            print(f"⚠️  Error storing user fact: {e}")
    
    def get_user_facts(self, query: str = "", category: str = None) -> List[str]:
        """Retrieve facts about the user"""
        try:
            if query:
                query_embedding = self.embedding_model.encode(query).tolist()
                results = self.user_facts.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )
            else:
                # Get all facts
                results = self.user_facts.get()
            
            facts = []
            if results['documents']:
                if isinstance(results['documents'][0], list):
                    facts = results['documents'][0]
                else:
                    facts = results['documents']
            
            # Filter by category if specified
            if category and results['metadatas']:
                filtered_facts = []
                metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
                for fact, meta in zip(facts, metadatas):
                    if meta.get('category') == category:
                        filtered_facts.append(fact)
                return filtered_facts
            
            return facts
        except Exception as e:
            print(f"⚠️  Error retrieving user facts: {e}")
            return []
    
    def _summarize_old_conversation(self):
        """Summarize old parts of conversation to maintain context window"""
        # This is where you'd implement conversation summarization
        # For now, just keep the most recent messages
        self.current_conversation = self.current_conversation[-config.memory.max_conversation_history//2:]
        print("📝 Conversation summarized to maintain context window")
    
    def save_conversation_session(self, session_name: str = None):
        """Save current conversation session to file"""
        if not session_name:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = os.path.join(config.conversations_dir, f"{session_name}.json")
        
        session_data = {
            "session_name": session_name,
            "start_time": self.current_conversation[0]["timestamp"] if self.current_conversation else datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "messages": self.current_conversation,
            "message_count": len(self.current_conversation)
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✅ Conversation saved: {filepath}")
        return filepath

# Example usage for testing
if __name__ == "__main__":
    memory = ConversationMemory()
    
    # Test storing and retrieving
    memory.add_message("user", "Hi, my name is John and I love programming")
    memory.add_message("assistant", "Nice to meet you John! What kind of programming do you enjoy?")
    memory.add_message("user", "I'm really into Python and machine learning")
    
    # Store a user fact
    memory.store_user_fact("User's name is John", "personal")
    memory.store_user_fact("User loves programming, especially Python and ML", "interests")
    
    # Search memories
    print("\nSearching for 'programming':")
    results = memory.search_memories("programming")
    for result in results:
        print(f"- {result['content']}")
    
    # Get context
    print("\nContext for 'What should I learn next?':")
    context = memory.get_context_for_response("What should I learn next?")
    print(context)