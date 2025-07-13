"""
Main AI Assistant Application
Conversational chatbot with memory, built for future audio/environmental integration
"""

import asyncio
import ollama
import streamlit as st
from datetime import datetime
from typing import Optional, Generator
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, setup_directories
from memory_system import ConversationMemory

class AIAssistant:
    def __init__(self):
        setup_directories()
        self.memory = ConversationMemory()
        self.session_active = False
        print("🤖 AI Assistant initialized")
    
    def generate_response(self, user_message: str) -> Generator[str, None, None]:
        """Generate response using Llama 3.3 with context from memory"""
        
        # Get relevant context from memory
        context = self.memory.get_context_for_response(user_message)
        
        # Get user facts for personalization
        user_facts = self.memory.get_user_facts()
        facts_context = ""
        if user_facts:
            facts_context = "\n\nWhat I know about you:\n" + "\n".join(f"- {fact}" for fact in user_facts[:5])
        
        # Build the full prompt
        full_prompt = f"""
{config.model.system_prompt}

{context}
{facts_context}

Current message: {user_message}

Please respond naturally and conversationally. Reference relevant information from our conversation history when appropriate.
"""
        
        try:
            # Stream response from Ollama
            stream = ollama.chat(
                model=config.model.name,
                messages=[{
                    'role': 'user',
                    'content': full_prompt
                }],
                stream=True,
                options={
                    'temperature': config.model.temperature,
                }
            )
            
            response_text = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    response_text += content
                    yield content
            
            # Store the complete interaction in memory
            self.memory.add_message("user", user_message)
            self.memory.add_message("assistant", response_text)
            
            # Extract and store user facts if mentioned
            self._extract_user_facts(user_message)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            yield error_msg
            print(f"❌ Error generating response: {e}")
    
    def _extract_user_facts(self, user_message: str):
        """Extract and store facts about the user from their message"""
        # Simple fact extraction - you could enhance this with NLP
        fact_indicators = [
            "my name is", "i am", "i like", "i love", "i work", 
            "i live", "my favorite", "i enjoy", "i hate", "i don't like"
        ]
        
        lower_message = user_message.lower()
        for indicator in fact_indicators:
            if indicator in lower_message:
                # Store the relevant part as a user fact
                self.memory.store_user_fact(user_message, "extracted")
                break
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.memory.current_conversation:
            return "No conversation yet."
        
        message_count = len(self.memory.current_conversation)
        start_time = self.memory.current_conversation[0]["timestamp"]
        
        return f"Conversation started at {start_time}. {message_count} messages exchanged."
    
    def save_session(self) -> str:
        """Save current session and return filepath"""
        return self.memory.save_conversation_session()

# Streamlit Interface
def create_streamlit_interface():
    """Create the Streamlit web interface"""
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Assistant with Memory")
    st.caption("Powered by Llama 3.3 | Built for conversation, audio & environmental integration")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = AIAssistant()
        st.session_state.messages = []
    
    # Sidebar with session info and controls
    with st.sidebar:
        st.header("Session Info")
        
        if st.button("💾 Save Session"):
            filepath = st.session_state.assistant.save_session()
            st.success(f"Session saved!")
        
        if st.button("🧠 Show User Facts"):
            facts = st.session_state.assistant.memory.get_user_facts()
            if facts:
                st.write("What I know about you:")
                for fact in facts:
                    st.write(f"• {fact}")
            else:
                st.write("No facts stored yet.")
        
        if st.button("🔄 New Session"):
            st.session_state.assistant.save_session()
            st.session_state.assistant = AIAssistant()
            st.session_state.messages = []
            st.rerun()
        
        # Show conversation summary
        summary = st.session_state.assistant.get_conversation_summary()
        st.write("**Session Summary:**")
        st.write(summary)
        
        # Future integration indicators
        st.header("Future Features")
        st.write("🎤 Audio: Ready for integration")
        st.write("📹 Environment: Ready for integration")
        st.write("🔒 Security: Ready for integration")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in st.session_state.assistant.generate_response(prompt):
                full_response += chunk
                response_placeholder.write(full_response + "▋")
            
            response_placeholder.write(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Terminal Interface (alternative)
def create_terminal_interface():
    """Create a terminal-based interface"""
    assistant = AIAssistant()
    
    print("🤖 AI Assistant Terminal Interface")
    print("=" * 50)
    print("Type 'quit' to exit, 'save' to save session, 'facts' to see user facts")
    print()
    
    try:
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                assistant.save_session()
                print("👋 Session saved. Goodbye!")
                break
            
            elif user_input.lower() == 'save':
                filepath = assistant.save_session()
                print(f"💾 Session saved to: {filepath}")
                continue
            
            elif user_input.lower() == 'facts':
                facts = assistant.memory.get_user_facts()
                if facts:
                    print("\n🧠 What I know about you:")
                    for fact in facts:
                        print(f"  • {fact}")
                else:
                    print("🧠 No facts stored yet.")
                continue
            
            elif not user_input:
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Generate response with streaming
            full_response = ""
            for chunk in assistant.generate_response(user_input):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()  # New line after response
            
    except KeyboardInterrupt:
        assistant.save_session()
        print("\n\n👋 Session saved. Goodbye!")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting AI Assistant...")
    
    # Check if Ollama is available
    try:
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        
        if config.model.name not in available_models:
            print(f"❌ Model {config.model.name} not found!")
            print(f"Available models: {available_models}")
            print(f"Please run: ollama pull {config.model.name}")
            sys.exit(1)
        
        print(f"✅ Using model: {config.model.name}")
        
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running!")
        sys.exit(1)
    
    # Choose interface
    if config.interface_type == "streamlit":
        print("🌐 Starting Streamlit interface...")
        print("Open your browser to the URL shown below:")
        create_streamlit_interface()
    else:
        create_terminal_interface()