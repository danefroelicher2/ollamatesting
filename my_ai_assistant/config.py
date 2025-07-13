"""
Configuration management for AI Assistant
This centralizes all settings and makes future integrations easier
"""

import os
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ModelConfig(BaseModel):
    """LLM Model configuration"""
    name: str = "llama3.3"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: str = """You are a helpful AI assistant with a great memory. 
    You remember previous conversations and can build relationships with users.
    You're friendly, intelligent, and genuinely interested in helping.
    You will eventually have access to audio and environmental sensors, but for now focus on great conversation."""

class MemoryConfig(BaseModel):
    """Memory system configuration"""
    max_conversation_history: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    memory_persistence: bool = True
    conversation_summary_threshold: int = 10

class AudioConfig(BaseModel):
    """Audio system configuration (for future use)"""
    enabled: bool = False
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    wake_word: str = "assistant"

class EnvironmentConfig(BaseModel):
    """Environmental sensors configuration (for future use)"""
    motion_detection: bool = False
    camera_enabled: bool = False
    microphone_monitoring: bool = False
    security_mode: bool = False

class AppConfig(BaseModel):
    """Main application configuration"""
    model: ModelConfig = ModelConfig()
    memory: MemoryConfig = MemoryConfig()
    audio: AudioConfig = AudioConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    
    # Data storage paths
    data_dir: str = "data"
    conversations_dir: str = "data/conversations"
    logs_dir: str = "data/logs"
    
    # Interface settings
    interface_type: str = "streamlit"  # streamlit, terminal, or api
    debug_mode: bool = True

# Global configuration instance
config = AppConfig()

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        config.data_dir,
        config.conversations_dir,
        config.logs_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Project directories created")

if __name__ == "__main__":
    setup_directories()
    print("Configuration loaded successfully")
    print(f"Model: {config.model.name}")
    print(f"Memory enabled: {config.memory.memory_persistence}")
    print(f"Audio ready for integration: {config.audio.enabled}")
    print(f"Environment sensors ready: {config.environment.motion_detection}")