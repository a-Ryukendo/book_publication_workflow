"""
Configuration settings for the Automated Book Publication Workflow
"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Book Publication Workflow"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # LLM Configuration
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, env="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Model Configuration
    writer_model: str = Field(default="gpt-4", env="WRITER_MODEL")
    reviewer_model: str = Field(default="gpt-4", env="REVIEWER_MODEL")
    editor_model: str = Field(default="gpt-4", env="EDITOR_MODEL")
    
    # ChromaDB Configuration
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    chroma_collection_name: str = Field(default="book_content", env="CHROMA_COLLECTION")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./book_workflow.db", env="DATABASE_URL")
    
    # File Storage
    data_dir: str = Field(default="./data", env="DATA_DIR")
    screenshots_dir: str = Field(default="./data/screenshots", env="SCREENSHOTS_DIR")
    content_dir: str = Field(default="./data/content", env="CONTENT_DIR")
    
    # RL Configuration
    rl_learning_rate: float = Field(default=0.001, env="RL_LEARNING_RATE")
    rl_discount_factor: float = Field(default=0.99, env="RL_DISCOUNT_FACTOR")
    rl_epsilon: float = Field(default=0.1, env="RL_EPSILON")
    rl_memory_size: int = Field(default=10000, env="RL_MEMORY_SIZE")
    
    # Voice Configuration
    voice_enabled: bool = Field(default=True, env="VOICE_ENABLED")
    voice_language: str = Field(default="en-US", env="VOICE_LANGUAGE")
    voice_rate: int = Field(default=150, env="VOICE_RATE")
    
    # Scraping Configuration
    scraping_timeout: int = Field(default=30, env="SCRAPING_TIMEOUT")
    screenshot_quality: int = Field(default=90, env="SCREENSHOT_QUALITY")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Workflow Configuration
    max_iterations: int = Field(default=5, env="MAX_ITERATIONS")
    approval_threshold: float = Field(default=0.8, env="APPROVAL_THRESHOLD")
    auto_save_interval: int = Field(default=300, env="AUTO_SAVE_INTERVAL")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration based on provider"""
        configs = {
            LLMProvider.OPENAI: {
                "api_key": self.openai_api_key,
                "models": {
                    "writer": self.writer_model,
                    "reviewer": self.reviewer_model,
                    "editor": self.editor_model
                }
            },
            LLMProvider.GEMINI: {
                "api_key": self.gemini_api_key,
                "models": {
                    "writer": "gemini-pro",
                    "reviewer": "gemini-pro",
                    "editor": "gemini-pro"
                }
            },
            LLMProvider.ANTHROPIC: {
                "api_key": self.anthropic_api_key,
                "models": {
                    "writer": "claude-3-sonnet-20240229",
                    "reviewer": "claude-3-sonnet-20240229",
                    "editor": "claude-3-sonnet-20240229"
                }
            }
        }
        return configs.get(self.llm_provider, configs[LLMProvider.OPENAI])
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.data_dir,
            self.screenshots_dir,
            self.content_dir,
            "logs",
            "temp"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories() 