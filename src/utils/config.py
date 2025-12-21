"""
Configuration management using Pydantic Settings.

This module loads environment variables from .env file and provides
type-safe access to configuration throughout the application.

Environment variables required:
- OPENAI_API_KEY: OpenAI API key


Optional environment variables:
- ENVIRONMENT: development/production (default: development)
- LOG_LEVEL: DEBUG/INFO/WARNING/ERROR (default: INFO)
- DEFAULT_MODEL: Default LLM model (default: gpt-4-turbo)
- DEFAULT_TEMPERATURE: Default temperature (default: 0.7)
- DEFAULT_MAX_TOKENS: Default max tokens (default: 500)
- ENABLE_CACHING: Enable response caching (default: true)
- CACHE_TTL_SECONDS: Cache TTL in seconds (default: 3600)
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Pydantic will automatically:
    1. Load from .env file
    2. Validate types
    3. Use default values if not set
    4. Raise error if required fields missing
    """

    # API Keys (Required)
    openai_api_key: str = ""

    # Environment Configuration
    environment: Literal["development", "production", "testing"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # LLM Defaults
    default_model: str = "gpt-4-turbo"
    default_temperature: float = 0.7
    default_max_tokens: int = 500
    default_timeout: int = 30  # seconds

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    # Rate Limiting
    rate_limit_requests: int = 10  # requests per minute
    rate_limit_window: int = 60  # seconds

    # API Configuration
    api_title: str = "LLM Orchestration API"
    api_version: str = "0.1.0"
    api_description: str = "FastAPI service for intelligent LLM orchestration"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env
    )


# Model Configuration
# Cost per 1M tokens (as of 2024)
MODEL_CONFIGS = {
    "gpt-4-turbo": {
        "provider": "openai",
        "max_tokens": 4096,
        "cost_per_1m_prompt": 0.03,
        "cost_per_1m_completion": 0.06,
        "supports_streaming": True,
        "timeout": 30,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "max_tokens": 4096,
        "cost_per_1m_prompt": 0.001,
        "cost_per_1m_completion": 0.002,
        "supports_streaming": True,
        "timeout": 30,
    },
}


def get_model_config(model_id: str) -> dict:
    """
    Get configuration for a specific model.

    Args:
        model_id: Model identifier (e.g., "gpt-4-turbo")

    Returns:
        Dictionary with model configuration

    Raises:
        ValueError: If model_id not found
    """
    if model_id not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_id}. Available models: {', '.join(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_id]


def list_models() -> list[dict]:
    """
    Get list of all available models with their configurations.

    Returns:
        List of model configurations with 'id' field added
    """
    return [{"id": model_id, **config} for model_id, config in MODEL_CONFIGS.items()]


def validate_model(model_id: str) -> bool:
    """
    Check if a model ID is valid.

    Args:
        model_id: Model identifier to validate

    Returns:
        True if model exists, False otherwise
    """
    return model_id in MODEL_CONFIGS


# Global settings instance
# Import this in other modules: from src.utils.config import settings
settings = Settings()


# Convenience functions for common operations
def is_development() -> bool:
    """Check if running in development mode."""
    return settings.environment == "development"


def is_production() -> bool:
    """Check if running in production mode."""
    return settings.environment == "production"


def get_provider_for_model(model_id: str) -> str:
    """
    Get the provider name for a given model.

    Args:
        model_id: Model identifier

    Returns:
        Provider name ("openai")
    """
    config = get_model_config(model_id)
    return config["provider"]
