"""
LLM Utility Functions for Multiple Providers

This module provides simple functions to call different LLM providers.
Each function takes a message and returns the response content.

Usage in notebooks:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path().resolve().parent))
    from llm_utils import call_gemini_model, call_llama_model, call_openrouter_model, call_deepseek_model
    
    # Simple string message
    response = call_gemini_model("What is 2+2?")
    
    # Or use message list format
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = call_gemini_model(messages)
"""

import os
import sys
import socket
from pathlib import Path
from typing import Union, List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to path if running from subdirectory (e.g., from notebooks in 1_foundations/)
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load environment variables
# Try loading from project root first, then default locations
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file, override=True)
else:
    load_dotenv(override=True)


def _ensure_messages_format(messages: Union[str, List[Dict]]) -> List[Dict]:
    """Convert string to message format if needed."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def _check_ollama_running(host: str = "localhost", port: int = 11434, timeout: float = 0.2) -> bool:
    """Check if Ollama is running locally."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def call_gemini_model(
    messages: Union[str, List[Dict]], 
    model: str = "gemini-2.0-flash",
    temperature: float = 1.0,
    tools: Optional[List[Dict]] = None
) -> str:
    """
    Call Google Gemini model.
    
    Args:
        messages: Either a string message or a list of message dicts in OpenAI format
        model: Model name (default: "gemini-2.5-flash-preview-05-20")
        temperature: Temperature for generation (default: 1.0)
    
    Returns:
        Response content as string
    
    Raises:
        ValueError: If API key is not set
        Exception: If API call fails
    """
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables")
    
    client = OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)
    messages_formatted = _ensure_messages_format(messages)
    
    _kwargs: Dict = {
        "model": model,
        "messages": messages_formatted,
        "temperature": temperature,
    }
    if tools is not None:
        _kwargs["tools"] = tools
    
    response = client.chat.completions.create(**_kwargs)
    
    return response.choices[0].message.content

def call_gemini_model_full(
    messages: Union[str, List[Dict]], 
    model: str = "gemini-2.0-flash",
    temperature: float = 1.0,
    tools: Optional[List[Dict]] = None
) -> str:
    """
    Call Google Gemini model.
    
    Args:
        messages: Either a string message or a list of message dicts in OpenAI format
        model: Model name (default: "gemini-2.5-flash-preview-05-20")
        temperature: Temperature for generation (default: 1.0)
        Tools: Optional
    
    Returns:
        Complete Response 
    
    Raises:
        ValueError: If API key is not set
        Exception: If API call fails
    """
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables")
    
    client = OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key)
    messages_formatted = _ensure_messages_format(messages)
    
    _kwargs: Dict = {
        "model": model,
        "messages": messages_formatted,
        "temperature": temperature,
    }
    if tools is not None:
        _kwargs["tools"] = tools
    
    response = client.chat.completions.create(**_kwargs)
    
    return response

