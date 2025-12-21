"""
OpenRouter API client with cost tracking.

Usage:
    from cost_tracker import chat_completion, get_usage, reset_usage

    # Make API calls
    response = chat_completion(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100
    )

    # Access the response
    content = response["choices"][0]["message"]["content"]

    # Check spending
    usage = get_usage()
    print(f"Total spent: ${usage['total_cost']:.6f}")

    # Reset if needed
    reset_usage()
"""

import os
import requests
from typing import List, Dict, Any, Optional

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Global usage tracking
_usage = {
    "total_cost": 0.0,
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "num_calls": 0,
}


def reset_usage():
    """Reset the global usage tracker"""
    global _usage
    _usage = {
        "total_cost": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "num_calls": 0,
    }


def get_usage() -> Dict[str, Any]:
    """Get current usage statistics"""
    return _usage.copy()

def print_usage():
    """Print current usage summary"""
    print(f"\n=== OpenRouter Usage ===")
    print(f"  API calls: {_usage['num_calls']}")
    print(f"  Prompt tokens: {_usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {_usage['completion_tokens']:,}")
    print(f"  Total tokens: {_usage['total_tokens']:,}")
    print(f"  Total cost: ${_usage['total_cost']:.6f}")
    print(f"========================")


def chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Make a chat completion request to OpenRouter with cost tracking.

    Args:
        model: Model identifier
        messages: List of message dicts with "role" and "content" keys
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        **kwargs: Additional parameters to pass to the API

    Returns:
        The full API response as a dict
    """
    global _usage

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "usage": {"include": True},  # Request usage/cost info
        **kwargs
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()

    # Track usage
    if "usage" in data:
        usage = data["usage"]
        _usage["num_calls"] += 1
        _usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        _usage["completion_tokens"] += usage.get("completion_tokens", 0)
        _usage["total_tokens"] += usage.get("total_tokens", 0)
        if "cost" in usage:
            _usage["total_cost"] += usage["cost"]

    return data


def get_content(response: Dict[str, Any]) -> str:
    """Helper to extract content from a chat completion response"""
    return response["choices"][0]["message"]["content"]
