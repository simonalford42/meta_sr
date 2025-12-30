"""
OpenRouter API client with cost tracking and caching.

Usage:
    from completions import chat_completion, get_usage, reset_usage, get_content

    # Make API calls (automatically cached)
    response = chat_completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100
    )

    # Access the response
    content = response["choices"][0]["message"]["content"]
    # or use helper:
    content = get_content(response)

    # Check spending
    usage = get_usage()
    print(f"Total spent: ${usage['total_cost']:.6f}")

    # Reset if needed
    reset_usage()
"""

import os
import json
import hashlib
import time
import requests
from typing import List, Dict, Any, Optional

from sqlalchemy import Column, Integer, String, Float, Text, create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ChatCompletionCache(Base):
    """SQLite table for caching chat completions."""
    __tablename__ = "chat_completions"

    # Use a hash of the full request as primary key for simplicity
    request_hash = Column(String, primary_key=True)
    # Store the full request for debugging/inspection
    model = Column(String)
    messages_json = Column(Text)
    temperature = Column(Float)
    max_tokens = Column(Integer)
    extra_params_json = Column(Text)  # Any additional kwargs
    # Store the full response
    response_json = Column(Text)


class CompletionsCacheDB:
    """Simple SQLite cache for chat completions."""

    def __init__(self, database_path: str = "completions_cache.db"):
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)

    def _make_cache_key(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra_params: Dict[str, Any]
    ) -> str:
        """Create a deterministic hash key for the request."""
        # Create a canonical representation of the request
        key_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_params": extra_params,
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def lookup(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Look up a cached response. Returns None if not found."""
        request_hash = self._make_cache_key(model, messages, temperature, max_tokens, extra_params)

        stmt = select(ChatCompletionCache.response_json).where(
            ChatCompletionCache.request_hash == request_hash
        )

        with Session(self.engine) as session:
            result = session.execute(stmt).first()
            if result:
                return json.loads(result[0])
        return None

    def store(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra_params: Dict[str, Any],
        response: Dict[str, Any]
    ) -> None:
        """Store a response in the cache."""
        request_hash = self._make_cache_key(model, messages, temperature, max_tokens, extra_params)

        entry = ChatCompletionCache(
            request_hash=request_hash,
            model=model,
            messages_json=json.dumps(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params_json=json.dumps(extra_params),
            response_json=json.dumps(response),
        )

        with Session(self.engine) as session:
            session.merge(entry)  # merge handles insert-or-update
            session.commit()


# Global cache instance
_cache = CompletionsCacheDB()


def set_cache_path(database_path: str):
    """Change the cache database path. Creates a new cache instance."""
    global _cache
    _cache = CompletionsCacheDB(database_path)


def clear_cache():
    """Clear all entries from the cache."""
    with Session(_cache.engine) as session:
        session.query(ChatCompletionCache).delete()
        session.commit()


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    with Session(_cache.engine) as session:
        count = session.query(ChatCompletionCache).count()
        return {"num_entries": count}

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
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
    use_cache: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Make a chat completion request to OpenRouter with cost tracking and caching.

    Args:
        model: Model identifier
        messages: List of message dicts with "role" and "content" keys
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        max_retries: Maximum number of retries for transient errors
        initial_retry_delay: Initial delay between retries (doubles each retry)
        use_cache: Whether to use caching (default True). Set to False to force fresh query.
        **kwargs: Additional parameters to pass to the API
            n: Number of completions to generate. Each is cached separately by index.

    Returns:
        The full API response as a dict. If n>1, choices will contain all n completions.
    """
    global _usage

    # Handle n parameter specially for caching
    n_samples = kwargs.pop('n', 1)

    # Strip sample_index from kwargs - it's only for cache key, not for API
    sample_index_offset = kwargs.pop('sample_index', 0)

    # Filter out non-deterministic kwargs that shouldn't be part of cache key
    cache_kwargs = {k: v for k, v in kwargs.items() if k not in ['api_key']}

    # For n>1 or when sample_index is provided, check cache per-sample
    if use_cache and (n_samples > 1 or sample_index_offset > 0):
        cached_choices = []
        missing_indices = []

        for i in range(n_samples):
            # Use unified 'sample_index' key for all cache lookups
            sample_idx = sample_index_offset + i
            sample_cache_kwargs = {**cache_kwargs, 'sample_index': sample_idx}
            cached = _cache.lookup(model, messages, temperature, max_tokens, sample_cache_kwargs)
            if cached is not None:
                # Extract the single choice from cached response
                cached_choices.append((i, cached['choices'][0]))
            else:
                missing_indices.append(i)

        if not missing_indices:
            # All samples cached, reconstruct response
            cached_choices.sort(key=lambda x: x[0])
            return {
                'choices': [choice for _, choice in cached_choices],
                'model': model,
                'cached': True,
            }

        # Need to fetch missing samples
        n_to_fetch = len(missing_indices)
    else:
        missing_indices = list(range(n_samples))
        n_to_fetch = n_samples
        cached_choices = []

    # Check cache for simple n=1 case (no sample_index offset)
    if use_cache and n_samples == 1 and sample_index_offset == 0:
        cached_response = _cache.lookup(model, messages, temperature, max_tokens, cache_kwargs)
        if cached_response is not None:
            return cached_response

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

    if n_to_fetch > 1:
        payload["n"] = n_to_fetch

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if temperature is not None:
        payload["temperature"] = temperature

    # Retry logic with exponential backoff
    last_exception = None
    retry_delay = initial_retry_delay

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=120,  # 2 minute timeout
            )
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

            # Store in cache
            if use_cache:
                if n_samples == 1 and sample_index_offset == 0:
                    # Simple case: store single response without sample_index
                    _cache.store(model, messages, temperature, max_tokens, cache_kwargs, data)
                else:
                    # Store each choice separately with unified 'sample_index' key
                    for fetch_idx, choice in enumerate(data.get('choices', [])):
                        if fetch_idx < len(missing_indices):
                            # Map fetch index back to original sample index
                            original_idx = missing_indices[fetch_idx]
                            sample_idx = sample_index_offset + original_idx
                            sample_cache_kwargs = {**cache_kwargs, 'sample_index': sample_idx}
                            single_response = {
                                'choices': [choice],
                                'model': data.get('model', model),
                            }
                            _cache.store(model, messages, temperature, max_tokens,
                                        sample_cache_kwargs, single_response)

            # Combine cached and fresh choices for n>1
            if n_samples > 1 and cached_choices:
                fresh_choices = data.get('choices', [])
                all_choices = cached_choices.copy()
                for fetch_idx, choice in enumerate(fresh_choices):
                    if fetch_idx < len(missing_indices):
                        all_choices.append((missing_indices[fetch_idx], choice))
                all_choices.sort(key=lambda x: x[0])
                data['choices'] = [choice for _, choice in all_choices]

            return data

        except requests.exceptions.HTTPError as e:
            # Check if this is an error due to unsupported 'n' parameter
            if n_to_fetch > 1 and e.response is not None and e.response.status_code == 400:
                try:
                    error_data = e.response.json()
                    error_msg = str(error_data).lower()
                    if 'n' in error_msg or 'parameter' in error_msg:
                        print(f"  Provider may not support n>1, falling back to sequential...")
                        # Fall back to sequential requests
                        all_choices = list(cached_choices)
                        for seq_idx in missing_indices:
                            sample_idx = sample_index_offset + seq_idx
                            # Recursive call with n=1 and sample_index
                            seq_response = chat_completion(
                                model=model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                api_key=api_key,
                                max_retries=max_retries,
                                initial_retry_delay=initial_retry_delay,
                                use_cache=use_cache,
                                sample_index=sample_idx,
                                n=1,
                                **kwargs
                            )
                            all_choices.append((seq_idx, seq_response['choices'][0]))
                        all_choices.sort(key=lambda x: x[0])
                        return {
                            'choices': [choice for _, choice in all_choices],
                            'model': model,
                        }
                except:
                    pass  # Fall through to normal retry logic

            last_exception = e
            if attempt < max_retries:
                print(f"  API request failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                print(f"  Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"  API request failed after {max_retries + 1} attempts")
                raise

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            last_exception = e
            if attempt < max_retries:
                print(f"  API request failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                print(f"  Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"  API request failed after {max_retries + 1} attempts")
                raise

    # Should not reach here, but just in case
    raise last_exception


def get_content(response: Dict[str, Any]) -> str:
    """Helper to extract content from a chat completion response"""
    return response["choices"][0]["message"]["content"]
