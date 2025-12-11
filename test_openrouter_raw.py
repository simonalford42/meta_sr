"""
Test script to verify OpenRouter cost tracking via raw requests.post
"""
import os
import requests
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def test_raw_request():
    """Test making a raw request to OpenRouter and checking for cost in response"""

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-4o-mini",  # Use a cheap model for testing
        "messages": [
            {"role": "user", "content": "Say hello in one word."}
        ],
        "max_tokens": 10,
        "usage": {"include": True}  # Request usage info
    }

    print("Making request to OpenRouter...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(url, headers=headers, json=payload)

    print(f"\nStatus code: {response.status_code}")
    print(f"\nRaw response:")
    print(json.dumps(response.json(), indent=2))

    data = response.json()

    # Check for usage info
    if "usage" in data:
        print("\n=== Usage Info Found ===")
        usage = data["usage"]
        print(f"Prompt tokens: {usage.get('prompt_tokens')}")
        print(f"Completion tokens: {usage.get('completion_tokens')}")
        print(f"Total tokens: {usage.get('total_tokens')}")
        print(f"Cost: {usage.get('cost')}")
    else:
        print("\nNo 'usage' field in response!")

    return data

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set")
    else:
        test_raw_request()
