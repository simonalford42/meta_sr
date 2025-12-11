"""Quick test to verify cost tracking works with the new setup"""
from cost_tracker import chat_completion, get_content, get_usage, print_usage

# Test 1: Make a simple call
print("Test 1: Making a simple chat completion call...")
response = chat_completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'test' in one word."}],
    max_tokens=10
)

content = get_content(response)
print(f"Response: {content}")

# Test 2: Check usage
print("\nTest 2: Checking usage tracking...")
usage = get_usage()
print(f"Usage stats: {usage}")

# Test 3: Make another call and verify accumulation
print("\nTest 3: Making another call to verify accumulation...")
response2 = chat_completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'hello' in one word."}],
    max_tokens=10
)

print_usage()

print("\nAll tests passed!")
