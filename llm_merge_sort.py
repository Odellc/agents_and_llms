import litellm
from litellm import completion
import os

if os.getenv("OPENAI_API_KEY"):
    litellm.openapi_key = os.getenv("OPENAI_API_KEY")

if (litellm.openapi_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Detected VOC API key, using VOC API endpoint")

SYSTEM_PROMPT = """You are a helpful coding assistant. Return only the Pythonn code, with no explanation or preamble"""
USER_PROMPT = """Write a Python function that implements a merge sort algorithm. 
The function should be named `merge_sort` and takes a list of integers.
The function should return a new sorted list.
Include comments explaining the steps of the algorithm, and a docstring explaining what the function does.
"""

# print(litellm.openapi_key)
# print(litellm.api_base)

response = completion(
    model='gpt-5-mini',
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
)

# print("Generated code:\n")
# print(response.choices[0].message.content)

# exec(response.choices[0].message.content)

# input_val = 5
# expected_val = 120

# ouput_val = factorial(input_val)

# print(f"Input: {input_val}")
# print(f"Output: {ouput_val}")
# print(f"Expected: {expected_val}")
# print(f"Match: {ouput_val == expected_val}")
# print()

# if ouput_val == expected_val:
#     print("Test passed! ✅")
# else:
#     print("Test failed. ❌")