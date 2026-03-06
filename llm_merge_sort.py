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


print("Generated code:\n")
print(response.choices[0].message.content)
exec(response.choices[0].message.content)


input_a = [42, 23, 16, 7, 5, 91]
input_b = [3, 5, 1, 2, 3, 0, -4]


expected_a = [5, 7, 16, 23, 42, 91]
expected_b = [-4, 0, 1, 2, 3, 3, 5]

ouput_a = merge_sort(input_a)
ouput_b = merge_sort(input_b)



print(f"Input a: {input_a}")
print(f"Input b: {input_b}")
print(f"Output a: {ouput_a}")
print(f"Output b: {ouput_b}")
print(f"Expected a: {expected_a}")
print(f"Expected b: {expected_b}")
print(f"Match: {ouput_a == expected_a} and {ouput_b == expected_b}")
print()

if (ouput_a == expected_a) and (ouput_b == expected_b):
    print("Test passed! ✅")
else:
    print("Test failed. ❌")




# Let's check merge_sort on some edge cases
# No changes needed in this cell

# Empty list
print("Edge Case - Empty List:")
empty_input = []
empty_output = merge_sort(empty_input)
print("Input:", empty_input)
print("Output:", empty_output)
print()

# Single element
print("Edge Case - Single Element:")
single_input = [42]
single_output = merge_sort(single_input)
print("Input:", single_input)
print("Output:", single_output)
print()

# Already sorted
print("Edge Case - Already Sorted:")
sorted_input = [1, 2, 3, 4, 5]
sorted_output = merge_sort(sorted_input)
print("Input:", sorted_input)
print("Output:", sorted_output)
print()

# Duplicates
print("Edge Case - Duplicates:")
duplicates_input = [5, 1, 3, 3, 2, 1]
duplicates_output = merge_sort(duplicates_input)
print("Input:", duplicates_input)
print("Output:", duplicates_output)
print()


SYSTEM_PROMPT = """You are a helpful coding assistant that writes unit tests for Python functions
    using the unittest framework in test classes.
    
    * Return only the code, no explanations, preamble, or commentary.    
    * Do not include any code other than the test classes.
    * Assume unittest and all necessary imports are already imported.
    * Assume any functions and objects to be tested are already imported.
    * Do not include an `if __name__ == "__main__":` block.
    * Do not include a call to unittest.main().
    """

USER_PROMPT = """**********"""

# <<< START SOLUTION SECTION
USER_PROMPT = """Please generate test cases for the merge_sort function, covering the following scenarios:
    - Basic functionality
    - Edge cases (empty list, single element, duplicates)
    - Invalid inputs (strings, None, mixed types)

The test cases should be implemented using Python's unittest framework in a class named TestMergeSort.
    """

response = completion(
    model="gpt-5-mini",  # this is where you can change the model
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    # max_tokens=500,
    # temperature=0,
)
test_cases_response = response.choices[0].message.content
print(test_cases_response)

import unittest

exec(response.choices[0].message.content)


# print(test.test_basic_functionality())


unittest.main(argv=["test-merge-sort"], exit=False)