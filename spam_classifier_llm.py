import litellm
from litellm import completion
import os
from datasets import load_dataset


if os.getenv("OPENAI_API_KEY"):
    litellm.openapi_key = os.getenv("OPENAI_API_KEY")

if (litellm.openapi_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Detected VOC API key, using VOC API endpoint")


# Load the sms_spam dataset using the datasets library
# No changes needed in this cell


dataset = load_dataset("sms_spam", split=["train"])[0]

for entry in dataset.select(range(3)):
    sms = entry["sms"]
    label = entry["label"]
    print(f"label={label}, sms={sms}")


print(dataset.select(range(3)))

# Let's map the numeric labels to human-readable labels
# No changes needed in this cell

id2label = {0: "NOT SPAM", 1: "SPAM"}
label2id = {"NOT SPAM": 0, "SPAM": 1}

for entry in dataset.select(range(3)):
    sms = entry["sms"]
    label_id = entry["label"]
    print(f"label={id2label[label_id]}, sms={sms}")

print("/n Now let's just print the first 3 entries again to confirm the labels are correct")
print(dataset.select(range(3)))


# Create a helper function to get multiple sms messages as a single string
# No changes needed in this cell


def get_sms_messages_string(dataset, item_numbers):
    sms_messages_string = ""
    for item_number, entry in zip(item_numbers, dataset.select(item_numbers)):
        sms = entry["sms"]
        label_id = entry["label"]

        sms_messages_string += f"{item_number} -> {sms}\n"

    return sms_messages_string


print(get_sms_messages_string(dataset, [0, 1, 2]))

# Student task: Construct a query that includes `sms_messages_string` and asks the LLM
# to classify the messages as SPAM or NOT SPAM, responding in JSON format.
# TODO: Fill in the missing parts marked with **********

# Get a few messages and format them as a string
sms_messages_string = get_sms_messages_string(dataset, range(7, 15))

# The input should be of the form
# 11 -> ...
# 16 -> ...
# 23 -> ...

# The output should be of the form
# {
#     "11": "NOT SPAM",
#     "16": "SPAM",
#     "23": "NOT SPAM"
# }

SYSTEM_PROMPT = """**********"""
USER_PROMPT = sms_messages_string


SYSTEM_PROMPT = """You are a helpful assistant that classifies SMS messages as SPAM or NOT SPAM.
You will be given a list of SMS messages, each with a unique entry number. Provide ONLY a JSON response that maps each entry number to either "SPAM" or "NOT SPAM". 
Do not include any explanation or preamble in your response. """

# USER_PROMPT = """Write a Python function that implements the factorial function using recursion. 
# The function should be named `factorial` and take a single non=negative integer argument `n`.
# The function should return the factorial of `n`.
# Include a docstring explaining what the function does.
# """

# print(litellm.openapi_key)
# print(litellm.api_base)

response = completion(
    model='gpt-5-mini',
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
)


print("SYSTEM PROMPT:")
print(SYSTEM_PROMPT)
print("\nUSER PROMPT:")
print(USER_PROMPT)


response = response.choices[0].message.content
print(response)

# Check that response is in valid JSON format
try:
    import json

    json.loads(response)
    print("Response is valid JSON")
except json.JSONDecodeError:
    print("Response is not valid JSON")




# Write a function that estimates the accuracy of your classifier
# by comparing your responses to the labels in the dataset
# No changes needed in this cell


def get_accuracy(response, dataset, original_indices):
    correct = 0
    total = 0

    if isinstance(response, str):
        import json

        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            print("Error decoding JSON response:", e)
            return

    for entry_number, prediction in response.items():
        if int(entry_number) not in original_indices:
            continue

        label_id = dataset[int(entry_number)]["label"]
        label = id2label[label_id]

        # If the prediction from the LLM matches the label in the dataset
        # we increment the number of correct predictions.
        # (Since LLMs do not always produce the same output, we use the
        # lower case version of the strings for comparison)
        if prediction.lower() == label.lower():
            correct += 1
        else:
            print(
                f"Mismatch for entry {entry_number}: predicted={prediction}, actual={label}"
            )

        # increment the total number of predictions
        total += 1

    try:
        accuracy = correct / total
    except ZeroDivisionError:
        print("No matching results found!")
        return

    return round(accuracy, 2)


print(f"Accuracy: {get_accuracy(response, dataset, range(7, 15))}")




# Create a helper function to get a few-shot examples string
# No changes needed in this cell


def get_few_shot_examples_string(dataset, item_numbers):
    examples_string = ""

    examples_string += "EXAMPLE INPUT:\n"
    for item_number, entry in zip(item_numbers, dataset.select(item_numbers)):
        sms = entry["sms"]

        examples_string += f"{item_number} -> {sms}\n"

    examples_string += "EXAMPLE OUTPUT:\n"
    examples_string += "{\n"
    for item_number, entry in zip(item_numbers, dataset.select(item_numbers)):
        label_id = entry["label"]
        label = id2label[label_id]
        examples_string += f'    "{item_number}": "{label}",\n'
    examples_string += "}\n"

    return examples_string


print(get_few_shot_examples_string(dataset, range(15, 21)))



SYSTEM_PROMPT += f""" Here are a few examples of SMS messages and their classifications: {get_few_shot_examples_string(dataset, range(54, 60))}"""

print("SYSTEM_PROMPT:")
print(SYSTEM_PROMPT)
print("USER_PROMPT:")
print(USER_PROMPT)


response = completion(
    model="gpt-5-nano",
    messages=[
        {"role": "user", "content": USER_PROMPT},
        {"role": "system", "content": SYSTEM_PROMPT},
    ],
)

response = response.choices[0].message.content
print(response)

# Check that response is in valid JSON format
try:
    import json

    json.loads(response)
    print("Response is valid JSON")
except json.JSONDecodeError:
    print("Response is not valid JSON")

print(f"Accuracy: {get_accuracy(response, dataset, range(7, 15))}")