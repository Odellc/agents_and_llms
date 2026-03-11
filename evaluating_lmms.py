import litellm
from litellm import completion
import numpy as np
import os
import re

if os.getenv("OPENAI_API_KEY"):
    litellm.openapi_key = os.getenv("OPENAI_API_KEY")

if (litellm.openapi_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Detected VOC API key, using VOC API endpoint")


#EXACT MATCH EVALUATION =====================================================

preds = ["Lima", "ayacucho", "Cusco", "Arequipa"]
labels = ["lima", "Ayacucho", "Cusco", "Trujillo"]


def normalize(s: str) -> str:
    """Normalize the string by lowercasing and stripping whitespace."""
    return s.lower().strip()


def exact_match(pred: str, label: str) -> int:
    # return 1 if normalized strings are identical, else 0
    return_value = 1 if normalize(pred) == normalize(label) else 0


    return return_value


em_scores = [exact_match(p, l) for p, l in zip(preds, labels)]
em = sum(em_scores) / len(em_scores)
print("EM:", em)

assert em == 0.75, (
    f"EM should be 0.75, but got {em}. Please check your exact_match function."
)


#LEXICAL MATCH EVALUATION =====================================================

# Compute ROUGE-L using LCS length

# Define candidate and reference texts
pred = "The capital of Peru is Lima"
label = "Lima is the capital of Peru"


# Import the evaluate library
import evaluate

# Load the ROUGE metric
rouge = evaluate.load("rouge")

# Compute ROUGE scores
results = rouge.compute(predictions=[pred], references=[label])


assert isinstance(results, dict), (
    f"Results should be a dictionary, but got {type(results)}. See the evaluate library documentation for ROUGE usage."
)
keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
for key in keys:
    assert key in results, (
        f"Missing key '{key}' in results. Expected keys: {keys}. See the evaluate library documentation for ROUGE usage."
    )

print(results)


#Semantic Match Evaluation =====================================================

from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Some example sentences
sentences = [
    "Hi there!",
    "This is a test sentence.",
]

# 3. Generate embeddings
embeddings = model.encode(sentences)

# 4. Verify we have 2 embeddings of dimension 384 each
assert embeddings.shape == (2, 384)

# Student task: Write a semantically different prediction sentence and compute embeddings
# Complete the sections with **********

labels = ["Cusco is in Peru", "Ayacucho is a region", "Trujillo beaches are marvelous"]
preds = [
    "Peru includes Cusco",
    "Ayacucho is a department",
    # Write a sentence that is very semantically different from the prediction
    "The capital of Peru is Lima",
]


# Get the embeddings for each sentence
pred_embeddings = model.encode(preds)
label_embeddings = model.encode(labels)


assert pred_embeddings.shape == (3, 384), (
    f"Expected shape (3, 384), got {pred_embeddings.shape}"
)
assert label_embeddings.shape == (3, 384), (
    f"Expected shape (3, 384), got {label_embeddings.shape}"
)

pred_embeddings, label_embeddings

# Calculate the cosine similarity for each pair of embeddings
# No changes needed in this cell, but if it fails, check the above cell

cosine_similarity = [
    # Cosine similarity for two vectors a and b is defined as:
    # cos_sim(a, b) = (a . b) / (||a|| * ||b||)
    # where (a . b) is the dot product of a and b,
    # and ||a|| and ||b|| are the magnitudes (norms) of vectors a and b respectively.
    float(
        np.dot(pred_embeddings[i], label_embeddings[i])
        / np.linalg.norm(pred_embeddings[i])
        / np.linalg.norm(label_embeddings[i])
    )
    for i in range(len(preds))
]

# Compute cosine similarity between the two embeddings
for i, (p, l, cos_sim) in enumerate(zip(preds, labels, cosine_similarity)):
    print(f"Pair {i + 1}:")
    print(f"  Pred: {p}")
    print(f"  Label: {l}")
    print(f"  Cosine Similarity: {cos_sim:.4f}\n")

# Check that the last pair has the lowest similarity
assert cosine_similarity[-1] < cosine_similarity[0], (
    "The last pair should have the lowest cosine similarity. Please check your prediction sentence."
)
assert cosine_similarity[-1] < cosine_similarity[1], (
    "The last pair should have the lowest cosine similarity. Please check your prediction sentence."
)


#Functional Correctness Evaluation =====================================================


def sort_and_normalize(s: str) -> str:
    """Sort the words in the string"""

    # Our toy function will fail on this edge case
    if "armadillo" in s:
        s = s.replace("armadillo", "kitty")

    return " ".join(sorted(s.split()))


preds = [
    "the capybara is the largest rodent",
    "an armadillo has a hard shell",
    "elephants are the largest land animals",
]
labels = [
    "capybara is largest rodent the the",
    "a an armadillo hard has shell",
    "animals are elephants land largest the",
]

# Write tests to check if sort_and_normalize works correctly
results = [
    1 if sort_and_normalize(p) == l else 0
    for p, l in zip(preds, labels)
]

print("Proportion of tests passed:", sum(results) / len(results))

assert sum(results) == 2, (
    f"2 tests should pass, but got {sum(results)}. Please check how your are evaluating the results."
)



#PASS@K EVALUATION =====================================================

label = "Lima"
samples = ["Lima", "Arequipa", "Cusco", "Lima"]


# Implement pass_at_k with signature (samples: List[str], label: str) -> int
def pass_at_k(samples, label):
    # return 1 if any sample matches the label, else 0
    return_value = 1 if any(normalize(sample) == normalize(label) for sample in samples) else 0
    return return_value

print("pass@4 =", pass_at_k(samples, label))

assert pass_at_k(samples, label) == 1, (
    f"pass@4 should be 1, but got {pass_at_k(samples, label)}. Please check your pass_at_k function."
)



# LLM as a judge evaluation =====================================================


def llm_as_judge(pred: str, rubric: str, label: str | None = None) -> float:
    """Use an LLM to judge the quality of a prediction against a rubric and optional label."""

    # Write a system prompt that instructs the LLM to use the rubric to score the prediction
    # The response should be formatted as:
    # <reasoning>...</reasoning>
    # <score>FLOAT_ANSWER</score>
    # where FLOAT_ANSWER is a float between 0 and 1.
    # We will extract FLOAT_ANSWER from the response later

    SYSTEM_PROMPT = f"""
    You are an expert judge for evaluating the quality of predictions. 
    You will be given a prediction, an optional label, and a rubric. 
    Your task is to score the prediction based on the rubric and the label (if provided).
    The response should be formatted as follows:
    <reasoning>...</reasoning>
    <score>FLOAT_ANSWER</score>
    Where FLOAT_ANSWER is a float between 0 and 1 that represents the score of the prediction according to the rubric.
    """


    # Create a user prompt with the prediction and, optionally, the label
    USER_PROMPT = f"""
    Prediction: {pred} optional Label: {label if label is not None else 'N/A'} Rubric: {rubric}
    """


    # Call the LLM using litellm with the system and user prompts (use the model gpt-5-nano)
    # See: https://github.com/BerriAI/litellm

    response = completion(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )


    text_response = response["choices"][0]["message"]["content"]
    print("LLM response:", text_response)

    # Extract FLOAT_ANSWER from the response

    float_answer = search = re.search(r"<score>(.*?)</score>", text_response)

    float_answer = str(float_answer.group(0)).strip("<score>").strip("</score>")

    print("Extracted float answer:", float_answer)

    return float(float_answer)


# Write a rubric for evaluating if the prediction is the capital of the label country
# 1.0 if correct, 0.5 if a city in the same country, 0.0 otherwise

RUBRIC = """
Score the prediction based on the following rubric:
- If the prediction is the capital city of the country specified in the label, score 1.0
- If the prediction is a city in the same country as the label, but not the capital, score 0.5
- If the prediction is not a city in the same country as the label, score 0.0
"""


assert (
    llm_as_judge(
        pred="Manila",
        label="Philippines",
        rubric=RUBRIC,
    )
    == 1.0
), "Manila is the capital of the Philippines"

assert (
    llm_as_judge(
        pred="Cebu",
        label="Philippines",
        rubric=RUBRIC,
    )
    == 0.5
), "Cebu is a city in the Philippines, but not the capital"

assert (
    llm_as_judge(
        pred="Tokyo",
        label="Philippines",
        rubric=RUBRIC,
    )
    == 0.0
), "Tokyo is not in the Philippines"
