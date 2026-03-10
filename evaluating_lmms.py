import litellm
from litellm import completion
import numpy as np
import os

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