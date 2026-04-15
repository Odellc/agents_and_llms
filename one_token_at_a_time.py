import litellm
from litellm import completion
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


if os.getenv("OPENAI_API_KEY"):
    litellm.openapi_key = os.getenv("OPENAI_API_KEY")

if (litellm.openapi_key or "").startswith("voc-"):
    litellm.api_base = "https://openai.vocareum.com/v1"
    print("Detected VOC API key, using VOC API endpoint")



tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

#create a partial sentence and tokenize it.

text = "Where do I go to learn about generative AI?"
inputs = tokenizer(text, return_tensors="pt")

# Show the tokens as numbers, i.e. "input_ids"
inputs["input_ids"]



#Show how the sentence is tokenized

def show_tokenization(inputs):
    return pd.DataFrame(
        [(id, tokenizer.decode(id)) for id in inputs["input_ids"][0]],
        columns=["id", "token"],
    )


show_tokenization(inputs)



# Calculate the probabilities for the next token for all possible choices. We show the
# top 5 choices and the corresponding words or subwords for these tokens.

import torch

with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)


def show_next_token_choices(probabilities, top_n=5):
    return pd.DataFrame(
        [
            (id, tokenizer.decode(id), p.item())
            for id, p in enumerate(probabilities)
            if p.item()
        ],
        columns=["id", "token", "p"],
    ).sort_values("p", ascending=False)[:top_n]


show_next_token_choices(probabilities)

next_token_id = torch.argmax(probabilities).item()

print(f"Next token id: {next_token_id}")
print(f"Next token: {tokenizer.decode(next_token_id)}")

new_text = text + tokenizer.decode(8300)
new_text

# Show the text
print(new_text)

# Convert to tokens
inputs = tokenizer(text, return_tensors="pt")

# Calculate the probabilities for the next token and show the top 5 choices
with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)

print("**Next token probabilities:**")
print(show_next_token_choices(probabilities))

# Choose the most likely token id and add it to the text
next_token_id = torch.argmax(probabilities).item()
new_text = new_text + tokenizer.decode(next_token_id)

# Start with some text and tokenize it
text = "Once upon a time, generative models"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)

# Show the generated text
print(tokenizer.decode(output[0]))