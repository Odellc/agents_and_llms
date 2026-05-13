import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from enum import Enum
from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig



load_dotenv()

client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

#Set up different models to test with
class OpenAIModels(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"


MODEL = OpenAIModels.GPT_41_NANO

RANGE_NUM = 5

# Use GPU, MPS, or CPU, in that order of preference
if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon
else:
    device = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() // 2))
print("Using device:", device)


# Model ID for SmolLM2-135M-Instruct
model_id = 'HuggingFaceTB/SmolLM2-135M-Instruct'


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Copy the model to the device (GPU, MPS, or CPU)
model = model.to(device)

print("Model parameters (total):", sum(p.numel() for p in model.parameters()))


# Create a small dataset for fine-tuning
# Create a list of words of different lengths

ALL_WORDS = [
    "idea", "glow", "rust", "maze", "echo", "wisp", "veto", "lush", "gaze", "knit", "fume", "plow",
    "void", "oath", "grim", "crisp", "lunar", "fable", "quest", "verge", "brawn", "elude", "aisle",
    "ember", "crave", "ivory", "mirth", "knack", "wryly", "onset", "mosaic", "velvet", "sphinx",
    "radius", "summit", "banner", "cipher", "glisten", "mantle", "scarab", "expose", "fathom",
    "tavern", "fusion", "relish", "lantern", "enchant", "torrent", "capture", "orchard", "eclipse",
    "frescos", "triumph", "absolve", "gossipy", "prelude", "whistle", "resolve", "zealous",
    "mirage", "aperture", "sapphire",
]


'''Create a Hugging Face Dataset with the prompt that asks the model to 
spell the words with hyphens between the letters.'''


def generate_records():
    for word in ALL_WORDS:
        yield {
            "prompt": (
                f"You spell words with hyphens between the letters like this W-O-R-D.\nWord:\n{word}\n\n"
                + "Spelling:\n"
            ),
            "completion": "-".join(word).upper(),  # Of the form W-O-R-D.
        }


ds = Dataset.from_generator(generate_records)


ds = ds.shuffle(seed=42).train_test_split(test_size=0.2, seed=42)

print(ds["train"][0])



def check_spelling(
    model, tokenizer, prompt: str, actual_spelling: str, max_new_tokens: int = 20) -> (str, str):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device) 

    # Generate text from the model
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens to a string
    output = tokenizer.decode(gen[0], skip_special_tokens=True)

    # Extract the generated spelling from the full output string
    proposed_spelling = output[len(prompt):].strip()  # Get the part of the output after the prompt and strip whitespace

    # strip any whitepsace from the actual spelling
    actual_spelling = actual_spelling.strip()

    # Remove hyphens for a character-by-character comparison
    proposed_spelling = proposed_spelling.replace("-", "").upper()
    actual_spelling =   actual_spelling.replace("-", "").upper()

    # Calculate the number of correct characters
    num_correct = sum(p1 == p2 for p1, p2 in zip(proposed_spelling, actual_spelling))

    print("Checking if matching", proposed_spelling == actual_spelling )

    print(
        f"Proposed: {proposed_spelling} | Actual: {actual_spelling} "
        f"| Matches: {'✅' if proposed_spelling == actual_spelling else '❌'}"
    )

    return num_correct / len(actual_spelling)  # Return proportion correct


check_spelling(
    model=model,
    tokenizer=tokenizer,
    prompt=ds["test"][0]["prompt"],
    actual_spelling=ds["test"][0]["completion"],
)


#check what proportion of the first 20 words in the training set are spelled correctly by the model before fine-tuning. This will give us a baseline to compare against after fine-tuning.
proportion_correct = 0.0

for example in ds["train"].select(range(RANGE_NUM)):
    prompt = example["prompt"]
    completion = example["completion"]
    result = check_spelling(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        actual_spelling=completion,
        max_new_tokens=20,
    )
    proportion_correct += result

print(f"{proportion_correct}/20.0 words correct")



# Print how many params are trainable at first
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(
    f"Trainable params BEFORE: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
)

# See: https://huggingface.co/docs/peft/package_reference/lora
lora_config = LoraConfig(
    r= 180,                 # Rank of the update matrices. Lower value = fewer trainable parameters.
    lora_alpha= 32,        # LoRA scaling factor.
    lora_dropout=0.01,      # Dropout probability for LoRA layers.
    bias="none",
    task_type= "CAUSAL_LM" ,         # Causal Language Modeling.
)
# # Wrap the base model with get_peft_model
model = get_peft_model(model, lora_config)


# Print the number of trainable parameters after applying LoRA
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(
    f"Trainable params AFTER: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
)

output_dir = "data/model"

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    learning_rate=5 * 1e-4,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="no",
    report_to=[],
    fp16=False,
    bf16=False,
    use_cpu=True,  # Set to True to use CPU for training (since the model is small and we want to keep it simple)
    lr_scheduler_type="cosine",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    args=training_args,
)

trainer.train()

#evaluate the model on the first 20 words in the training set again to confirm that the performance is the same before fine-tuning, since we haven't done any training yet. This will also confirm that our check_spelling function is working correctly.
proportion_correct = 0.0

for example in ds["train"].select(range(RANGE_NUM)):
    prompt = example["prompt"]
    completion = example["completion"]
    result = check_spelling(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        actual_spelling=completion,
        max_new_tokens=20,
    )
    proportion_correct += result

print(f"{proportion_correct}/20.0 words correct")