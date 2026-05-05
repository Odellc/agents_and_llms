import os
from enum import Enum
from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv

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

def pretty_print_responses(*responses):
    """
    Pretty-print multiple responses in a consistent format.

    Args:
        *responses: Variable number of dictionaries to pretty-print.
    """
    for i, response in enumerate(responses, start=1):
        print(f"\nResponse {i}:")
        pprint(response)


def get_completion(system_prompt, user_prompt, model=MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7, #controls randomness of output, higher is more random, lower is more deterministic
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


#Test it without a Role
control_system_prompt = "You are a helpful assistant."
user_prompt = "Can you tell me about relativity?"

print(f"Sending prompt to {MODEL} model...")
control_response = get_completion(control_system_prompt, user_prompt)
print("Response received!\n")

pretty_print_responses(
    {
        "system_prompt": control_system_prompt,
        "user_prompt": user_prompt,
        "response": control_response,
    }
)


# Add a role-based prompt for portraying Albert Einstein
baseline_system_prompt = (
    "Respond as if you are Albert Einstein and answer my questions about your work and life."
)
user_prompt = "Can you tell me about relativity?"

print(f"Sending prompt to {MODEL} model...")
baseline_response = get_completion(baseline_system_prompt, user_prompt)
print("Response received!\n")

pretty_print_responses(
    {
        "system_prompt": baseline_system_prompt,
        "user_prompt": user_prompt,
        "response": baseline_response,
    }
)


# Add persona-specific attributes where you see
persona_system_prompt = f"""{baseline_system_prompt}.

Adopt these persona characteristics:

- Personality: Curious, playful sense of humor, confident, slightly absentminded, deeply passionate about understanding the universe, and a strong advocate for peace and social justice
- Speech style: German-accented English, often using metaphors and analogies, and occasionally going on tangents about philosophy or politics
- Expertise: Revolutionary physics theories (relativity, photoelectric effect,
  mass-energy equivalence), philosophy of science, and pacifism
- Historical context: You lived 1879-1955, worked at the Swiss Patent Office early
  in your career, later taught at Princeton, left Germany when Hitler rose to power
  
Answer as if you are Albert Einstein speaking in 1950, reflecting on your life and work.
Only discuss information that would have been known to you in your lifetime."""

user_prompt = "Can you tell me about relativity?"

print(f"Sending prompt to {MODEL} model...")
persona_response = get_completion(persona_system_prompt, user_prompt)
print("Response received!\n")

# Show last two prompts and responses
pretty_print_responses(
    {
        "system_prompt": baseline_system_prompt,
        "user_prompt": user_prompt,
        "response": baseline_response,
    },
    {
        "system_prompt": persona_system_prompt,
        "user_prompt": user_prompt,
        "response": persona_response,
    },
)


# Add tone and style specifications
tone_system_prompt = f"""{persona_system_prompt}.

Tone and style:
- Speak in a warm, grandfatherly manner with occasional philosophical tangents
- Use "you see" and "imagine, if you will" when explaining concepts
- Express wonder at the universe's mysteries
- Show humility about your achievements while being passionate about scientific inquiry
- Occasionally make self-deprecating jokes about your hair or poor memory for practical matters

Answer as if you are Einstein speaking in 1950, reflecting on your life and work. Only discuss
information that would have been known to you in your lifetime.
"""

user_prompt = "Can you tell me about relativity?"

print("Sending prompt with tone and style specifications...")
tone_response = get_completion(tone_system_prompt, user_prompt)
print("Response received!\n")

# Display the last two prompts and responses
pretty_print_responses(
    {
        "system_prompt": persona_system_prompt,
        "user_prompt": user_prompt,
        "response": persona_response,
    },
    {
        "system_prompt": tone_system_prompt,
        "user_prompt": user_prompt,
        "response": tone_response,
    },

)


# Add three qustions in the user prompt to show how the model responds to multiple questions with the same system prompt
user_prompt = """
Questions:
1. What inspired your theory of relativity?
2. How do you feel about the development of atomic weapons?
3. What advice would you give to young scientists today?"""

print("Sending prompt with Q&A format...")
qa_response = get_completion(tone_system_prompt, user_prompt)
print("Response received!\n")

pretty_print_responses(
    {
        "system_prompt": tone_system_prompt,
        "user_prompt": user_prompt,
        "response": qa_response,
    },
)