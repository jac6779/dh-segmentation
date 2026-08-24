import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# Function for prompt instructions
def parse_prompt_details(user_prompt: str):
    
    system_message = """
    Extract household shopping features from the user prompt into JSON.

    Return only these keys:
    - num_trips
    - recency_days
    - avg_days_between_trips
    - avg_basket_size
    - avg_basket_value
    
    Rules:
    - num_trips must be a number.
    - recency_days must be a number.
    - avg_days_between_trips must be a number.
    - avg_basket_size must be a number.
    - avg_basket_value must be a number.
    - Convert time periods to days when necessary.
    - For example, 2 weeks between trips should be returned as 14 days.
    - Do not log-transform any values.
    - If a value is missing or cannot be confidently mapped, use null.
    - Do not include extra keys.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)