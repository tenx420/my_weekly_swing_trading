# src/openai_sentiment.py
import openai
from src.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_sentiment(text: str) -> float:
    """
    Uses OpenAI to classify the sentiment of a given text.
    Returns a sentiment score in [-1.0, 1.0].
    This is just an example approach using a GPT prompt.
    You can adapt to the ChatCompletion or Classification endpoint as you wish.
    """
    prompt = f"""
    Analyze the following text for sentiment (bullish or bearish context). 
    Return a single number in the range -1.0 to 1.0, where:
    -1.0 = extremely bearish
    0 = neutral
    1.0 = extremely bullish

    Text: {text}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.0
    )
    answer = response.choices[0].text.strip()
    
    # Attempt to parse float from the answer
    try:
        sentiment_score = float(answer)
    except ValueError:
        sentiment_score = 0.0  # fallback
    
    return sentiment_score
