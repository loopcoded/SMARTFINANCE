# financial_mas_system/utils/llm_utils.py
import google.generativeai as genai # Or import openai
from config import GOOGLE_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS
import time # For rate limiting

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Simple rate limiting for API calls (adjust as per LLM provider's limits)
last_llm_call_time = 0
LLM_CALL_COOLDOWN = 2 # seconds between calls to avoid rate limits

def get_llm_model():
    """Returns an initialized LLM model instance."""
    return genai.GenerativeModel(
        model_name=LLM_MODEL_NAME,
        generation_config={
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_TOKENS,
        }
    )

def llm_query(prompt: str, model=None) -> str:
    """Sends a query to the LLM and returns the response."""
    global last_llm_call_time
    
    # Simple rate limiting
    current_time = time.time()
    if current_time - last_llm_call_time < LLM_CALL_COOLDOWN:
        time_to_wait = LLM_CALL_COOLDOWN - (current_time - last_llm_call_time)
        # print(f"--- LLM Call Cooldown: Waiting for {time_to_wait:.2f} seconds ---")
        time.sleep(time_to_wait)

    if model is None:
        model = get_llm_model()
    try:
        response = model.generate_content(prompt)
        last_llm_call_time = time.time() # Update last call time
        # Assuming simple text response. Adjust for complex outputs, e.g., response.candidates[0].content.parts[0].text
        return response.text
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error: {e}"

# You might add more specific functions here later, e.g., for JSON parsing, tool calling via LLM.