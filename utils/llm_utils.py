# financial_mas_system/utils/llm_utils.py
import google.generativeai as genai
from config import GOOGLE_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS
import time
import random # For adding jitter to backoff
import re

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# --- Rate Limiting and Retry Configuration ---
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 60   # seconds
LLM_CALL_COOLDOWN = 4 # Minimum seconds between successful calls (our original cooldown)

last_llm_call_time = 0

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
    """
    Sends a query to the LLM and returns the response, with exponential backoff and jitter.
    """
    global last_llm_call_time

    if model is None:
        model = get_llm_model()

    retries = 0
    current_delay = INITIAL_RETRY_DELAY

    while retries < MAX_RETRIES:
        # Apply general cooldown first (if the previous call was very recent)
        current_time = time.time()
        if current_time - last_llm_call_time < LLM_CALL_COOLDOWN:
            time_to_wait = LLM_CALL_COOLDOWN - (current_time - last_llm_call_time)
            # print(f"--- LLM Cooldown: Waiting for {time_to_wait:.2f}s ---")
            time.sleep(time_to_wait)

        try:
            response = model.generate_content(prompt)
            last_llm_call_time = time.time() # Update last call time on success
            return response.text
        except genai.types.BlockedPromptException as e:
            print(f"Error querying LLM: Prompt was blocked by safety settings: {e}")
            return f"Error: Prompt blocked."
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                # It's a rate limit error
                retries += 1
                sleep_time = min(MAX_RETRY_DELAY, current_delay + random.uniform(0, current_delay * 0.1)) # Add jitter
                print(f"--- LLM Rate Limit Exceeded (429). Retrying {retries}/{MAX_RETRIES} in {sleep_time:.2f}s ---")
                time.sleep(sleep_time)
                current_delay *= 2 # Exponential increase
            else:
                # Other unexpected errors
                print(f"Error querying LLM: {e}")
                return f"Error: {e}"

    print(f"Error: Max retries ({MAX_RETRIES}) reached for LLM query due to quota issues. Aborting.")
    return "Error: LLM quota exceeded after multiple retries."
   
    
def extract_json_from_llm_output(text: str) -> str:
    """
    Attempts to extract a JSON string from an LLM's response,
    handling markdown code blocks and conversational filler.
    """
    # Remove any leading/trailing whitespace
    cleaned_text = text.strip()

    # Look for JSON markdown block
    json_match = re.search(r"```json\n(.*?)```", cleaned_text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # If no markdown block, try to find the first '{' and last '}' or first '[' and last ']'
    # This is a heuristic and might fail on very complex, non-standard outputs
    curly_brace_start = cleaned_text.find('{')
    curly_brace_end = cleaned_text.rfind('}')
    square_bracket_start = cleaned_text.find('[')
    square_bracket_end = cleaned_text.rfind(']')

    if curly_brace_start != -1 and curly_brace_end != -1 and curly_brace_end > curly_brace_start:
        return cleaned_text[curly_brace_start:curly_brace_end+1]
    elif square_bracket_start != -1 and square_bracket_end != -1 and square_bracket_end > square_bracket_start:
        return cleaned_text[square_bracket_start:square_bracket_end+1]
    
    return cleaned_text # Return as is if no clear JSON structure detected