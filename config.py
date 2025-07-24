# financial_mas_system/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# API Keys
# SWARMS_API_KEY is commented out as we're using the local 'swarms' library, not the cloud platform
# SWARMS_API_KEY = os.getenv("SWARMS_API_KEY") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Or OPENAI_API_KEY
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# LLM Configuration
LLM_MODEL_NAME = "gemini-2.0-flash" # Or "gpt-4", "gpt-3.5-turbo", etc.
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# Knowledge Graph Configuration
SHARED_KG_PATH = "knowledge_graphs/shared_financial_kg.gml" # GML is a simple graph format for NetworkX
PERSONAL_KG_DIR = "knowledge_graphs/personal_kbs/" # Directory to store individual agent KGs

# Simulation Environment
SIM_START_DATE = "2024-01-01" # Changed to current year for more realistic simulation
SIM_END_DATE = "2024-12-31"
SIM_INITIAL_CASH = 100000

# Agent Specifics
DEFAULT_AGENT_GOAL = "Optimize portfolio for maximum risk-adjusted returns."