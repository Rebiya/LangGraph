"""
Configuration file for GraphFlow
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys (module-level snapshot; validate_config re-reads at call time)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Configuration
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "graphflow-project")
LIMIT_TOKENS = 1500
DATABASE_PATH = "conversations.db"

# Validate required API keys
def validate_config():
    """Validate that required API keys are present.

    Re-loads environment variables at call time to reflect runtime changes
    (e.g., tests modifying os.environ).
    """
    # Read live environment at call time (do not reload from .env here)
    current_google_key = os.getenv("GOOGLE_API_KEY")
    current_tavily_key = os.getenv("TAVILY_API_KEY")

    missing_keys = []
    if not current_google_key:
        missing_keys.append("GOOGLE_API_KEY")
    if not current_tavily_key:
        missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

    return True
