"""configuration settings for the application"""

# openai model settings
OPENAI_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.0,
    "max_tokens": 100
}

# paths
DATA_PATHS = {
    "raw_data": "data/raw",
    "processed_data": "data/processed",
    "output": "data/output"
}
