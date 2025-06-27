from dotenv import load_dotenv
import os

load_dotenv()  # Load .env into environment variables

class Settings:
    model_path = os.getenv("MODEL_PATH", "models")
    model_name = os.getenv("MODEL_NAME", "rf_model.pkl")

settings = Settings()