# core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str

    # JWT settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        # This tells Pydantic to read the variables from a .env file
        env_file = ".env"

# Create a single instance of the settings that can be imported elsewhere
settings = Settings()