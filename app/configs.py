import os

from pydantic import BaseSettings, __version__, BaseModel


class OpenAIModel(BaseModel):
    name: str = "gpt-3.5-turbo"
    temperature: int = 0


class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "chatassistantbot"
    PROJECT_DESCRIPTION: str = "Reldyn Buddy powered by OpenAI."
    COMPANY_NAME: str = "Reldyn"
    COMPANY_EMAIL: str = "hello@reldyn.com"
    PROJECT_VERSION: str = __version__
    KNOWLEDGE_INDEX_CACHE_DURATION: int = 1
    TIME_ZONE = "Asia/Jakarta"
    PROMPT_MAX_INPUT_SIZE = 4096
    PROMPT_NUM_OUTPUTS = 512
    PROMPT_MAX_CHUNK_OVERLAP = 20
    PROMPT_CHUNK_SIZE_LIMIT = 600
    OPENAI_MODEL = OpenAIModel()
    DEBUG = False

    class Config:
        # Place your .env file under this path
        env_file = ".env"
        env_prefix = "RELDYN_"
        case_sensitive = True


assert (
    os.getenv("OPENAI_API_KEY") is not None
), "Please set the OPENAI_API_KEY environment variable. " \
   "see https://platform.openai.com/account/api-keys"


try:
    import playwright
except ImportError:
    raise ValueError(
        "playwright not installed, which is needed Playwright module provides "
        "a method to launch a browser instance."
        "Please install it with `pip install playwright==1.30.0`."
    )


settings = Settings()
