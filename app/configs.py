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

    model_cost_mapping = {
        "gpt-4": 0.03,
        "gpt-4-0314": 0.03,
        "gpt-4-completion": 0.06,
        "gpt-4-0314-completion": 0.06,
        "gpt-4-32k": 0.06,
        "gpt-4-32k-0314": 0.06,
        "gpt-4-32k-completion": 0.12,
        "gpt-4-32k-0314-completion": 0.12,
        "gpt-3.5-turbo": 0.002,
        "gpt-3.5-turbo-0301": 0.002,
        "text-ada-001": 0.0004,
        "ada": 0.0004,
        "text-babbage-001": 0.0005,
        "babbage": 0.0005,
        "text-curie-001": 0.002,
        "curie": 0.002,
        "text-davinci-003": 0.02,
        "text-davinci-002": 0.02,
        "code-davinci-002": 0.02,
    }

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
