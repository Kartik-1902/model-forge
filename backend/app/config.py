from functools import lru_cache
from pydantic_settings import BaseSettings ,SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

    app_name: str = 'Model Forge'
    app_version: str = '0.1.0'
    debug: bool = False
    environment:str = 'development'
    
    port: int = 8000
    host: str = '127.0.0.1'

    database_url: str = 'postgresql+asyncpg://postgres:postgres@localhost:5432/modelforge'

    redis_url: str = 'redis://localhost:6379/0'

    mlflow_tracking_uri: str = 'http://localhost:5001'

    api_key_header: str = 'X-API-Key'
    bootstrap_admin_key: str | None = None

@lru_cache
def get_settings() -> Settings:
    return Settings()