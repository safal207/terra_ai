from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_dir: Path = Path.home() / ".terra" / "data"
    cache_dir: Path = Path.home() / ".terra" / "cache"
    seed: int = 42

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)
