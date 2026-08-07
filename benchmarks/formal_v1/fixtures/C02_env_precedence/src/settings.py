import os
from dataclasses import dataclass

@dataclass
class Settings:
    timeout_seconds: int = 5
    @classmethod
    def from_sources(cls, cli_timeout=None):
        return cls(int(os.getenv('APP_TIMEOUT', 5)))
