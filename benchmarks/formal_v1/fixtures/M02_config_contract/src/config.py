from dataclasses import dataclass

@dataclass
class Config:
    timeout: int = 5

def load_config(data):
    return Config(timeout=int(data.get('timeout', 5)))
