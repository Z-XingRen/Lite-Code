from .config import Config

class Client:
    def __init__(self, config: Config):
        self.timeout = config.timeout
    def timeout_value(self):
        return self.timeout
