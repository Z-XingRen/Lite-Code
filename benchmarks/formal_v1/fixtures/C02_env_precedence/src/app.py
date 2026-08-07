from .settings import Settings

def timeout_for(cli_timeout=None):
    return Settings.from_sources(cli_timeout).timeout_seconds
