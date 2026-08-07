from src.settings import Settings

def test_default():
    assert Settings.from_sources().timeout_seconds == 5
