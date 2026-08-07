from src.config import load_config

def test_old_input_alias():
    assert load_config({'timeout': 7}).timeout_seconds == 7
