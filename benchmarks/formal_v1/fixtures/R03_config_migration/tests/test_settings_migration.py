from src.settings import load

def test_alias():
    assert load({'retries': 2})['max_retries'] == 2
