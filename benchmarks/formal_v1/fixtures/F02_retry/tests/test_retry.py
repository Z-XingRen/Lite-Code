from src.retry import execute_with_retry

def test_success():
    assert execute_with_retry(lambda: 3, 2) == 3
