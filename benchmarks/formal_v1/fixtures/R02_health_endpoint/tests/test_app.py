from src.app import health

def test_health_status():
    assert health()['status'] in {'starting','ok'}
