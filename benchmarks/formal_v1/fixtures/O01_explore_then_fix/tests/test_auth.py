from src.auth import authorize

def test_truthy(): assert authorize('x') is True
