from src.pricing import calculate_total

def test_basic():
    assert calculate_total(100, 15, 8.5) == 93.5
