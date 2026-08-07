from src.queue import Queue

def test_pop():
    assert Queue([1]).pop_safe() == 1
