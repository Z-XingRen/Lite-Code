from src.events import build_event
from src.consumer import consume

def test_event():
    assert consume(build_event('created', {'id': 1}), 'created')['id'] == 1
