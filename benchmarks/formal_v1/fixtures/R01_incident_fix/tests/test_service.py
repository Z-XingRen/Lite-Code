from src.service import request

class C:
    def get(self, **kwargs): return kwargs

def test_request():
    assert request(C(), 2500)['timeout'] == 2.5
