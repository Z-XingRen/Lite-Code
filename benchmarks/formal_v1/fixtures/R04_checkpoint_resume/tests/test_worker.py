from src.worker import order_jobs

def test_order():
    assert [j['id'] for j in order_jobs([{'id':2,'priority':1},{'id':1,'priority':2}])] == [1,2]
