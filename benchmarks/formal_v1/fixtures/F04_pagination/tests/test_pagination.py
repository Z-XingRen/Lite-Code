from src.pagination import paginate

def test_second_page():
    assert paginate([1,2,3,4,5], 2, 2) == [3,4]
