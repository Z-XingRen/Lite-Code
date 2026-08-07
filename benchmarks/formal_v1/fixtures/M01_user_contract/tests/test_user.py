from src.models import User
from src.serializer import serialize_user

def test_contract():
    assert serialize_user(User(1, 'A', 'a@x'))['name'] == 'A'
