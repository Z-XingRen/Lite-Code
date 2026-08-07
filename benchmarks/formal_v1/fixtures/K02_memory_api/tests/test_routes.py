from src.routes import user_route

def test_route_shape(): assert user_route().endswith('/users')
