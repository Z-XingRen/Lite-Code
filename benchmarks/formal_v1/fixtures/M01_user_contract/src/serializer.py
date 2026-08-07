from .models import User

def serialize_user(user: User):
    return {'id': user.user_id, 'name': user.name, 'email': user.email}
