from src.policy import retry_policy

def test_shape():
    assert set(retry_policy()) == {'max_attempts','retry_validation_errors'}
