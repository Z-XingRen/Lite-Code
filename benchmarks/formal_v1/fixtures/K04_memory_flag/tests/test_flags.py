from src.flags import fast_path_enabled

def test_default(): assert fast_path_enabled() is False
