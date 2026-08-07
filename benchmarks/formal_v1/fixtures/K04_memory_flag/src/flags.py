import os

def fast_path_enabled():
    return os.getenv('ENABLE_FAST', '0') == '1'
