from src.config_parser import parse_env_lines

def test_simple():
    assert parse_env_lines('A=1\n') == {'A': '1'}
