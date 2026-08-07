def parse_env_lines(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split('=')
        result[key] = value
    return result
