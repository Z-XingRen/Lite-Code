def request(client, timeout_ms):
    return client.get(timeout=timeout_ms)
