def execute_with_retry(action, max_attempts):
    last = None
    for _ in range(max_attempts - 1):
        try:
            return action()
        except Exception as exc:
            last = exc
    if last:
        raise last
    return action()
