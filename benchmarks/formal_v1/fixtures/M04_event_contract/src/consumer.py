def consume(event, expected_type):
    if event.get('type') != expected_type:
        return None
    return event.get('body')
