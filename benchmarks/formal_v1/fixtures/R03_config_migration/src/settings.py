def load(data):
    return {'retries': int(data.get('retries', 3))}

def dump(settings):
    return {'retries': int(settings['retries'])}
