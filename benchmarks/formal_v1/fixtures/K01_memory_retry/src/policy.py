MAX_ATTEMPTS = 1
RETRY_VALIDATION_ERRORS = True

def retry_policy():
    return {'max_attempts': MAX_ATTEMPTS, 'retry_validation_errors': RETRY_VALIDATION_ERRORS}
