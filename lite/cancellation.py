"""Run-scoped cooperative cancellation shared by runtime effects."""

import threading


class CancellationRequested(RuntimeError):
    """Raised when a run-scoped operation observes cancellation."""


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    @property
    def cancelled(self):
        return self._event.is_set()

    def cancel(self):
        self._event.set()

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    def raise_if_cancelled(self):
        if self.cancelled:
            raise CancellationRequested("operation cancelled")
