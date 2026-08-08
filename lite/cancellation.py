"""Run-scoped cooperative cancellation shared by runtime effects."""

import threading


class CancellationRequested(RuntimeError):
    """Raised when a run-scoped operation observes cancellation."""


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks = {}
        self._next_callback_id = 0

    @property
    def cancelled(self):
        return self._event.is_set()

    def cancel(self):
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def add_callback(self, callback):
        """Run ``callback`` once on cancellation and return an unregister hook."""

        with self._lock:
            if self._event.is_set():
                callback_id = None
            else:
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback
        if callback_id is None:
            try:
                callback()
            except Exception:
                pass

        def remove_callback():
            if callback_id is None:
                return
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return remove_callback

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    def raise_if_cancelled(self):
        if self.cancelled:
            raise CancellationRequested("operation cancelled")
