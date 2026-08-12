"""Run-scoped cooperative cancellation shared by runtime effects."""

import threading
import time


class CancellationRequested(RuntimeError):
    """Raised when a run-scoped operation observes cancellation."""


class CancellationAcknowledgement:
    """Signals that a cancellable operation completed its cleanup."""

    def __init__(self):
        self._event = threading.Event()

    @property
    def acknowledged(self):
        return self._event.is_set()

    def acknowledge(self):
        self._event.set()

    def wait(self, timeout=None):
        return self._event.wait(timeout)


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks = {}
        self._acknowledgements = {}
        self._next_callback_id = 0
        self._next_acknowledgement_id = 0

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

    def register_acknowledgement(self, acknowledgement):
        """Register cleanup state that callers can await after cancellation."""

        if not isinstance(acknowledgement, CancellationAcknowledgement):
            raise TypeError("acknowledgement must be CancellationAcknowledgement")
        with self._lock:
            if self._event.is_set():
                immediate = True
                acknowledgement_id = None
            else:
                immediate = False
                acknowledgement_id = self._next_acknowledgement_id
                self._next_acknowledgement_id += 1
                self._acknowledgements[acknowledgement_id] = acknowledgement
        if immediate:
            acknowledgement.acknowledge()

        def remove_acknowledgement():
            if acknowledgement_id is None:
                return
            with self._lock:
                if not self._event.is_set():
                    self._acknowledgements.pop(acknowledgement_id, None)

        return remove_acknowledgement

    def wait_for_acknowledgements(self, timeout=None):
        """Wait until all cancellation cleanup registered on this token is done."""

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._lock:
            acknowledgements = tuple(self._acknowledgements.values())
        for acknowledgement in acknowledgements:
            remaining = None
            if timeout is not None:
                remaining = max(0.0, deadline - time.monotonic())
            if not acknowledgement.wait(remaining):
                return False
        return True

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    def raise_if_cancelled(self):
        if self.cancelled:
            raise CancellationRequested("operation cancelled")
