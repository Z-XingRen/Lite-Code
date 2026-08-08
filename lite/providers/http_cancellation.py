"""Thread-safe lifecycle tracking for cancellable provider HTTP responses."""

import threading

from ..cancellation import CancellationRequested


class HttpResponseController:
    """Track active responses so cancellation and explicit abort can close them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._responses = {}

    def register(self, response, cancellation_token=None):
        registration = _HttpResponseRegistration(self, response)
        with self._lock:
            self._responses[registration.key] = registration
        try:
            if cancellation_token is not None:
                registration.remove_cancel_callback = (
                    cancellation_token.add_callback(registration.cancel)
                )
                cancellation_token.raise_if_cancelled()
        except BaseException:
            registration.release()
            raise
        return registration

    def abort(self):
        with self._lock:
            registrations = tuple(self._responses.values())
        for registration in registrations:
            registration.cancel()

    def release(self, key):
        with self._lock:
            self._responses.pop(key, None)


class _HttpResponseRegistration:
    def __init__(self, controller, response):
        self.controller = controller
        self.response = response
        self.key = id(self)
        self.remove_cancel_callback = None
        self._release_lock = threading.Lock()
        self._released = False
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()
        self.close_response()

    def raise_if_cancelled(self):
        if self._cancelled.is_set():
            raise CancellationRequested("provider HTTP response cancelled")

    def close_response(self):
        _close_response(self.response)

    def release(self):
        with self._release_lock:
            if self._released:
                return
            self._released = True
        remove_callback = self.remove_cancel_callback
        if remove_callback is not None:
            remove_callback()
        self.controller.release(self.key)
        self.close_response()


def _close_response(response):
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
