"""POSIX process-tree termination for cancellable sandbox commands."""

import os
import signal
import time


class PlatformProcessTree:
    def __init__(self, process):
        self.process = process

    def terminate(self):
        process = self.process
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            try:
                process.terminate()
            except OSError:
                return
        deadline = time.monotonic() + 0.2
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass

    def wait_for_exit(self, timeout):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def close(self):
        return None
