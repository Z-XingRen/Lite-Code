"""Cancellable subprocess execution with bounded process-tree cleanup."""

import os
import subprocess
import threading

from ...cancellation import CancellationAcknowledgement, CancellationRequested

if os.name == "nt":
    from .process_windows import PlatformProcessTree
else:
    from .process_posix import PlatformProcessTree


_TERMINATION_GRACE_SECONDS = 1.0
_TERMINATION_WAIT_SECONDS = 4.0


def run_cancellable_process(
    command,
    *,
    cwd,
    env,
    timeout,
    shell=False,
    executable=None,
    cancellation_token=None,
):
    """Run a captured process and terminate its tree on cancel or timeout."""

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    kwargs = {
        "cwd": cwd,
        "env": env,
        "shell": shell,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if executable is not None:
        kwargs["executable"] = executable
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    termination_acknowledgement = CancellationAcknowledgement()
    remove_acknowledgement = (
        cancellation_token.register_acknowledgement(termination_acknowledgement)
        if cancellation_token is not None
        else lambda: None
    )
    process = None
    terminator = None

    def remove_callback():
        return None

    try:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        process = subprocess.Popen(command, **kwargs)
        terminator = _ProcessTreeTerminator(process, termination_acknowledgement)
        remove_callback = (
            cancellation_token.add_callback(terminator.terminate)
            if cancellation_token is not None
            else lambda: None
        )
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        assert terminator is not None
        terminator.terminate()
        stdout, stderr = _collect_after_termination(process)
        if cancellation_token is not None and cancellation_token.cancelled:
            raise CancellationRequested("shell command cancelled") from exc
        exc.stdout = stdout
        exc.stderr = stderr
        raise
    except BaseException:
        if terminator is not None and process is not None:
            terminator.terminate()
            _collect_after_termination(process)
        else:
            termination_acknowledgement.acknowledge()
        raise
    finally:
        remove_callback()
        remove_acknowledgement()
        if terminator is not None:
            terminator.close()

    if cancellation_token is not None and cancellation_token.cancelled:
        raise CancellationRequested("shell command cancelled")
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


class _ProcessTreeTerminator:
    def __init__(self, process, acknowledgement):
        self.process = process
        self.acknowledgement = acknowledgement
        self._lock = threading.Lock()
        self._terminated = False
        self._finished = threading.Event()
        self._error = None
        self._platform = PlatformProcessTree(process)

    def terminate(self):
        with self._lock:
            if self._terminated:
                owner = False
            else:
                self._terminated = True
                owner = True
        if not owner:
            self._wait_for_termination()
            return
        try:
            self._platform.terminate()
            self._confirm_process_exit()
            if not self._platform.wait_for_exit(_TERMINATION_GRACE_SECONDS):
                raise RuntimeError("process tree cleanup timed out")
            if self.process.poll() is None:
                raise RuntimeError("process exited without a terminal status")
            self.acknowledgement.acknowledge()
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._finished.set()

    def _confirm_process_exit(self):
        try:
            self.process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                self.process.kill()
            except OSError:
                pass
            self.process.wait(timeout=_TERMINATION_GRACE_SECONDS)

    def close(self):
        if self._terminated:
            self._wait_for_termination()
        self._platform.close()

    def _wait_for_termination(self):
        if not self._finished.wait(_TERMINATION_WAIT_SECONDS):
            raise RuntimeError("process tree termination did not finish")
        if self._error is not None:
            raise RuntimeError("process tree termination failed") from self._error


def _collect_after_termination(process):
    try:
        return process.communicate(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired as exc:
        try:
            process.kill()
        except OSError:
            pass
        try:
            return process.communicate(timeout=_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except (AttributeError, OSError):
                    pass
            try:
                process.wait(timeout=_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired as cleanup_exc:
                raise RuntimeError("shell process cleanup timed out") from cleanup_exc
            for name in ("stdout_thread", "stderr_thread"):
                thread = getattr(process, name, None)
                if thread is not None and thread.is_alive():
                    thread.join(_TERMINATION_GRACE_SECONDS)
                if thread is not None and thread.is_alive():
                    raise RuntimeError("shell pipe reader cleanup timed out")
            return exc.stdout or "", exc.stderr or ""
