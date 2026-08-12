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

    process = subprocess.Popen(command, **kwargs)
    termination_acknowledgement = CancellationAcknowledgement()
    remove_acknowledgement = (
        cancellation_token.register_acknowledgement(termination_acknowledgement)
        if cancellation_token is not None
        else lambda: None
    )
    terminator = _ProcessTreeTerminator(process, termination_acknowledgement)
    remove_callback = (
        cancellation_token.add_callback(terminator.terminate)
        if cancellation_token is not None
        else lambda: None
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminator.terminate()
        stdout, stderr = _collect_after_termination(process)
        if cancellation_token is not None and cancellation_token.cancelled:
            raise CancellationRequested("shell command cancelled") from exc
        exc.stdout = stdout
        exc.stderr = stderr
        raise
    except BaseException:
        terminator.terminate()
        _collect_after_termination(process)
        raise
    finally:
        remove_callback()
        remove_acknowledgement()
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
        self._platform = PlatformProcessTree(process)

    def terminate(self):
        with self._lock:
            if self._terminated:
                return
            self._terminated = True
        try:
            self._platform.terminate()
            self._confirm_process_exit()
        finally:
            if self.process.poll() is not None:
                self.acknowledgement.acknowledge()

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
        self._platform.close()


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
