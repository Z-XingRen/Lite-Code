"""Cancellable subprocess execution with bounded process-tree cleanup."""

import ctypes
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

if os.name == "nt":
    from ctypes import wintypes

from ...cancellation import CancellationRequested


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
    terminator = _ProcessTreeTerminator(process)
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
    def __init__(self, process):
        self.process = process
        self._lock = threading.Lock()
        self._terminated = False
        self._job = _WindowsJob(process) if os.name == "nt" else None

    def terminate(self):
        with self._lock:
            if self._terminated:
                return
            self._terminated = True
        if os.name == "nt":
            if self._job is not None:
                self._job.terminate()
            _terminate_windows_tree(self.process)
        else:
            _terminate_posix_tree(self.process)

    def close(self):
        if self._job is not None:
            self._job.close()


def _terminate_windows_tree(process):
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    taskkill = system_root / "System32" / "taskkill.exe"
    try:
        subprocess.run(
            [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_TERMINATION_GRACE_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


if os.name == "nt":
    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]


    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]


    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class _WindowsJob:
    def __init__(self, process):
        self._handle = None
        if os.name != "nt":
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x00002000
        configured = kernel32.SetInformationJobObject(
            handle,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        assigned = configured and kernel32.AssignProcessToJobObject(
            handle, wintypes.HANDLE(process._handle)
        )
        if not assigned:
            kernel32.CloseHandle(handle)
            return
        self._handle = handle

    def terminate(self):
        if self._handle is not None:
            ctypes.windll.kernel32.TerminateJobObject(self._handle, 1)

    def close(self):
        handle, self._handle = self._handle, None
        if handle is not None:
            ctypes.windll.kernel32.CloseHandle(handle)


def _terminate_posix_tree(process):
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
