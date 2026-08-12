"""Windows process-tree termination for cancellable sandbox commands."""

import ctypes
import os
import subprocess
import time
from ctypes import wintypes
from pathlib import Path


_TERMINATION_GRACE_SECONDS = 1.0


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


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _WindowsJob:
    def __init__(self, process):
        self._handle = None
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
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
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

    def wait_for_exit(self, timeout):
        if self._handle is None:
            return None
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            info = _BasicAccountingInformation()
            returned = wintypes.DWORD()
            queried = ctypes.windll.kernel32.QueryInformationJobObject(
                self._handle,
                1,
                ctypes.byref(info),
                ctypes.sizeof(info),
                ctypes.byref(returned),
            )
            if not queried:
                return False
            if info.ActiveProcesses == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def close(self):
        handle, self._handle = self._handle, None
        if handle is not None:
            ctypes.windll.kernel32.CloseHandle(handle)


class PlatformProcessTree:
    def __init__(self, process):
        self.process = process
        self._job = _WindowsJob(process)
        self._taskkill_completed = False

    def terminate(self):
        self._job.terminate()
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        taskkill = system_root / "System32" / "taskkill.exe"
        try:
            completed = subprocess.run(
                [str(taskkill), "/PID", str(self.process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_TERMINATION_GRACE_SECONDS,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._taskkill_completed = completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            pass
        if self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass

    def wait_for_exit(self, timeout):
        job_result = self._job.wait_for_exit(timeout)
        if job_result is not None:
            return job_result
        return self._taskkill_completed and self.process.poll() is not None

    def close(self):
        self._job.close()
