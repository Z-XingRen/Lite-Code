from pathlib import Path

from lite.evaluation.verifier import _select_bash_path


def test_windows_bash_selection_prefers_git_install_over_path_launcher(tmp_path):
    windows_launcher = tmp_path / "Windows" / "System32" / "bash.exe"
    git_bash = tmp_path / "Program Files" / "Git" / "bin" / "bash.exe"
    windows_launcher.parent.mkdir(parents=True)
    git_bash.parent.mkdir(parents=True)
    windows_launcher.touch()
    git_bash.touch()

    selected = _select_bash_path(
        platform_name="nt",
        discovered=str(windows_launcher),
        windows_candidates=(git_bash,),
    )

    assert selected == str(git_bash)


def test_non_windows_bash_selection_uses_discovered_path(tmp_path):
    discovered = tmp_path / "bin" / "bash"
    discovered.parent.mkdir()
    discovered.touch()

    selected = _select_bash_path(
        platform_name="posix",
        discovered=str(discovered),
        windows_candidates=(Path("unused"),),
    )

    assert selected == str(discovered)
