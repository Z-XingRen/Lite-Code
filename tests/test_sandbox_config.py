import pytest

from lite.features.sandbox.config import SandboxConfig, resolve_sandbox_config


def test_sandbox_config_defaults_to_off():
    config = resolve_sandbox_config({})

    assert config == SandboxConfig()
    assert config.enabled is False


def test_sandbox_config_accepts_required_bubblewrap_mode():
    config = resolve_sandbox_config(
        {
            "sandbox": {
                "mode": "required",
                "backend": "bubblewrap",
                "workspace_write": False,
                "network_access": True,
                "container_image": "example/lite:dev",
                "excluded_commands": ["git *"],
                "filesystem": {
                    "extra_readonly_paths": ["/usr/bin"],
                    "extra_writable_paths": ["/tmp/cache"],
                    "deny_read": ["/tmp/private"],
                    "deny_write": ["/"],
                },
            }
        }
    )

    assert config.mode == "required"
    assert config.backend == "bubblewrap"
    assert config.workspace_write is False
    assert config.network_access is True
    assert config.container_image == "example/lite:dev"
    assert config.excluded_commands == ("git *",)
    assert config.extra_readonly_paths == ("/usr/bin",)
    assert config.extra_writable_paths == ("/tmp/cache",)
    assert config.deny_read == ("/tmp/private",)
    assert config.deny_write == ("/",)


def test_sandbox_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="sandbox.mode"):
        resolve_sandbox_config({"sandbox": {"mode": "strict"}})


def test_sandbox_config_accepts_cross_platform_backends():
    for backend in ("sandbox-exec", "docker", "podman"):
        config = resolve_sandbox_config(
            {"sandbox": {"mode": "required", "backend": backend}}
        )
        assert config.backend == backend
