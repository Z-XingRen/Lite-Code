"""Runtime secret redaction and shell environment helpers."""

import os

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"
_PLATFORM_SHELL_ENV = {
    name: os.environ[name]
    for name in ("SystemRoot", "WINDIR", "ComSpec", "PATHEXT")
    if name in os.environ
}


class RuntimeSecretsMixin:
    @staticmethod
    def looks_sensitive_env_name(name):
        upper = str(name).upper()
        return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)

    def is_secret_env_name(self, name):
        upper = str(name).upper()
        return upper in self.secret_env_names or self.looks_sensitive_env_name(upper)

    def configured_secret_env_items(self):
        items = [(name, value) for name, value in os.environ.items() if str(name).upper() in self.secret_env_names and value]
        items.sort(key=lambda item: item[0])
        return items

    def detected_secret_env_items(self):
        items = [(name, value) for name, value in os.environ.items() if self.is_secret_env_name(name) and value]
        items.sort(key=lambda item: item[0])
        return items

    def secret_env_summary(self):
        names = [name for name, _ in self.configured_secret_env_items()]
        return {"secret_env_count": len(names), "secret_env_names": names}

    def detected_secret_env_summary(self):
        names = [name for name, _ in self.detected_secret_env_items()]
        return {"secret_env_count": len(names), "secret_env_names": names}

    def redact_text(self, text):
        secret_items = sorted(
            self.detected_secret_env_items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        return self._redact_text_with_items(text, secret_items)

    @staticmethod
    def _redact_text_with_items(text, secret_items):
        text = str(text)
        for _, value in secret_items:
            text = text.replace(value, REDACTED_VALUE)
        return text

    def redact_artifact(self, value, key=None):
        secret_items = tuple(
            sorted(
                self.detected_secret_env_items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )
        return self._redact_artifact(value, key=key, secret_items=secret_items)

    def _redact_artifact(self, value, *, key, secret_items):
        if key and self.is_secret_env_name(key):
            return REDACTED_VALUE
        if isinstance(value, dict):
            return {
                str(item_key): self._redact_artifact(
                    item_value,
                    key=item_key,
                    secret_items=secret_items,
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [
                self._redact_artifact(item, key=key, secret_items=secret_items)
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                self._redact_artifact(item, key=key, secret_items=secret_items)
                for item in value
            ]
        if isinstance(value, str):
            return self._redact_text_with_items(value, secret_items)
        return value

    def shell_env(self):
        # Windows programs need SystemRoot even when callers intentionally
        # provide a minimal environment (for example, secret-isolation tests).
        env = dict(_PLATFORM_SHELL_ENV) if os.name == "nt" else {}
        env.update(
            {
                name: os.environ[name]
                for name in self.shell_env_allowlist
                if name in os.environ
            }
        )
        env["PWD"] = str(self.root)
        if "PATH" not in env and os.environ.get("PATH"):
            env["PATH"] = os.environ["PATH"]
        return env
