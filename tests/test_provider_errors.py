import json
from unittest.mock import patch

from lite.providers.clients import OpenAICompatibleModelClient
from lite.providers.errors import ProviderError, sanitize_url


def test_sanitize_url_drops_credentials_query_and_fragment_from_malformed_url():
    sanitized = sanitize_url("http://user:secret@[::1/v1?api_key=x#frag")

    assert "user" not in sanitized
    assert "secret" not in sanitized
    assert "api_key" not in sanitized
    assert "#" not in sanitized
    assert sanitized.startswith("http://")


def test_sanitize_url_drops_credentials_from_scheme_less_url():
    sanitized = sanitize_url("user:secret@example.com/v1?api_key=x#frag")

    assert sanitized == "example.com/v1"


def test_provider_error_metadata_sanitizes_url_credentials():
    error = ProviderError(
        "failed",
        provider="openai",
        model="gpt-test",
        base_url=(
            "https://user:secret@example.test:8443/v1"
            "?api_key=sk-real-secret#frag"
        ),
        code="server_error",
    )

    metadata = error.to_metadata()["provider_error"]

    assert metadata["base_url"] == "https://example.test:8443/v1"
    assert "secret" not in metadata["base_url"]
    assert "api_key" not in metadata["base_url"]


def test_provider_success_metadata_sanitizes_url_credentials():
    class FakeResponse:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"output_text": "<final>ok</final>"}).encode(
                "utf-8"
            )

    client = OpenAICompatibleModelClient(
        model="right.codes/codex-mini",
        base_url=(
            "https://user:secret@example.test:8443/v1"
            "?api_key=sk-real-secret"
        ),
        api_key="sk-test",
        temperature=0.2,
        timeout=30,
    )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert client.complete("hello", 42) == "<final>ok</final>"

    assert (
        client.last_completion_metadata["provider_base_url"]
        == "https://example.test:8443/v1"
    )


def test_provider_url_sanitizer_handles_invalid_ports_and_ipv6():
    assert (
        ProviderError(
            "failed",
            base_url="https://example.test:bad/v1?api_key=sk-real-secret",
        ).base_url
        == "https://example.test/v1"
    )
    assert (
        ProviderError(
            "failed", base_url="http://user:secret@[::1]:8080/v1"
        ).base_url
        == "http://[::1]:8080/v1"
    )
