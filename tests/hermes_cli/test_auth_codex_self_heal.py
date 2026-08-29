"""Regression tests for isolation of Hermes-owned Codex credentials."""

import json

import pytest

import hermes_cli.auth as auth
from hermes_cli.auth import (
    AuthError,
    _refresh_codex_auth_tokens,
    resolve_codex_runtime_credentials,
)


STALE = {"access_token": "stale-access", "refresh_token": "stale-refresh"}


def test_terminal_refresh_error_requires_fresh_hermes_login(monkeypatch):
    """A dead Hermes grant must not be replaced with another client's cache."""

    def _rejected(*_args, **_kwargs):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)

    with pytest.raises(AuthError) as exc_info:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert exc_info.value.code == "invalid_grant"
    assert exc_info.value.relogin_required is True


def test_missing_singleton_access_token_does_not_import_codex_cli_cache(
    tmp_path, monkeypatch
):
    """Malformed Hermes state fails closed even when Codex CLI is logged in."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {"refresh_token": "stale-refresh"},
                        "last_refresh": "2026-06-01T00:00:00Z",
                        "auth_mode": "chatgpt",
                    },
                },
            }
        )
    )
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "fresh-access",
                    "refresh_token": "fresh-refresh",
                },
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(AuthError) as exc_info:
        resolve_codex_runtime_credentials()

    assert exc_info.value.code == "codex_auth_missing_access_token"
    stored = json.loads((hermes_home / "auth.json").read_text())
    tokens = stored["providers"]["openai-codex"]["tokens"]
    assert "access_token" not in tokens
    assert tokens["refresh_token"] == "stale-refresh"
