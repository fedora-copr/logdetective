import pytest
import yaml
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

import logdetective.server
from logdetective.models import APITokens
from logdetective.server import authenticate_api_token
from logdetective.utils import load_api_tokens


def make_request() -> Request:
    """Create a minimal request for direct dependency tests."""
    return Request({"type": "http", "method": "GET", "path": "/"})


def credentials(token: str) -> HTTPAuthorizationCredentials:
    """Create parsed HTTP bearer credentials."""
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_authentication_disabled_without_tokens(monkeypatch):
    monkeypatch.setattr(logdetective.server, "API_TOKENS", None)
    request = make_request()

    assert authenticate_api_token(request, None) is None
    assert request.state.api_token_name is None
    assert load_api_tokens(None) is None


def test_matching_token_returns_and_records_name(monkeypatch):
    monkeypatch.setattr(
        logdetective.server,
        "API_TOKENS",
        APITokens.model_validate(
            {"packit": "packit-secret", "monitoring": "monitoring-secret"}
        ),
    )
    request = make_request()

    assert authenticate_api_token(request, credentials("monitoring-secret")) == (
        "monitoring"
    )
    assert request.state.api_token_name == "monitoring"


@pytest.mark.parametrize("supplied_credentials", [None, credentials("wrong-secret")])
def test_missing_or_invalid_token_is_rejected_without_echoing_secret(
    monkeypatch, supplied_credentials
):
    monkeypatch.setattr(
        logdetective.server,
        "API_TOKENS",
        APITokens.model_validate({"packit": "secret"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        authenticate_api_token(make_request(), supplied_credentials)

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}
    assert "wrong-secret" not in exc_info.value.detail


def test_load_api_tokens(tmp_path):
    token_path = tmp_path / "api_tokens.yml"
    token_path.write_text(yaml.safe_dump({"packit": "secret"}), encoding="utf-8")

    tokens = load_api_tokens(str(token_path))

    assert tokens is not None
    assert tokens.root["packit"].get_secret_value() == "secret"
    assert "secret" not in repr(tokens)


def test_load_api_tokens_rejects_duplicate_names(tmp_path):
    token_path = tmp_path / "api_tokens.yml"
    token_path.write_text(
        "packit: first-secret\npackit: second-secret\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate API token name 'packit'") as exc_info:
        load_api_tokens(str(token_path))

    assert "first-secret" not in str(exc_info.value)
    assert "second-secret" not in str(exc_info.value)


@pytest.mark.parametrize("token", [" secret", "secret ", "\tsecret", "secret\n"])
def test_load_api_tokens_rejects_surrounding_whitespace(tmp_path, token):
    token_path = tmp_path / "api_tokens.yml"
    token_path.write_text(yaml.safe_dump({"packit": token}), encoding="utf-8")

    with pytest.raises(ValueError, match="surrounding whitespace"):
        load_api_tokens(str(token_path))


@pytest.mark.parametrize(
    "contents",
    [
        {},
        [],
        {"": "secret"},
        {"packit": ""},
        {1: "secret"},
        {"packit": 123},
        {"packit": "duplicate", "monitoring": "duplicate"},
    ],
)
def test_load_api_tokens_rejects_invalid_files(tmp_path, contents):
    token_path = tmp_path / "api_tokens.yml"
    token_path.write_text(yaml.safe_dump(contents), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_api_tokens(str(token_path))

    assert "duplicate" not in str(exc_info.value)
