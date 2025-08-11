import pytest
from fastapi import HTTPException
import logdetective.server.server
from logdetective.server.server import requires_token_when_set


INVALID_HEADER = "Bearer:SOMETHING"
VALID_HEADER_MATCHING = "Bearer sometoken"
VALID_HEADER_NOT_MATCHING = "Bearer othertoken"
API_TOKEN = "sometoken"


@pytest.fixture
def set_token(monkeypatch):
    monkeypatch.setattr(logdetective.server.server, "API_TOKEN", API_TOKEN)


def test_unset_token():
    assert requires_token_when_set("") is None


def test_invalid_header(set_token):
    with pytest.raises(HTTPException):
        requires_token_when_set(INVALID_HEADER)


def test_valid_header_not_matching(set_token):
    with pytest.raises(HTTPException):
        requires_token_when_set(VALID_HEADER_NOT_MATCHING)


def test_valid_header_matching(set_token):
    assert requires_token_when_set(VALID_HEADER_MATCHING) is None
