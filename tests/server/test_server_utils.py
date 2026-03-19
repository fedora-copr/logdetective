import pytest

from fastapi.responses import Response as BasicResponse

from logdetective.server.utils import (
    get_version,
)


def test_obtain_version_number():
    """Test that we can retrieve a valid version string
    and that it can be used to make a response for API"""
    version = get_version()
    assert isinstance(version, str)
    assert len(version) != 0

    response = BasicResponse(content=get_version())

    assert response.status_code == 200
