import datetime
import pytest
from pytest_mock import MockerFixture
import yaml

from pydantic import ValidationError
from logdetective.server.models import (
    TimePeriod,
    Config,
    ExtractorConfig,
    ArtifactFile,
    AnalysisRequest,
)
from logdetective.constants import DEFAULT_MAXIMUM_ARTIFACT_MIB
from logdetective.utils import mib_to_bytes


def test_TimePeriod():
    p = TimePeriod(weeks=3)
    assert p.weeks == 3

    p = TimePeriod()
    assert p.days == 2

    with pytest.raises(ValueError):
        TimePeriod(days=5, hours=5)

    with pytest.raises(ValidationError):
        TimePeriod(hours=-3)


def test_TimePeriod_get_period_start_time():
    p = TimePeriod(weeks=3)
    now = datetime.datetime.now(datetime.timezone.utc)
    d1 = p.get_period_start_time(now)
    d2 = now - datetime.timedelta(weeks=3)

    assert d1 == d2


def test_parse_deployed_config():
    with open("server/config.yml", "r", encoding="utf8") as config_file:
        config_def = yaml.safe_load(config_file)
        config = Config.model_validate(config_def)
        assert config


def test_default_initialization_and_configuration():
    """Tests that ExtractorConfig initializes with default values and configures
    only the DrainExtractor when no data is provided.
    """

    config = ExtractorConfig()

    assert config.max_clusters == 8
    assert config.verbose is False
    assert config.max_snippet_len == 2000
    assert config.csgrep is False


def test_initialization_with_custom_data(mocker: MockerFixture):
    """Tests that ExtractorConfig correctly uses custom values from a provided
    data dictionary and instantiates all relevant extractors.
    """
    mocker.patch("logdetective.server.models.check_csgrep", return_value=True)

    custom_data = {
        "max_clusters": 15,
        "verbose": True,
        "max_snippet_len": 500,
        "csgrep": True,
    }
    config = ExtractorConfig.model_validate(custom_data)

    assert config.max_clusters == 15
    assert config.verbose is True
    assert config.max_snippet_len == 500
    assert config.csgrep is True


class TestAnalysisRequestValidation:
    """Check proper validation of requests submitted to LLM analysis.

    This includes /analyze endpoint.
    Since we currently support log submission via URL and via file list with logs
    passed as raw strings, we also check various field and model validators.

    NOTE: In the future, we will add also various URL validators.
    Tests for these cases can be included here.

    NOTE: We do not test for long content size in BuildArtifactFile, since this is
    treated on the request level by pre-fetching Content-Length header.
    """

    # pylint: disable=missing-function-docstring

    def test_invalid_field_present(self):
        with pytest.raises(ValidationError):
            AnalysisRequest(
                files=[ArtifactFile(name="test.log", content="test content")],
                invalid_field="should fail",
            )
        with pytest.raises(ValidationError):
            AnalysisRequest(
                files=[ArtifactFile(name="test.log", content="test content")],
                invalid_field="should fail",
            )

    def test_file_missing_name(self):
        with pytest.raises(ValidationError):
            ArtifactFile(content="test content")

    def test_file_missing_content(self):
        with pytest.raises(ValidationError):
            ArtifactFile(name="test.log")

    def test_empty_files_list(self):
        """Empty files list should be caught (CRITICAL BUG TEST)"""
        with pytest.raises(
            ValidationError, match="Value should have at least 1 item after validation"
        ):
            AnalysisRequest(files=[])

    def test_empty_file_name(self):
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            ArtifactFile(name="", content="test content")

    def test_invalid_file_name(self):
        with pytest.raises(ValidationError, match="String should match pattern"):
            AnalysisRequest(
                files=[ArtifactFile(name="bad@log\\name", content="test content")]
            )

    def test_duplicit_file_name(self):
        with pytest.raises(ValidationError, match="Duplicate filenames detected"):
            AnalysisRequest(
                files=[
                    ArtifactFile(name="build.log", content="test content"),
                    ArtifactFile(name="build.log", content="test content"),
                ]
            )
