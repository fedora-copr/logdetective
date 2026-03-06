import datetime
import pytest
from pytest_mock import MockerFixture
import yaml

from pydantic import ValidationError
from logdetective.server.models import (
    TimePeriod,
    Config,
    ExtractorConfig,
    BuildLogFile,
    BuildLogRequest,
)
from logdetective.constants import MAXIMUM_LOG_LENGTH


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


class TestBuildLogRequestValidation:
    """Check proper validation of requests submitted to LLM analysis.

    This includes /analyze, /analyze/staged, /analyze/stream endpoints.
    Since we currently support log submission via URL and via file list with logs
    passed as raw strings, we also check various field and model validators.

    NOTE: In the future, we will add also various URL validators.
    Tests for these cases can be included here.
    """
    # pylint: disable=missing-function-docstring

    def test_missing_url_and_files(self):
        with pytest.raises(ValidationError, match="Must provide exactly one"):
            BuildLogRequest()

    def test_url_and_files_present(self):
        with pytest.raises(ValidationError, match="Must provide exactly one"):
            BuildLogRequest(
                url="http://example.com/log.txt",
                files=[BuildLogFile(name="test.log", content="test content")]
            )

    def test_invalid_field_present(self):
        with pytest.raises(ValidationError):
            BuildLogRequest(
                url="http://example.com/log.txt",
                invalid_field="should fail"
            )
        with pytest.raises(ValidationError):
            BuildLogRequest(
                files=[
                    BuildLogFile(name="test.log", content="test content")
                ],
                invalid_field="should fail"
            )

    def test_file_missing_name(self):
        with pytest.raises(ValidationError):
            BuildLogFile(content="test content")

    def test_file_missing_content(self):
        with pytest.raises(ValidationError):
            BuildLogFile(name="test.log")

    def test_file_content_too_long(self):
        with pytest.raises(ValidationError, match="String should have at most"):
            # NOTE: not really sure if there is a better way of testing this...
            BuildLogFile(name="test.log", content="x" * (MAXIMUM_LOG_LENGTH + 1))

    def test_empty_files_list(self):
        """Empty files list should be caught (CRITICAL BUG TEST)"""
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            BuildLogRequest(files=[])

    def test_empty_file_name(self):
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            BuildLogFile(name="", content="test content")

    def test_invalid_file_name(self):
        with pytest.raises(ValidationError, match="String should match pattern"):
            BuildLogRequest(files=[
                BuildLogFile(name="bad@log\\name", content="test content")
            ])

    def test_duplicit_file_name(self):
        with pytest.raises(ValidationError, match="Duplicate filenames detected"):
            BuildLogRequest(files=[
                BuildLogFile(name="build.log", content="test content"),
                BuildLogFile(name="build.log", content="test content"),
            ])
