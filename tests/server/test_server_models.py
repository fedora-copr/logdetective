import datetime
import pytest
import yaml

from pydantic import ValidationError
from logdetective.server.models import TimePeriod, Config, ExtractorConfig
from logdetective.extractors import DrainExtractor, CSGrepExtractor


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
        config = Config(data=config_def)
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

    extractors = config.get_extractors()
    assert len(extractors) == 1


def test_initialization_with_custom_data():
    """Tests that ExtractorConfig correctly uses custom values from a provided
    data dictionary and instantiates all relevant extractors.
    """

    custom_data = {
        "max_clusters": 15,
        "verbose": True,
        "max_snippet_len": 500,
        "csgrep": True,
    }
    config = ExtractorConfig(data=custom_data)

    assert config.max_clusters == 15
    assert config.verbose is True
    assert config.max_snippet_len == 500
    assert config.csgrep is True

    extractors = config.get_extractors()
    assert len(extractors) == 2


def test_csgrep_is_not_included_when_false():
    """Tests that get_extractors returns only the DrainExtractor when csgrep is explicitly False."""

    config = ExtractorConfig(data={"csgrep": False})

    extractors = config.get_extractors()
    assert len(extractors) == 1
    assert isinstance(extractors[0], DrainExtractor)


def test_csgrep_is_included_when_true():
    """Tests that get_extractors returns both extractors when csgrep is explicitly True."""
    config = ExtractorConfig(data={"csgrep": True})

    extractors = config.get_extractors()
    assert len(extractors) == 2
    assert isinstance(extractors[0], DrainExtractor)
    assert isinstance(extractors[1], CSGrepExtractor)
