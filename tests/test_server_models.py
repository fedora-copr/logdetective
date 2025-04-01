import datetime
import pytest

from pydantic import ValidationError
from logdetective.server.models import TimePeriod


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
