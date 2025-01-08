import pytest
from logdetective.utils import compute_certainty


@pytest.mark.parametrize("probs",(
    [{"a": 66.6}],
    [{"b": 99.9}, {"c": 1.0}]
))
def test_compute_certainty(probs):
    """ test compute_certainty and make sure we can use numpy correctly """
    compute_certainty(probs)
