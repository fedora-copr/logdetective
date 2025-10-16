import pytest

from logdetective.server.utils import (
    format_analyzed_snippets,
    filter_snippets,
    get_version,
)
from logdetective.server.models import (
    AnalyzedSnippet,
    SnippetAnalysis,
    RatedSnippetAnalysis,
)

from tests.base.test_helpers import test_snippets

snippets_same_relevance = [
    AnalyzedSnippet(
        text=e[1],
        line_number=e[0],
        explanation=RatedSnippetAnalysis(
            text=f"Comment for snippet on line {e[0]}",
            relevance=55,
        ),
    )
    for e in test_snippets[1]
]

snippets_varied_relevance = [
    AnalyzedSnippet(
        text=e[1],
        line_number=e[0],
        explanation=RatedSnippetAnalysis(
            text=f"Comment for snippet on line {e[0]}",
            relevance=i**2,
        ),
    )
    for i, e in enumerate(test_snippets[1])
]

snippets_no_relevance = [
    AnalyzedSnippet(
        text=e[1],
        line_number=e[0],
        explanation=SnippetAnalysis(
            text=f"Comment for snippet on line {e[0]}",
        ),
    )
    for e in test_snippets[1]
]

test_analyzed_snippets_relevance = [
    snippets_varied_relevance,
    snippets_same_relevance,
]

test_analyzed_snippets = [
    snippets_no_relevance,
] + test_analyzed_snippets_relevance


@pytest.mark.parametrize("snippets", test_analyzed_snippets)
def test_format_analyzed_snippets(snippets):
    """Test snippet formatting for snippets with LLM generated explanations"""
    format_analyzed_snippets(snippets)


def test_filter_snippets_varied_relevance():
    """Test snippet filtering with rated snippets.
    Returned list must contain only portion of snippets in original list,
    and they must be sorted by line in ascending order."""

    filtered_snippets = filter_snippets(snippets_varied_relevance, 5)
    assert len(filtered_snippets) == 5

    last_line_n = 0
    for snippet in filtered_snippets:
        assert snippet.line_number > last_line_n
        last_line_n = snippet.line_number


def test_filter_snippets_same_relevance():
    """Test snippet filtering with rated snippets, when all snippets have the same relevance.
    Returned list must contain all snippets in the original list,
    and they must be sorted by line in ascending order."""

    filtered_snippets = filter_snippets(snippets_same_relevance, 5)
    assert len(filtered_snippets) == len(snippets_same_relevance)

    last_line_n = 0
    for snippet in filtered_snippets:
        assert snippet.line_number > last_line_n
        last_line_n = snippet.line_number


def test_filter_snippets_no_relevance():
    """Test that snippet filtering correctly raises Value error when encountering snippets
    without relevance score."""

    with pytest.raises(ValueError):
        filter_snippets(snippets_no_relevance, 5)


@pytest.mark.parametrize("snippets", test_analyzed_snippets_relevance)
def test_filter_too_large_k(snippets):
    """Test that filtering with k set to higher or equal to number of snippets
    doesn't change the list."""

    filtered_snippets = filter_snippets(snippets, len(snippets))
    assert len(filtered_snippets) == len(snippets)
    for i, snippet in enumerate(filtered_snippets):
        assert snippet.line_number == snippets[i].line_number
        assert snippet.explanation == snippets[i].explanation

    filtered_snippets = filter_snippets(snippets, len(snippets) + 1)
    assert len(filtered_snippets) == len(snippets)
    for i, snippet in enumerate(filtered_snippets):
        assert snippet.line_number == snippets[i].line_number
        assert snippet.explanation == snippets[i].explanation


def test_obtain_version_number():
    response = get_version()
    assert response.status_code == 200
