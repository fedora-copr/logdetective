import pytest
from logdetective.utils import (
    compute_certainty,
    format_snippets,
    format_analyzed_snippets,
)
from logdetective.server.models import AnalyzedSnippet, Explanation

test_snippets = [
    # Simple
    ["Snippet 1", "Snippet 2", "Snippet 3"],
    # Tuples
    [(10, "Snippet 1"), (120, "Snippet 1"), (240, "Snippet 1")],
]

test_analyzed_snippets = [
    [
        AnalyzedSnippet(
            text=e[1],
            line_number=e[0],
            explanation=Explanation(
                text=f"Comment for snippet on line {e[0]}",
                logprobs=[{"logprob": 66.6}, {"logprob": 99.9}, {"logprob": 1.0}]))
        for e in test_snippets[1]
    ]
]


@pytest.mark.parametrize(
    "probs", ([{"logprob": 66.6}], [{"logprob": 99.9}, {"logprob": 1.0}])
)
def test_compute_certainty(probs):
    """test compute_certainty and make sure we can use numpy correctly"""
    compute_certainty(probs)


@pytest.mark.parametrize("snippets", test_snippets)
def test_format_snippets(snippets):
    """Test snippet formatting with both simple snippets, and line numbers"""
    format_snippets(snippets)


@pytest.mark.parametrize("snippets", test_analyzed_snippets)
def test_format_analyzed_snippets(snippets):
    """Test snippet formatting for snippets with LLM generated explanations"""
    format_analyzed_snippets(snippets)
