import pytest

from logdetective.server.llm import format_analyzed_snippets
from logdetective.server.models import AnalyzedSnippet, Explanation

from tests.base.test_helpers import test_snippets

test_analyzed_snippets = [
    [
        AnalyzedSnippet(
            text=e[1],
            line_number=e[0],
            explanation=Explanation(
                text=f"Comment for snippet on line {e[0]}",
                logprobs=[{"logprob": 66.6}, {"logprob": 99.9}, {"logprob": 1.0}],
            ),
        )
        for e in test_snippets[1]
    ]
]


@pytest.mark.parametrize("snippets", test_analyzed_snippets)
def test_format_analyzed_snippets(snippets):
    """Test snippet formatting for snippets with LLM generated explanations"""
    format_analyzed_snippets(snippets)
