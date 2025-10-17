import pytest

from logdetective.models import CSGrepOutput, CSGrepDefect, CSGrepEvent


test_snippets = [
    # Simple
    [
        "Snippet 1",
        "Snippet 2",
        "Snippet 3",
    ],
    # Tuples
    [
        (10, "Snippet 1"),
        (120, "Snippet 2"),
        (240, "Snippet 3"),
        (250, "Snippet 4"),
        (255, "Snippet 5"),
        (268, "Snippet 6"),
        (300, "Snippet 7"),
        (303, "Snippet 8"),
        (9999, "Snippet Final"),
    ],
]

test_prompts = """
prompt_template: This is basic template.

snippet_prompt_template: This is template for snippets.
"""

# For testing snippet filtering
test_filter_patterns = {
    "starts_one_or_two": "^[12]",
    "starts_with_capital_a": "^A",
    "contains_c": ".*c.*",
    "contains_x_followed_by_y": "x.*y",
}
# Following must be kept in sync with `test_filter_patterns`
# bool values of tuples must equal to output of the `filter_snippet_patterns`
test_snippets_filtering = [
    ("This is a snippet number 1", False),
    ("This snippet has more than 2 characters", True),
    ("This snippet contains capital A and \n", True),
    ("", False),
    (".....=====.....", False),
    ("A nice matching snippet", True),
    ("x is a good name for independent variable, unlike y", True),
    ("1. This snippet should be skipped", True),
]


# pylint: disable=line-too-long
DNF_PACKAGE_UNAVAILABLE_EXPECTED_SNIPPETS = [
    """WARNING: DNF5 command failed, retrying, attempt #1, sleeping 10s""",
    """Updating and loading repositories:
 Copr repository                        100% |  32.8 KiB/s |   1.5 KiB |  00m00s""",
    """Problem: conflicting requests
  - nothing provides python(abi) = 3.11 needed by python3-something.fc37.noarch from copr_base
  - nothing provides python3.11dist(typing-extensions) >= 3.7.4 needed by python3-something.fc37.noarch from copr_base
  - nothing provides python3.11dist(setuptools) >= 16 needed by python3-something.fc37.noarch from copr_base
  - nothing provides python3.11dist(opentelemetry-api) = 1.11.1 needed by python3-something.fc37.noarch from copr_base
  - nothing provides python3.11dist(opentelemetry-semantic-conventions) = 0.30~b1 needed by python3-something.fc37.noarch from copr_base""",
    """You can try to add to command line:
  --skip-unavailable to skip unavailable packages""",
]


@pytest.fixture
def simple_log():
    """Provides a simple log for testing."""
    return """This is a test log.
This is another test log.
An error occurred: file not found.
An error occurred: permission denied.
Another line."""


@pytest.fixture
def package_unavailable_log():
    return "\n".join(DNF_PACKAGE_UNAVAILABLE_EXPECTED_SNIPPETS)


@pytest.fixture
def csgrep_output():
    """Provides a sample csgrep JSON output using the new data structures."""
    return CSGrepOutput(
        defects=[
            CSGrepDefect(
                checker="some-checker",
                language="C",
                tool="gcc",
                key_event_idx=0,
                events=[
                    CSGrepEvent(
                        file_name="test.c",
                        line=3,
                        event="error",
                        message="An error occurred: file not found.",
                        verbosity_level=1,
                    )
                ],
            ),
            CSGrepDefect(
                checker="another-checker",
                language="C++",
                tool="g++",
                key_event_idx=0,
                events=[
                    CSGrepEvent(
                        file_name="test.cpp",
                        line=4,
                        event="error",
                        message="An error occurred: permission denied.",
                        verbosity_level=1,
                    )
                ],
            ),
        ]
    ).model_dump_json()
