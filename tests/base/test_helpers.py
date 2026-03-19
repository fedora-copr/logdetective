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
def simple_log() -> list[str]:
    """Provides a simple log for testing."""
    return [
        "This is a test log.\n",
        "This is another test log.\n",
        "An error occurred: file not found.\n",
        "An error occurred: permission denied.\n",
        "Another line.\n",
        "",  # Empty line
        """This is a message with continuation:
            it continues here,
            here,
            and here.\n""",
        """This message is splint into an introduction:
            and a very long continuation, and a very long continuation, and a very long continuation,
            and a very long continuation ............."""
    ]


@pytest.fixture
def package_unavailable_log() -> str:
    return "\n".join(DNF_PACKAGE_UNAVAILABLE_EXPECTED_SNIPPETS)


@pytest.fixture
def csgrep_output_simple() -> str:
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
                        input_file="simple.log",
                        input_line=3,
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
                        input_file="simple.log",
                        input_line=4,
                        event="error",
                        message="An error occurred: permission denied.",
                        verbosity_level=1,
                    )
                ],
            ),
        ]
    ).model_dump_json()


@pytest.fixture
def siril_log_snippet() -> str:
    """see https://github.com/fedora-copr/logdetective-sample/blob/main/data/21ad14e5-f01f-4f88-ae60-eea095478e45/build.log#L657"""
    lines = [
        (
            "/usr/bin/ld: "
            "/builddir/build/BUILD/siril-1.2.5/redhat-linux-build/../src/gui/newdeconv.c:835:"
            "(.text+0x6003): undefined reference to `gf_estimate_kernel'"
        ),
        "collect2: error: ld returned 1 exit status",
        (
            "[234/235] g++  -o src/siril-cli src/siril-cli.p/main-cli.c.o -Wl,--as-needed "
            "-Wl,--no-undefined -Wl,--whole-archive -Wl,--start-group src/libsiril.a"
        ),
    ]
    return "\n".join(lines)


@pytest.fixture
def dolphin_emu_log_snippet() -> str:
    """see https://github.com/fedora-copr/logdetective-sample/blob/main/data/449ca0a3-a264-4f86-a3ec-b0a7b0af9b4d/build.log#L600"""
    lines = [
        (
            "/builddir/build/BUILD/dolphin-emu-2409-build/dolphin-2409/"
            "Source/Core/Common/MsgHandler.h:45:49: "
            "error: expected primary-expression before > token [-Wtemplate-body]"
        ),
        "   45 |   static_assert(fmt::detail::is_compile_string<S>::value);",
        "      |                                                 ^",
    ]
    return "\n".join(lines)


@pytest.fixture
def csgrep_output_siril() -> str:
    """Provides a sample csgrep JSON-validated output from an actual build log file."""
    return CSGrepOutput(
        defects=[
            CSGrepDefect(
                checker="COMPILER_WARNING",
                language="c/c++",
                tool="gcc",
                key_event_idx=0,
                events=[
                    CSGrepEvent(
                        file_name="collect2",
                        line=0,
                        input_file="siril.log",
                        input_line=2,
                        event="error",
                        message="ld returned 1 exit status",
                        verbosity_level=0,
                    )
                ]
            ),
        ]
    ).model_dump_json()


@pytest.fixture
def csgrep_output_dolphin_emu() -> str:
    """
    Provides a raw csgrep JSON output from an actual build log file -.
    Note: csgrep may include 'column' for some events,
    but it is irrelevant for Log Detective purposes
    """
    return (
        '{'
        '    "defects": ['
        '        {'
        '            "checker": "COMPILER_WARNING",'
        '            "language": "c/c++",'
        '            "tool": "gcc",'
        '            "key_event_idx": 0,'
        '            "events": ['
        '                {'
        '                    "file_name": "/builddir/build/BUILD/dolphin-emu-2409-build/dolphin-2409/Source/Core/Common/MsgHandler.h",'
        '                    "line": 45,'
        '                    "column": 49,'
        '                    "input_file": "dolphin-emu.log",'
        '                    "input_line": 1,'
        '                    "event": "error[-Wtemplate-body]",'
        '                    "message": "expected primary-expression before > token",'
        '                    "verbosity_level": 0'
        '                },'
        '                {'
        '                    "file_name": "",'
        '                    "line": 0,'
        '                    "input_file": "dolphin-emu.log",'
        '                    "input_line": 2,'
        '                    "event": "#",'
        '                    "message": "   45 |   static_assert(fmt::detail::is_compile_string<S>::value);",'
        '                    "verbosity_level": 1'
        '                },'
        '                {'
        '                    "file_name": "",'
        '                    "line": 0,'
        '                    "input_file": "dolphin-emu.log",'
        '                    "input_line": 3,'
        '                    "event": "#",'
        '                    "message": "      |                                                 ^",'
        '                    "verbosity_level": 1'
        '                }'
        '            ]'
        '        }'
        '    ]'
        '}'
    )
