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
