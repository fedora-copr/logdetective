test_snippets = [
    # Simple
    [
        "Snippet 1",
        "Snippet 2",
        "Snippet 3",
        "This is a snippet number 1",
        "This snippet has more than 2 characters",
        "This snippet contains capital A and \n",
        "",
        ".....=====....."
    ],
    # Tuples
    [(10, "Snippet 1"), (120, "Snippet 1"), (240, "Snippet 1")],
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
    "contains_x_followed_by_y": "x.*y"
}
