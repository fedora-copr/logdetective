# Tools

## Extractors

Agents use extractors to retrieve informative snippets from submitted artifacts.

Following extractors are available:

- `drain_extractor`: Base extractor using Drain3 algorithm
- `csgrep_extractor`: Optional extractor for retrieval GCC traces
- `python_traceback_extractor`: Optional extractor for retrieval of Python exception tracebacks

All optional extractors must be enabled in the `extractor` section of the config
file to be available.


### Using `csgrep_extractor`

When working with logs containing messages from GCC, it can be beneficial to employ
additional extractor based on `csgrep` tool, to ensure that the messages are kept intact.
Since `csgrep` is not available as a python package, it must be installed separately,
with a package manager or from [source](https://github.com/csutils/csdiff).

The binary is available as part of `csdiff` package on Fedora.

```sh
dnf install csdiff
```

While in server mode, the `csgrep` field in `extractor` config needs to be set to `true`.

```yaml
extractor:
  csgrep: true
```

The option is disabled by default and error will be produced if the option is used,
but `csgrep` is not present in the $PATH.

The container images are built with `csdiff` installed.


## Skipping Snippets

Certain log chunks may not contribute to the analysis of the problem under any circumstances.
Users can specify regular expressions matching such chunks using the Skip Snippets feature.

Patterns are defined in a TOML file. Each entry is a TOML table with a required `pattern` key
(a regular expression) and an optional `files` key listing exact filenames the pattern applies to.
When `files` is omitted the pattern applies to every log file processed.

Use single-quoted TOML strings for patterns — they are taken verbatim with no escape processing,
so backslashes and other special characters work as-is.

```toml
# applies to every file
[child_exit_code_zero]
pattern = '.*Child return code was: 0'

# applies only to backend.log and app.log
[skip_debug_messages]
pattern = '^DEBUG:.*'
files = ['backend.log', 'app.log']
```

Use the [default skip patterns](../logdetective/skip_snippets.toml) as a starting
point. Log Detective uses these patterns when no custom definition file is provided.

## Snippet lookup

Database of annotated snippets, associated issues and resolutions, is searched using
`annotated_snippet_lookup` when available.
Vector database backend uses provided snippet to find closest matches and retrieve
annotation, described failure and resolution.

The tool is only enabled if `annotation_lookup_tool` is set to true in `general`
section of the config, and if the embedding model is available.

Choice of snippet text used for searching the database is up to agent.

## Snippet annotation

`snippet_annotation` attaches a brief description to a useful extracted snippet.
Its `source_file` and `line_number` must match extractor output. The description
goes in the `snippet_analysis` argument.
