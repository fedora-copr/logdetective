"""Exception classes for Log Detective base tool/utility."""


class RemoteLogError(Exception):
    """Base exception for Log Detective remote log access."""
    status_code = 500  # Default: Internal Server Error


class RemoteLogRequestError(RemoteLogError):
    """Error when accessing a log via URL, possibly invalid URL."""
    status_code = 400  # Bad Request


class RemoteLogHeaderError(RemoteLogError):
    """Missing or invalid header when accessing a log via URL."""
    status_code = 411  # Length Required


class RemoteLogAccessError(RemoteLogError):
    """Access via URL failed due to network errors etc.

    In server environment, this indicates an issue with where the log is
    supposed to be, not with our server, so 502 status makes sense.
    """
    status_code = 502  # Bad Gateway


class RemoteLogTooLargeError(RemoteLogError):
    """The log accessed via URL exceeds the configured maximum size."""
    status_code = 413  # Content Too Large
