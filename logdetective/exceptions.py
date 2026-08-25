"""Exception classes for Log Detective."""


class RemoteLogError(Exception):
    """Base exception for Log Detective remote log access."""
    status_code = 500  # Default: Internal Server Error


class RemoteLogRequestError(RemoteLogError):
    """Error when accessing a log via URL, possibly invalid URL."""
    status_code = 400  # Bad Request


class RemoteLogHeaderError(RemoteLogError):
    """Invalid Content-Length header when accessing a log via URL."""
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


class LogDetectiveException(Exception):
    """Base exception for Log Detective server."""


class LogsMissingError(LogDetectiveException):
    """The logs are missing, possibly due to garbage-collection"""


class LogDetectiveMetricsError(LogDetectiveException):
    """Exception was encountered while recording metrics"""


class LogDetectiveAgentResponseFailure(LogDetectiveException):
    """Log Detective agent did not return a valid response."""


class LogDetectiveInferenceError(LogDetectiveException):
    """Inference service encountered some issue."""
    http_status_code = 500


class LogDetectiveInferenceTimeout(LogDetectiveInferenceError):
    """Inference server took longer than allowed to respond."""
    http_status_code = 500


class LogDetectiveInferenceRateLimit(LogDetectiveInferenceError):
    """Inference service (temporarily) unavailable. Try again later."""
    http_status_code = 503


class LogDetectiveAgentTimeoutError(LogDetectiveException):
    """Agent didn't complete analysis on time"""
    http_status_code = 504


class LogDetectiveKojiException(LogDetectiveException):
    """Base exception for Koji-related errors."""


class KojiInvalidTaskID(LogDetectiveKojiException):
    """The task ID is invalid."""


class UnknownTaskType(LogDetectiveKojiException):
    """The task type is not supported."""


class NoFailedTask(LogDetectiveKojiException):
    """The task is not in the FAILED state."""


class LogDetectiveConnectionError(LogDetectiveKojiException):
    """A connection error occurred."""


class LogsTooLargeError(LogDetectiveKojiException):
    """The log archive exceeds the configured maximum size"""


class LogDetectiveArtifactsMissingError(LogDetectiveConnectionError):
    """Request for build artifacts has returned 404 response"""


class InvalidKojiTaskResultResponse(LogDetectiveKojiException):
    """Call to `getTaskResult` has returned an unexpected data structure"""
