"""Exception classes for Log Detective server."""


class LogDetectiveException(Exception):
    """Base exception for Log Detective server."""


class LogsMissingError(LogDetectiveException):
    """The logs are missing, possibly due to garbage-collection"""


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


class LogDetectiveMetricsError(LogDetectiveException):
    """Exception was encountered while recording metrics"""


class LogDetectiveArtifactsMissingError(LogDetectiveConnectionError):
    """Request for build artifacts has returned 404 response"""


class InvalidKojiTaskResultResponse(LogDetectiveKojiException):
    """Call to `getTaskResult` has returned an unexpected data structure"""


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
