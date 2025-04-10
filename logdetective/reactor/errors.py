class LogDetectiveReactorError(RuntimeError):
    """Base class for errors in the Log Detective Reactor"""


class LogsTooLargeError(LogDetectiveReactorError):
    """The log archive exceeds the configured maximum size"""
