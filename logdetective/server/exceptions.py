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
