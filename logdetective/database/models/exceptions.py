"""Database model exceptions for logdetective."""


class AnalysisTaskNotFoundError(Exception):
    """Exception raised when analysis task is not found"""


class TaskConflictError(Exception):
    """A client task id already exists with different ownership or input."""


class TaskTerminalError(Exception):
    """Cancellation was requested for an already terminal task."""
