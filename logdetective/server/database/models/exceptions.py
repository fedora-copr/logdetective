"""Database model exceptions for logdetective."""


class KojiTaskNotFoundError(Exception):
    """Exception raised when a koji task is not found"""


class KojiTaskNotAnalyzedError(Exception):
    """Exception raised when a koji task analysis is still in progress"""


class KojiTaskAnalysisTimeoutError(Exception):
    """Exception raised when a koji task analysis has timed out"""
