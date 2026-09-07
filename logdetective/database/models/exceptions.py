"""Database model exceptions for logdetective."""


class AnalysisTaskNotFoundError(Exception):
    """Exception raised when analysis task is not found"""


class TaskNotAnalyzedError(Exception):
    """Exception raised when a task analysis is still in progress"""


class TaskAnalysisTimeoutError(Exception):
    """Exception raised when a task analysis has timed out"""


class AnalyzeRequestMetricsNotFoundError(Exception):
    """Exception raised when AnalyzeRequestMetrics is not found"""
