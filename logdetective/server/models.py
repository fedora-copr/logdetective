from typing import List, Dict, Optional
from pydantic import BaseModel


class BuildLog(BaseModel):
    """Model of data submitted to API."""

    url: str


class Response(BaseModel):
    """Model of data returned by Log Detective API

    explanation: CreateCompletionResponse
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama_types.CreateCompletionResponse
    response_certainty: float
    """

    explanation: Dict
    response_certainty: float


class StagedResponse(Response):
    """Model of data returned by Log Detective API when called when staged response
    is requested. Contains list of reponses to prompts for individual snippets.

    explanation: CreateCompletionResponse
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama_types.CreateCompletionResponse
    response_certainty: float
    snippets:
        list of dictionaries {
        'snippet' : '<original_text>,
        'comment': CreateCompletionResponse,
        'line_number': '<location_in_log>' }
    """

    snippets: List[Dict[str, str | Dict | int]]


class InferenceConfig(BaseModel):
    """Model for inference configuration of logdetective server."""

    max_tokens: int = -1
    log_probs: int = 1

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.max_tokens = data.get("max_tokens", -1)
        self.log_probs = data.get("log_probs", 1)


class ExtractorConfig(BaseModel):
    """Model for extractor configuration of logdetective server."""

    context: bool = True
    max_clusters: int = 8
    verbose: bool = False

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.context = data.get("context", True)
        self.max_clusters = data.get("max_clusters", 8)
        self.verbose = data.get("verbose", False)


class GitLabConfig(BaseModel):
    """Model for GitLab configuration of logdetective server."""

    url: str = None
    api_token: str = None

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.url = data.get("url", "https://gitlab.com")
        self.api_token = data.get("api_token", None)


class Config(BaseModel):
    """Model for configuration of logdetective server."""

    inference: InferenceConfig = InferenceConfig()
    extractor: ExtractorConfig = ExtractorConfig()
    gitlab: GitLabConfig = GitLabConfig()

    def __init__(self, data: Optional[dict] = None):
        super().__init__()

        if data is None:
            return

        self.inference = InferenceConfig(data.get("inference"))
        self.extractor = ExtractorConfig(data.get("extractor"))
        self.gitlab = GitLabConfig(data.get("gitlab"))
