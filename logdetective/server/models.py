from typing import List, Dict, Optional, Literal
from pydantic import BaseModel


class BuildLog(BaseModel):
    """Model of data submitted to API."""

    url: str


class Explanation(BaseModel):
    """Model of snippet or general log explanation from Log Detective
    """

    text: str
    logprobs: Optional[List[Dict]] = None


class Response(BaseModel):
    """Model of data returned by Log Detective API

    explanation: Explanation
    response_certainty: float
    """

    explanation: Explanation
    response_certainty: float


class StagedResponse(Response):
    """Model of data returned by Log Detective API when called when staged response
    is requested. Contains list of reponses to prompts for individual snippets.

    explanation: Explanation
    response_certainty: float
    snippets:
        list of dictionaries { 'snippet' : '<original_text>, 'comment': CreateCompletionResponse }
    """

    snippets: List[Dict[str, str | Explanation]]


class InferenceConfig(BaseModel):
    """Model for inference configuration of logdetective server."""

    max_tokens: int = -1
    log_probs: int = 1
    api_endpoint: Optional[Literal["/chat/completions", "/completions"]] = "/chat/completions"

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.max_tokens = data.get("max_tokens", -1)
        self.log_probs = data.get("log_probs", 1)
        self.api_endpoint = data.get("api_endpoint", "/chat/completions")


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


class Config(BaseModel):
    """Model for configuration of logdetective server."""

    inference: InferenceConfig = InferenceConfig()
    extractor: ExtractorConfig = ExtractorConfig()

    def __init__(self, data: Optional[dict] = None):
        super().__init__()

        if data is None:
            return

        self.inference = InferenceConfig(data.get("inference"))
        self.extractor = ExtractorConfig(data.get("extractor"))
