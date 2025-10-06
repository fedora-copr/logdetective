import asyncio
from collections import defaultdict
import datetime
from logging import BASIC_FORMAT
from typing import List, Dict, Optional
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    NonNegativeFloat,
    HttpUrl,
)

import aiohttp

from aiolimiter import AsyncLimiter
from gitlab import Gitlab
import koji

from logdetective.constants import (
    DEFAULT_TEMPERATURE,
    LLM_DEFAULT_MAX_QUEUE_SIZE,
    LLM_DEFAULT_REQUESTS_PER_MINUTE,
    SYSTEM_ROLE_DEFAULT,
    USER_ROLE_DEFAULT,
)

from logdetective.extractors import Extractor, DrainExtractor, CSGrepExtractor
from logdetective.utils import check_csgrep


class BuildLog(BaseModel):
    """Model of data submitted to API."""

    url: str


class JobHook(BaseModel):
    """Model of Job Hook events sent from GitLab.
    Full details of the specification are available at
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    This model implements only the fields that we care about. The webhook
    sends many more fields that we will ignore."""

    # The unique job ID on this GitLab instance.
    build_id: int

    # The identifier of the job. We only care about 'build_rpm' and
    # 'build_centos_stream_rpm' jobs.
    build_name: str = Field(pattern=r"^build.*rpm$")

    # A string representing the job status. We only care about 'failed' jobs.
    build_status: str = Field(pattern=r"^failed$")

    # The kind of webhook message. We are only interested in 'build' messages
    # which represents job tasks in a pipeline.
    object_kind: str = Field(pattern=r"^build$")

    # The unique ID of the enclosing pipeline on this GitLab instance.
    pipeline_id: int

    # The unique ID of the project triggering this event
    project_id: int


class EmojiMergeRequest(BaseModel):
    """Model of the 'merge_request' subsection of Emoji webhook messages.
    This model implements only the fields that we care about. The webhook
    sends many more fields that we will ignore."""

    # The identifier of the target project
    target_project_id: int

    # The internal identifier (relative to the target project)
    iid: int


class EmojiHook(BaseModel):
    """Model of Job Hook events sent from GitLab.
    Full details of the specification are available at
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    This model implements only the fields that we care about. The webhook
    sends many more fields that we will ignore."""

    # The kind of webhook message. We are only interested in 'emoji' messages
    # which represents awarding or revoking emoji reactions on notes.
    object_kind: str = Field(pattern=r"^emoji$")

    # Information about the merge request this emoji applies to, if any.
    merge_request: EmojiMergeRequest = Field(default=None)


class SnippetAnalysis(BaseModel):
    """Model of snippet analysis from LLM."""

    text: str = Field(description="Analysis of log snippet contents.")


class RatedSnippetAnalysis(SnippetAnalysis):
    """Model for rated snippet analysis. This model is used to generate
    json schema for inference with structured output."""

    relevance: int = Field(
        ge=0,
        le=100,
        description="Estimate of likelyhood that snippet contains an error, "
        "with 0 standing for completely unlikely, 100 for absolutely certain.",
    )


class Explanation(BaseModel):
    """Model of snippet or general log explanation from Log Detective"""

    text: str
    logprobs: Optional[List[Dict]] = None

    def __str__(self):
        """Return text of the Explanation"""
        return self.text


class AnalyzedSnippet(BaseModel):
    """Model for snippets already processed by Log Detective.

    explanation: LLM output in form of plain text and logprobs dictionary
    text: original snippet text
    line_number: location of snippet in original log
    """

    explanation: SnippetAnalysis | RatedSnippetAnalysis
    text: str
    line_number: int


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
    snippets: list of AnalyzedSnippet objects
    """

    snippets: List[AnalyzedSnippet]


class KojiStagedResponse(BaseModel):
    """Model of data returned by Log Detective API when called when a Koji build
    analysis is requested. Contains list of reponses to prompts for individual
    snippets.
    """

    task_id: int
    log_file_name: str
    response: StagedResponse


class InferenceConfig(BaseModel):  # pylint: disable=too-many-instance-attributes
    """Model for inference configuration of logdetective server."""

    max_tokens: int = -1
    log_probs: bool = True
    url: str = ""
    # OpenAI client library requires a string to be specified for API token
    # even if it is not checked on the server side
    api_token: str = "None"
    model: str = ""
    temperature: NonNegativeFloat = DEFAULT_TEMPERATURE
    max_queue_size: int = LLM_DEFAULT_MAX_QUEUE_SIZE
    http_timeout: float = 5.0
    user_role: str = USER_ROLE_DEFAULT
    system_role: str = SYSTEM_ROLE_DEFAULT
    _http_session: aiohttp.ClientSession = None
    _limiter: AsyncLimiter = AsyncLimiter(LLM_DEFAULT_REQUESTS_PER_MINUTE)

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.max_tokens = data.get("max_tokens", -1)
        self.log_probs = data.get("log_probs", True)
        self.url = data.get("url", "")
        self.http_timeout = data.get("http_timeout", 5.0)
        self.api_token = data.get("api_token", "None")
        self.model = data.get("model", "default-model")
        self.temperature = data.get("temperature", DEFAULT_TEMPERATURE)
        self.max_queue_size = data.get("max_queue_size", LLM_DEFAULT_MAX_QUEUE_SIZE)
        self.user_role = data.get("user_role", USER_ROLE_DEFAULT)
        self.system_role = data.get("system_role", SYSTEM_ROLE_DEFAULT)
        self._requests_per_minute = data.get(
            "requests_per_minute", LLM_DEFAULT_REQUESTS_PER_MINUTE
        )
        self._limiter = AsyncLimiter(self._requests_per_minute)

    def __del__(self):
        # Close connection when this object is destroyed
        if self._http_session:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._http_session.close())
            except RuntimeError:
                # No loop running, so create one to close the session
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._http_session.close())
                loop.close()
            except Exception:  # pylint: disable=broad-exception-caught
                # We should only get here if we're shutting down, so we don't
                # really care if the close() completes cleanly.
                pass

    def get_http_session(self):
        """Return the internal HTTP session so it can be used to contect the
        LLM server. May be used as a context manager."""

        # Create the session on the first attempt. We need to do this "lazily"
        # because it needs to happen once the event loop is running, even
        # though the initialization itself is synchronous.
        if not self._http_session:
            self._http_session = aiohttp.ClientSession(
                base_url=self.url,
                timeout=aiohttp.ClientTimeout(
                    total=self.http_timeout,
                    connect=3.07,
                ),
            )

        return self._http_session

    def get_limiter(self):
        """Return the limiter object so it can be used as a context manager"""
        return self._limiter


class ExtractorConfig(BaseModel):
    """Model for extractor configuration of logdetective server."""

    max_clusters: int = 8
    verbose: bool = False
    max_snippet_len: int = 2000
    csgrep: bool = False

    _extractors: List[Extractor] = []

    def _setup_extractors(self):
        """Initialize extractors with common settings."""
        self._extractors = [
            DrainExtractor(
                verbose=self.verbose,
                max_snippet_len=self.max_snippet_len,
                max_clusters=self.max_clusters,
            )
        ]

        if self.csgrep:
            self._extractors.append(
                CSGrepExtractor(
                    verbose=self.verbose,
                    max_snippet_len=self.max_snippet_len,
                )
            )

    def __init__(self, data: Optional[dict] = None):
        super().__init__(data=data)

        if data is None:
            self._setup_extractors()
            return

        self.max_clusters = data.get("max_clusters", 8)
        self.verbose = data.get("verbose", False)
        self.max_snippet_len = data.get("max_snippet_len", 2000)
        self.csgrep = data.get("csgrep", False)

        self._setup_extractors()

    def get_extractors(self) -> List[Extractor]:
        """Return list of initialized extractors, each will be applied in turn
        on original log text to retrieve snippets."""
        return self._extractors

    @field_validator("csgrep", mode="after")
    @classmethod
    def validate_csgrep(cls, value: bool) -> bool:
        """Verify that csgrep is available if requested."""
        if not check_csgrep():
            raise ValueError(
                "Requested csgrep extractor but `csgrep` binary is not in the PATH"
            )
        return value


class GitLabInstanceConfig(BaseModel):  # pylint: disable=too-many-instance-attributes
    """Model for GitLab configuration of logdetective server."""

    name: str = None
    url: str = None
    # Path to API of the gitlab instance, assuming `url` as prefix.
    api_path: str = None
    api_token: str = None

    # This is a list to support key rotation.
    # When the key is being changed, we will add the new key as a new entry in
    # the configuration and then remove the old key once all of the client
    # webhook configurations have been updated.
    # If this option is left empty or unspecified, all requests will be
    # considered authorized.
    webhook_secrets: Optional[List[str]] = None

    timeout: float = 5.0
    _conn: Gitlab = None
    _http_session: aiohttp.ClientSession = None

    # Maximum size of artifacts.zip in MiB. (default: 300 MiB)
    max_artifact_size: int = 300 * 1024 * 1024

    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.name = name
        self.url = data.get("url", "https://gitlab.com")
        self.api_path = data.get("api_path", "/api/v4")
        self.api_token = data.get("api_token", None)
        self.webhook_secrets = data.get("webhook_secrets", None)
        self.max_artifact_size = int(data.get("max_artifact_size", 300)) * 1024 * 1024

        self.timeout = data.get("timeout", 5.0)
        self._conn = Gitlab(
            url=self.url,
            private_token=self.api_token,
            timeout=self.timeout,
        )

    def get_connection(self):
        """Get the Gitlab connection object"""
        return self._conn

    def get_http_session(self):
        """Return the internal HTTP session so it can be used to contect the
        Gitlab server. May be used as a context manager."""

        # Create the session on the first attempt. We need to do this "lazily"
        # because it needs to happen once the event loop is running, even
        # though the initialization itself is synchronous.
        if not self._http_session:
            self._http_session = aiohttp.ClientSession(
                base_url=self.url,
                headers={"Authorization": f"Bearer {self.api_token}"},
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout,
                    connect=3.07,
                ),
            )

        return self._http_session

    def __del__(self):
        # Close connection when this object is destroyed
        if self._http_session:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._http_session.close())
            except RuntimeError:
                # No loop running, so create one to close the session
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._http_session.close())
                loop.close()
            except Exception:  # pylint: disable=broad-exception-caught
                # We should only get here if we're shutting down, so we don't
                # really care if the close() completes cleanly.
                pass


class GitLabConfig(BaseModel):
    """Model for GitLab configuration of logdetective server."""

    instances: Dict[str, GitLabInstanceConfig] = {}

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        for instance_name, instance_data in data.items():
            instance = GitLabInstanceConfig(instance_name, instance_data)
            self.instances[instance.url] = instance


class KojiInstanceConfig(BaseModel):
    """Model for Koji configuration of logdetective server."""

    name: str = ""
    xmlrpc_url: str = ""
    tokens: List[str] = []

    _conn: Optional[koji.ClientSession] = None
    _callbacks: defaultdict[int, set[str]] = defaultdict(set)

    def __init__(self, name: str, data: Optional[dict] = None):
        super().__init__()

        self.name = name
        if data is None:
            # Set some reasonable defaults
            self.xmlrpc_url = "https://koji.fedoraproject.org/kojihub"
            self.tokens = []
            self.max_artifact_size = 1024 * 1024
            return

        self.xmlrpc_url = data.get(
            "xmlrpc_url", "https://koji.fedoraproject.org/kojihub"
        )
        self.tokens = data.get("tokens", [])

    def get_connection(self):
        """Get the Koji connection object"""
        if not self._conn:
            self._conn = koji.ClientSession(self.xmlrpc_url)
        return self._conn

    def register_callback(self, task_id: int, callback: str):
        """Register a callback for a task"""
        self._callbacks[task_id].add(callback)

    def clear_callbacks(self, task_id: int):
        """Unregister a callback for a task"""
        try:
            del self._callbacks[task_id]
        except KeyError:
            pass

    def get_callbacks(self, task_id: int) -> set[str]:
        """Get the callbacks for a task"""
        return self._callbacks[task_id]


class KojiConfig(BaseModel):
    """Model for Koji configuration of logdetective server."""

    instances: Dict[str, KojiInstanceConfig] = {}
    analysis_timeout: int = 15
    max_artifact_size: int = 300 * 1024 * 1024

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        # Handle analysis_timeout with default 15
        self.analysis_timeout = data.get("analysis_timeout", 15)

        # Handle max_artifact_size with default 300
        self.max_artifact_size = data.get("max_artifact_size", 300) * 1024 * 1024

        # Handle instances dictionary
        instances_data = data.get("instances", {})
        for instance_name, instance_data in instances_data.items():
            self.instances[instance_name] = KojiInstanceConfig(
                instance_name, instance_data
            )


class LogConfig(BaseModel):
    """Logging configuration"""

    name: str = "logdetective"
    level_stream: str | int = "INFO"
    level_file: str | int = "INFO"
    path: str | None = None
    format: str = BASIC_FORMAT

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.name = data.get("name", "logdetective")
        self.level_stream = data.get("level_stream", "INFO").upper()
        self.level_file = data.get("level_file", "INFO").upper()
        self.path = data.get("path")
        self.format = data.get("format", BASIC_FORMAT)


class GeneralConfig(BaseModel):
    """General config options for Log Detective"""

    packages: List[str] = None
    excluded_packages: List[str] = None
    devmode: bool = False
    sentry_dsn: HttpUrl | None = None
    collect_emojis_interval: int = 60 * 60  # seconds
    top_k_snippets: int = 0

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.packages = data.get("packages", [])
        self.excluded_packages = data.get("excluded_packages", [])
        self.devmode = data.get("devmode", False)
        self.sentry_dsn = data.get("sentry_dsn")
        self.collect_emojis_interval = data.get(
            "collect_emojis_interval", 60 * 60
        )  # seconds
        self.top_k_snippets = data.get("top_k_snippets", 0)


class Config(BaseModel):
    """Model for configuration of logdetective server."""

    log: LogConfig = LogConfig()
    inference: InferenceConfig = InferenceConfig()
    snippet_inference: InferenceConfig = InferenceConfig()
    # TODO(jpodivin): Extend to work with multiple extractor configs
    extractor: ExtractorConfig = ExtractorConfig()
    gitlab: GitLabConfig = GitLabConfig()
    koji: KojiConfig = KojiConfig()
    general: GeneralConfig = GeneralConfig()

    def __init__(self, data: Optional[dict] = None):
        super().__init__()

        if data is None:
            return

        self.log = LogConfig(data.get("log"))
        self.inference = InferenceConfig(data.get("inference"))
        self.extractor = ExtractorConfig(data.get("extractor"))
        self.gitlab = GitLabConfig(data.get("gitlab"))
        self.koji = KojiConfig(data.get("koji"))
        self.general = GeneralConfig(data.get("general"))

        if snippet_inference := data.get("snippet_inference", None):
            self.snippet_inference = InferenceConfig(snippet_inference)
        else:
            self.snippet_inference = self.inference


class TimePeriod(BaseModel):
    """Specification for a period of time.

    If no indication is given
    it falls back to a 2 days period of time.

    Can't be smaller than a hour"""

    weeks: Optional[int] = None
    days: Optional[int] = None
    hours: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def check_exclusive_fields(cls, data):
        """Check that only one key between weeks, days and hours is defined"""
        if isinstance(data, dict):
            how_many_fields = sum(
                1
                for field in ["weeks", "days", "hours"]
                if field in data and data[field] is not None
            )

            if how_many_fields == 0:
                data["days"] = 2  # by default fallback to a 2 days period

            if how_many_fields > 1:
                raise ValueError("Only one of months, weeks, days, or hours can be set")

        return data

    @field_validator("weeks", "days", "hours")
    @classmethod
    def check_positive(cls, v):
        """Check that the given value is positive"""
        if v is not None and v <= 0:
            raise ValueError("Time period must be positive")
        return v

    def get_time_period(self) -> datetime.timedelta:
        """Get the period of time represented by this input model.

        Returns:
            datetime.timedelta: The time period as a timedelta object.
        """
        delta = None
        if self.weeks:
            delta = datetime.timedelta(weeks=self.weeks)
        elif self.days:
            delta = datetime.timedelta(days=self.days)
        elif self.hours:
            delta = datetime.timedelta(hours=self.hours)
        return delta

    def get_period_start_time(
        self, end_time: datetime.datetime = None
    ) -> datetime.datetime:
        """Calculate the start time of this period based on the end time.

        Args:
            end_time (datetime.datetime, optional): The end time of the period.
                Defaults to current UTC time if not provided.

        Returns:
            datetime.datetime: The start time of the period.
        """
        time = end_time or datetime.datetime.now(datetime.timezone.utc)
        if time.tzinfo is None:
            end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        return time - self.get_time_period()
