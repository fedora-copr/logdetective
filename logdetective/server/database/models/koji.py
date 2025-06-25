import enum
from sqlalchemy import Column, BigInteger, Enum, ForeignKey, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.exc import OperationalError
import backoff
from logdetective.server.config import SERVER_CONFIG

from logdetective.server.compressors import LLMResponseCompressor
from logdetective.server.database.models.metrics import AnalyzeRequestMetrics
from logdetective.server.database.base import Base, transaction, DB_MAX_RETRIES


class TaskNotFoundError(Exception):
    """Exception raised when a task is not found"""


class TaskNotAnalyzedError(Exception):
    """Exception raised when a task analysis is still in progress"""


class KojihubInstance(str, enum.Enum):
    """List of koji instances recognized by logdetective"""

    # These identifiers match those provided by the fedora-packager,
    # centos-packager and redhat-packager RPMs in /etc/koji.conf.d
    koji_instance = Column(String(255), nullable=False, index=True)
    cbs = "https://cbs.centos.org/kojihub/"  # pylint: disable=(invalid-name)
    fedora = "https://koji.fedoraproject.org/kojihub"  # pylint: disable=(invalid-name)
    s390 = "https://s390.koji.fedoraproject.org/kojihub"  # pylint: disable=(invalid-name)
    stg = "https://koji.stg.fedoraproject.org/kojihub"  # pylint: disable=(invalid-name)
    stream = "https://kojihub.stream.rdu2.redhat.com/kojihub"  # pylint: disable=(invalid-name)


class KojiTaskAnalysis(Base):
    """Store details for the koji task analysis"""

    __tablename__ = "koji_task_analysis"

    id = Column(Integer, primary_key=True)
    koji_instance = Column(String(255), nullable=False, index=True)
    task_id = Column(BigInteger, nullable=False, index=True, unique=True)
    response_id = Column(
        Integer,
        ForeignKey("analyze_request_metrics.id"),
        nullable=True,
        index=False,
        comment="The id of the analyze request metrics for this task",
    )
    response = relationship("AnalyzeRequestMetrics")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def create(cls, koji_instance: KojihubInstance, task_id: int):
        """Create a new koji task analysis"""
        with transaction(commit=True) as session:
            koji_task_analysis = KojiTaskAnalysis()
            koji_task_analysis.koji_instance = koji_instance
            koji_task_analysis.task_id = task_id
            session.add(koji_task_analysis)
            session.flush()

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def add_response(cls, task_id: int, metric_id: int):
        """Add a response to a koji task analysis"""
        with transaction(commit=True) as session:
            koji_task_analysis = session.query(cls).filter_by(task_id=task_id).first()
            metric = (
                session.query(AnalyzeRequestMetrics).filter_by(id=metric_id).first()
            )
            koji_task_analysis.response = metric
            session.add(koji_task_analysis)
            session.flush()

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def get_response_by_task_id(cls, task_id: int):
        """Get a koji task analysis by task id"""
        with transaction(commit=False) as session:
            koji_task_analysis = session.query(cls).filter_by(task_id=task_id).first()
            if not koji_task_analysis:
                raise TaskNotFoundError(f"Task {task_id} not yet analyzed")

            if not koji_task_analysis.response:
                raise TaskNotAnalyzedError(
                    f"Task {task_id} analysis is still in progress"
                )

            # We need to decompress the response message and return it
            return LLMResponseCompressor.unzip(
                koji_task_analysis.response.compressed_response
            )
