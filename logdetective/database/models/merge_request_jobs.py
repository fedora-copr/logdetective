from __future__ import annotations
import enum
import datetime
from typing import Optional, List, Self, TYPE_CHECKING

from sqlalchemy import (
    Enum,
    BigInteger,
    DateTime,
    String,
    ForeignKey,
    UniqueConstraint,
    desc,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from logdetective.database.base import Base, transaction
from logdetective.utils import retry_database_error


if TYPE_CHECKING:
    from .metrics import AnalyzeRequestMetrics


class Forge(str, enum.Enum):
    """List of forges managed by logdetective"""

    gitlab_com = "https://gitlab.com"  # pylint: disable=(invalid-name)
    gitlab_cee_redhat_com = "http://gitlab.cee.redhat.com/"  # pylint: disable=(invalid-name)


class GitlabMergeRequestJobs(Base):
    """Store details for the merge request jobs
    which triggered logdetective.
    """

    __tablename__ = "gitlab_merge_request_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    forge: Mapped[Forge] = mapped_column(
        Enum(Forge),
        nullable=False,
        index=True,
        comment="The forge name"
    )
    project_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="The project gitlab id",
    )
    mr_iid: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=False,
        comment="The merge request gitlab iid",
    )
    job_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="The job gitlab id",
    )

    __table_args__ = (
        UniqueConstraint("forge", "job_id", name="uix_forge_job"),
        UniqueConstraint(
            "forge", "project_id", "mr_iid", "job_id", name="uix_mr_project_job"
        ),
    )

    comment: Mapped[List["Comments"]] = relationship(
        "Comments", back_populates="merge_request_job", uselist=False
    )  # 1 comment for 1 job

    request_metrics: Mapped[List["AnalyzeRequestMetrics"]] = relationship(
        "AnalyzeRequestMetrics",
        back_populates="mr_job"
    )

    @classmethod
    @retry_database_error
    async def create(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
    ) -> int:
        """Create a new merge request job entry,
        returns its PostgreSQL id

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
        """
        async with transaction(commit=True) as session:
            mr = cls()
            mr.forge = forge
            mr.project_id = project_id
            mr.mr_iid = mr_iid
            mr.job_id = job_id
            session.add(mr)
            await session.flush()
            return mr.id

    @classmethod
    async def get_by_id(
        cls,
        id_: int,
    ) -> Optional["GitlabMergeRequestJobs"]:
        """Search for a given PostgreSQL id"""
        query = select(cls).where(cls.id == id_)
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            mr = query_result.scalars().first()
            return mr

    @classmethod
    async def get_by_details(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
    ) -> Optional["GitlabMergeRequestJobs"]:
        """Search for a detailed merge request.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
        """
        query = select(cls).where(
            cls.forge == forge,
            cls.project_id == project_id,
            cls.mr_iid == mr_iid,
            cls.job_id == job_id,
        )
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            mr = query_result.scalars().first()
            return mr

    @classmethod
    async def get_by_mr_iid(cls, forge: Forge, project_id: int, mr_iid) -> List[Self]:
        """Get all the mr jobs saved for the specified mr iid and project id."""
        query = select(cls).where(
            GitlabMergeRequestJobs.forge == forge,
            GitlabMergeRequestJobs.project_id == project_id,
            GitlabMergeRequestJobs.mr_iid == mr_iid,
        )

        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            comments = query_result.scalars().all()

            return comments

    @classmethod
    async def get_or_create(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
    ) -> Self:
        """Search for a detailed merge request
        or create a new one if not found.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
        """
        mr = await GitlabMergeRequestJobs.get_by_details(
            forge, project_id, mr_iid, job_id
        )
        if mr is None:
            id_ = await GitlabMergeRequestJobs.create(forge, project_id, mr_iid, job_id)
            mr = await GitlabMergeRequestJobs.get_by_id(id_)
        return mr


class Comments(Base):
    """Store details for comments
    created by logdetective for a merge request job."""

    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    merge_request_job_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("gitlab_merge_request_jobs.id"),
        nullable=False,
        unique=True,  # 1 comment for 1 job
        index=True,
        comment="The associated merge request job (db) id",
    )
    forge: Mapped[Forge] = mapped_column(
        Enum(Forge),
        nullable=False,
        index=True,
        comment="The forge name"
    )
    comment_id: Mapped[str] = mapped_column(
        String(50),  # e.g. 'd5a3ff139356ce33e37e73add446f16869741b50'
        nullable=False,
        index=True,
        comment="The comment gitlab id",
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Timestamp when the comment was created",
    )

    __table_args__ = (
        UniqueConstraint("forge", "comment_id", name="uix_forge_comment_id"),
    )

    merge_request_job: Mapped["GitlabMergeRequestJobs"] = relationship(
        "GitlabMergeRequestJobs",
        back_populates="comment"
    )

    @classmethod
    @retry_database_error
    async def create(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: str,
        metrics: Optional["AnalyzeRequestMetrics"] = None,  # noqa: F821
    ) -> int:
        """Create a new comment id entry,
        returns its PostgreSQL id.
        A Gitlab comment id is unique within a
        gitlab instance.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
          comment_id: forge comment id
        """
        async with transaction(commit=True) as session:
            mr_job = await GitlabMergeRequestJobs.get_or_create(
                forge, project_id, mr_iid, job_id
            )

            if metrics:
                metrics.mr_job = mr_job
                session.add(metrics)

            comment = cls()
            comment.forge = forge
            comment.comment_id = comment_id
            comment.created_at = datetime.datetime.now(datetime.timezone.utc)
            comment.merge_request_job_id = mr_job.id
            session.add(comment)
            await session.flush()
            return comment.id

    @classmethod
    async def get_by_id(
        cls,
        id_: int,
    ) -> Optional["Comments"]:
        """Search for a given PostgreSQL id"""
        query = select(cls).where(cls.id == id_)
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            comment = query_result.scalars().first()
            return comment

    @classmethod
    async def get_by_gitlab_id(
        cls,
        forge: Forge,
        comment_id: str,
    ) -> Optional[Self]:
        """Search for a detailed comment
        by its unique forge comment id.

        Args:
          forge: forge name
          comment_id: forge comment id
        """
        query = (
            select(cls)
            .join(
                GitlabMergeRequestJobs,
                cls.merge_request_job_id == GitlabMergeRequestJobs.id,
            )
            .filter(GitlabMergeRequestJobs.forge == forge, cls.comment_id == comment_id)
        )
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            comment = query_result.scalars().first()
            return comment

    @classmethod
    async def get_latest_comment(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
    ) -> Optional["Comments"]:
        """Search for the latest comment in the merge request.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
        """
        query = (
            select(cls)
            .join(
                GitlabMergeRequestJobs,
                cls.merge_request_job_id == GitlabMergeRequestJobs.id,
            )
            .filter(
                GitlabMergeRequestJobs.forge == forge,
                GitlabMergeRequestJobs.project_id == project_id,
                GitlabMergeRequestJobs.mr_iid == mr_iid,
            )
            .order_by(desc(cls.created_at))
        )
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            comment = query_result.scalars().first()
            return comment

    @classmethod
    async def get_mr_comments(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
    ) -> List[Self]:
        """Search for all merge request comments.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
        """
        query = (
            select(cls)
            .join(
                GitlabMergeRequestJobs,
                cls.merge_request_job_id == GitlabMergeRequestJobs.id,
            )
            .filter(
                GitlabMergeRequestJobs.forge == forge,
                GitlabMergeRequestJobs.project_id == project_id,
                GitlabMergeRequestJobs.mr_iid == mr_iid,
            )
            .order_by(desc(cls.created_at))
        )
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            comments = query_result.scalars().all()
            return comments

    @classmethod
    async def get_or_create(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: str,
    ) -> Self:
        """Search for a detailed comment
        or create a new one if not found.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
          comment_id: forge comment id
        """
        comment = await Comments.get_by_gitlab_id(forge, comment_id)
        if comment is None:
            id_ = await Comments.create(forge, project_id, mr_iid, job_id, comment_id)
            comment = await Comments.get_by_id(id_)
        return comment

    @classmethod
    async def get_since(cls, time: datetime.datetime) -> List[Self]:
        """Get all the comments created after the given time."""
        query = select(cls).filter(Comments.created_at > time)
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            comments = query_result.scalars().all()

            return comments

    @classmethod
    async def get_by_mr_job(
        cls, merge_request_job: GitlabMergeRequestJobs
    ) -> Optional["Comments"]:
        """Get the comment added for the specified merge request's job."""
        query = select(cls).filter(Comments.merge_request_job == merge_request_job)
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            comments = query_result.scalars().first()
            return comments
