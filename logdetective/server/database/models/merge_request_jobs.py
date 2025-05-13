import enum
import datetime
from typing import Optional, List, Tuple

import backoff

from sqlalchemy import (
    Enum,
    Column,
    BigInteger,
    DateTime,
    String,
    ForeignKey,
    UniqueConstraint,
    desc,
)
from sqlalchemy.orm import relationship
from sqlalchemy.exc import OperationalError
from logdetective.server.database.base import Base, transaction, DB_MAX_RETRIES


class Forge(str, enum.Enum):
    """List of forges managed by logdetective"""

    gitlab_com = "https://gitlab.com"  # pylint: disable=(invalid-name)
    gitlab_cee_redhat_com = "http://gitlab.cee.redhat.com/"  # pylint: disable=(invalid-name)


class GitlabMergeRequestJobs(Base):
    """Store details for the merge request jobs
    which triggered logdetective.
    """

    __tablename__ = "gitlab_merge_request_jobs"

    id = Column(BigInteger, primary_key=True)
    forge = Column(Enum(Forge), nullable=False, index=True, comment="The forge name")
    project_id = Column(
        BigInteger,
        nullable=False,
        index=True,
        comment="The project gitlab id",
    )
    mr_iid = Column(
        BigInteger,
        nullable=False,
        index=False,
        comment="The merge request gitlab iid",
    )
    job_id = Column(
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

    comment = relationship(
        "Comments", back_populates="merge_request_job", uselist=False
    )  # 1 comment for 1 job

    request_metrics = relationship("AnalyzeRequestMetrics", back_populates="mr_job")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def create(
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
        with transaction(commit=True) as session:
            mr = cls()
            mr.forge = forge
            mr.project_id = project_id
            mr.mr_iid = mr_iid
            mr.job_id = job_id
            session.add(mr)
            session.flush()
            return mr.id

    @classmethod
    def get_by_id(
        cls,
        id_: int,
    ) -> Optional["GitlabMergeRequestJobs"]:
        """Search for a given PostgreSQL id"""
        with transaction(commit=False) as session:
            mr = session.query(cls).filter_by(id=id_).first()
            return mr

    @classmethod
    def get_by_details(
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
        with transaction(commit=False) as session:
            mr = (
                session.query(cls)
                .filter_by(
                    forge=forge, project_id=project_id, mr_iid=mr_iid, job_id=job_id
                )
                .first()
            )
            return mr

    @classmethod
    def get_by_mr_iid(
        cls, forge: Forge, project_id: int, mr_iid
    ) -> Optional["GitlabMergeRequestJobs"]:
        """Get all the mr jobs saved for the specified mr iid and project id."""
        with transaction(commit=False) as session:
            comments = (
                session.query(cls)
                .filter(
                    GitlabMergeRequestJobs.forge == forge,
                    GitlabMergeRequestJobs.project_id == project_id,
                    GitlabMergeRequestJobs.mr_iid == mr_iid,
                )
                .all()
            )

            return comments

    @classmethod
    def get_or_create(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
    ) -> Optional["GitlabMergeRequestJobs"]:
        """Search for a detailed merge request
        or create a new one if not found.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
        """
        mr = GitlabMergeRequestJobs.get_by_details(forge, project_id, mr_iid, job_id)
        if mr is None:
            id_ = GitlabMergeRequestJobs.create(forge, project_id, mr_iid, job_id)
            mr = GitlabMergeRequestJobs.get_by_id(id_)
        return mr


class Comments(Base):
    """Store details for comments
    created by logdetective for a merge request job."""

    __tablename__ = "comments"

    id = Column(BigInteger, primary_key=True)
    merge_request_job_id = Column(
        BigInteger,
        ForeignKey("gitlab_merge_request_jobs.id"),
        nullable=False,
        unique=True,  # 1 comment for 1 job
        index=True,
        comment="The associated merge request job (db) id",
    )
    forge = Column(Enum(Forge), nullable=False, index=True, comment="The forge name")
    comment_id = Column(
        String(50),  # e.g. 'd5a3ff139356ce33e37e73add446f16869741b50'
        nullable=False,
        index=True,
        comment="The comment gitlab id",
    )
    created_at = Column(
        DateTime, nullable=False, comment="Timestamp when the comment was created"
    )

    __table_args__ = (
        UniqueConstraint("forge", "comment_id", name="uix_forge_comment_id"),
    )

    merge_request_job = relationship("GitlabMergeRequestJobs", back_populates="comment")
    reactions = relationship("Reactions", back_populates="comment")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def create(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
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
        with transaction(commit=True) as session:
            mr_job = GitlabMergeRequestJobs.get_or_create(
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
            session.flush()
            return comment.id

    @classmethod
    def get_by_id(
        cls,
        id_: int,
    ) -> Optional["Comments"]:
        """Search for a given PostgreSQL id"""
        with transaction(commit=False) as session:
            comment = session.query(cls).filter_by(id=id_).first()
            return comment

    @classmethod
    def get_by_gitlab_id(
        cls,
        forge: Forge,
        comment_id: str,
    ) -> Optional["Comments"]:
        """Search for a detailed comment
        by its unique forge comment id.

        Args:
          forge: forge name
          comment_id: forge comment id
        """
        with transaction(commit=False) as session:
            comment = (
                session.query(cls)
                .join(
                    GitlabMergeRequestJobs,
                    cls.merge_request_job_id == GitlabMergeRequestJobs.id,
                )
                .filter(
                    GitlabMergeRequestJobs.forge == forge,
                    cls.comment_id == comment_id,
                )
                .first()
            )

            return comment

    @classmethod
    def get_latest_comment(
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
        with transaction(commit=False) as session:
            comment = (
                session.query(cls)
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
                .first()
            )

            return comment

    @classmethod
    def get_mr_comments(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
    ) -> Optional["Comments"]:
        """Search for all merge request comments.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
        """
        with transaction(commit=False) as session:
            comments = (
                session.query(cls)
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
                .all()
            )

            return comments

    @classmethod
    def get_or_create(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: str,
    ) -> Optional["Comments"]:
        """Search for a detailed comment
        or create a new one if not found.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
          comment_id: forge comment id
        """
        comment = Comments.get_by_gitlab_id(forge, comment_id)
        if comment is None:
            id_ = Comments.create(forge, project_id, mr_iid, job_id, comment_id)
            comment = GitlabMergeRequestJobs.get_by_id(id_)
        return comment

    @classmethod
    def get_since(cls, time: datetime.datetime) -> Optional["Comments"]:
        """Get all the comments created after the given time."""
        with transaction(commit=False) as session:
            comments = (
                session.query(cls)
                .filter(
                    Comments.created_at > time,
                )
                .all()
            )

            return comments

    @classmethod
    def get_by_mr_job(
        cls, merge_request_job: GitlabMergeRequestJobs
    ) -> Optional["Comments"]:
        """Get the comment added for the specified merge request's job."""
        with transaction(commit=False) as session:
            comments = (
                session.query(cls)
                .filter(
                    Comments.merge_request_job == merge_request_job,
                )
                .first()  # just one
            )

            return comments


class Reactions(Base):
    """Store and count reactions received for
    logdetective comments"""

    __tablename__ = "reactions"

    id = Column(BigInteger, primary_key=True)
    comment_id = Column(
        BigInteger,
        ForeignKey("comments.id"),
        nullable=False,
        index=True,
        comment="The associated comment (db) id",
    )
    reaction_type = Column(
        String(127),  # e.g. 'thumbs-up'
        nullable=False,
        comment="The type of reaction",
    )
    count = Column(
        BigInteger,
        nullable=False,
        comment="The number of reactions, of this type, given in the comment",
    )

    __table_args__ = (
        UniqueConstraint("comment_id", "reaction_type", name="uix_comment_reaction"),
    )

    comment = relationship("Comments", back_populates="reactions")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def create_or_update(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: str,
        reaction_type: str,
        count: int,
    ) -> int:
        """Create or update the given reaction, and its associated count,
        for a specified comment.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
          comment_id: forge comment id
          reaction_type: a str, ex. thumb_up
          count: number of reactions, of this type, given in the comment
        """
        comment = Comments.get_or_create(forge, project_id, mr_iid, job_id, comment_id)
        reaction = cls.get_reaction_by_type(
            forge, project_id, mr_iid, job_id, comment_id, reaction_type
        )
        with transaction(commit=True) as session:
            if reaction:
                reaction.count = count  # just update
            else:
                reaction = cls()
                reaction.comment_id = comment.id
                reaction.reaction_type = reaction_type
                reaction.count = count
            session.add(reaction)
            session.flush()
            return reaction.id

    @classmethod
    def get_all_reactions(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: str,
    ) -> int:
        """Get all reactions for a comment

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
          comment_id: forge comment id
        """
        with transaction(commit=False) as session:
            reactions = (
                session.query(cls)
                .join(Comments, cls.comment_id == Comments.id)
                .join(
                    GitlabMergeRequestJobs,
                    Comments.merge_request_job_id == GitlabMergeRequestJobs.id,
                )
                .filter(
                    Comments.comment_id == comment_id,
                    GitlabMergeRequestJobs.forge == forge,
                    GitlabMergeRequestJobs.project_id == project_id,
                    GitlabMergeRequestJobs.mr_iid == mr_iid,
                    GitlabMergeRequestJobs.job_id == job_id,
                )
                .all()
            )

            return reactions

    @classmethod
    def get_reaction_by_type(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: str,
        reaction_type: str,
    ) -> int:
        """Get reaction, of a given type,
        for a comment

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
          comment_id: forge comment id
          reaction_type: str like "thumb-up"
        """
        with transaction(commit=False) as session:
            reaction = (
                session.query(cls)
                .join(Comments, cls.comment_id == Comments.id)
                .join(
                    GitlabMergeRequestJobs,
                    Comments.merge_request_job_id == GitlabMergeRequestJobs.id,
                )
                .filter(
                    Comments.comment_id == comment_id,
                    GitlabMergeRequestJobs.forge == forge,
                    GitlabMergeRequestJobs.project_id == project_id,
                    GitlabMergeRequestJobs.mr_iid == mr_iid,
                    GitlabMergeRequestJobs.job_id == job_id,
                    Reactions.reaction_type == reaction_type,
                )
                .first()
            )

            return reaction

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def delete(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
        comment_id: str,
        reaction_types: List[str],
    ) -> None:
        """Remove rows with given reaction types

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
          comment_id: forge comment id
          reaction_type: a str iterable, ex. ['thumbsup', 'thumbsdown']
        """
        for reaction_type in reaction_types:
            reaction = cls.get_reaction_by_type(
                forge, project_id, mr_iid, job_id, comment_id, reaction_type
            )
            with transaction(commit=True) as session:
                session.delete(reaction)
                session.flush()

    @classmethod
    def get_since(
        cls, time: datetime.datetime
    ) -> List[Tuple[datetime.datetime, "Comments"]]:
        """Get all the reactions on comments created after the given time
        and the comment creation time."""
        with transaction(commit=False) as session:
            reactions = (
                session.query(Comments.created_at, cls)
                .join(Comments, cls.comment_id == Comments.id)
                .filter(Comments.created_at > time)
                .all()
            )

            return reactions
