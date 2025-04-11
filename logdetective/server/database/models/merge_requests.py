import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    DateTime,
    String,
    ForeignKey,
    UniqueConstraint,
    desc,
)
from sqlalchemy.orm import relationship
from logdetective.server.database.base import Base, transaction


class MergeRequests(Base):
    """Store details for the merge requests
    which triggered logdetective.
    """

    __tablename__ = "merge_requests"

    id = Column(Integer, primary_key=True)
    mr_id = Column(
        Integer,
        nullable=False,
        index=True,
        comment="The merge request id",
    )
    project_id = Column(
        Integer,
        nullable=False,
        index=True,
        comment="The project id",
    )
    job_id = Column(
        Integer,
        nullable=False,
        index=True,
        comment="The job id",
    )

    __table_args__ = (
        UniqueConstraint("mr_id", "project_id", "job_id", name="uix_mr_project_job"),
    )

    comment = relationship(
        "Comments", back_populates="merge_request", uselist=False
    )  # 1 comment for 1 job

    request_metrics = relationship("AnalyzeRequestMetrics", back_populates="mr")

    @classmethod
    def create(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
    ) -> int:
        """Create a new merge request job entry,
        returns its PostgreSQL id"""
        with transaction(commit=True) as session:
            mr = cls()
            mr.mr_id = mr_id
            mr.project_id = project_id
            mr.job_id = job_id
            session.add(mr)
            session.flush()
            return mr.id

    @classmethod
    def get_by_id(
        cls,
        id_: int,
    ) -> Optional["MergeRequests"]:
        """Search for a given PostgreSQL id"""
        with transaction(commit=False) as session:
            mr = session.query(cls).filter_by(id=id_).first()
            return mr

    @classmethod
    def get_by_details(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
    ) -> Optional["MergeRequests"]:
        """Search for a detailed merge request.

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
        """
        with transaction(commit=False) as session:
            mr = (
                session.query(cls)
                .filter_by(mr_id=mr_id, project_id=project_id, job_id=job_id)
                .first()
            )
            return mr

    @classmethod
    def get_or_create(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
    ) -> Optional["MergeRequests"]:
        """Search for a detailed merge request
        or create a new one if not found.

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
        """
        mr = MergeRequests.get_by_details(mr_id, project_id, job_id)
        if mr is None:
            id_ = MergeRequests.create(mr_id, project_id, job_id)
            mr = MergeRequests.get_by_id(id_)
        return mr


class Comments(Base):
    """Store details for comments
    created by logdetective in merge requests."""

    __tablename__ = "comments"

    id = Column(Integer, primary_key=True)
    merge_request_id = Column(
        Integer,
        ForeignKey("merge_requests.id"),
        nullable=False,
        unique=True,  # 1 comment for 1 job
        index=True,
        comment="The associated merge request id",
    )
    comment_id = Column(
        Integer,
        nullable=False,
        index=True,
        comment="The comment id",
    )
    created_at = Column(
        DateTime, nullable=False, comment="Timestamp when the comment was created"
    )

    merge_request = relationship("MergeRequests", back_populates="comment")
    reaction = relationship("Reactions", back_populates="comment")

    @classmethod
    def create(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
        comment_id: int,
    ) -> int:
        """Create a new comment id entry,
        returns its PostgreSQL id.

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
          comment_id: forge comment id
        """
        mr = MergeRequests.get_or_create(mr_id, project_id, job_id)
        with transaction(commit=True) as session:
            comment = cls()
            comment.comment_id = comment_id
            comment.created_at = datetime.datetime.now(datetime.timezone.utc)
            comment.merge_request_id = mr.id
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
    def get_by_details(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
        comment_id: int,
    ) -> Optional["Comments"]:
        """Search for a detailed comment.

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
          comment_id: forge comment id
        """
        with transaction(commit=False) as session:
            comment = (
                session.query(cls)
                .join(MergeRequests, cls.merge_request_id == MergeRequests.id)
                .filter(
                    MergeRequests.mr_id == mr_id,
                    MergeRequests.project_id == project_id,
                    MergeRequests.job_id == job_id,
                    cls.comment_id == comment_id,
                )
                .first()
            )

            return comment

    @classmethod
    def get_latest_comment(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
    ) -> Optional["Comments"]:
        """Search for the latest comment in the merge request.

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
        """
        with transaction(commit=False) as session:
            comment = (
                session.query(cls)
                .join(MergeRequests, cls.merge_request_id == MergeRequests.id)
                .filter(
                    MergeRequests.mr_id == mr_id,
                    MergeRequests.project_id == project_id,
                    MergeRequests.job_id == job_id,
                )
                .order_by(desc(cls.created_at))
                .first()
            )

            return comment

    @classmethod
    def get_or_create(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
        comment_id: int,
    ) -> Optional["Comments"]:
        """Search for a detailed comment
        or create a new one if not found.

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
          comment_id: forge comment id
        """
        comment = Comments.get_by_details(mr_id, project_id, job_id, comment_id)
        if comment is None:
            id_ = Comments.create(mr_id, project_id, job_id, comment_id)
            comment = MergeRequests.get_by_id(id_)
        return comment


class Reactions(Base):
    """Store and count reactions received for
    logdetective comments"""

    __tablename__ = "reactions"

    id = Column(Integer, primary_key=True)
    comment_id = Column(
        Integer,
        ForeignKey("comments.id"),
        nullable=False,
        index=True,
        comment="The associated comment id",
    )
    reaction_type = Column(
        String(50),  # e.g. 'thumbs-up'
        nullable=False,
        comment="The type of reaction",
    )

    comment = relationship("Comments", back_populates="reaction")

    @classmethod
    def add(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
        comment_id: int,
        reaction_type: str,
    ) -> int:
        """Add a new reaction for a specified comment.

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
          comment_id: forge comment id
          reaction_type: a str, ex. thumb_up
        """
        comment = Comments.get_or_create(mr_id, project_id, job_id, comment_id)
        with transaction(commit=True) as session:
            reaction = cls()
            reaction.comment_id = comment.id
            reaction.reaction_type = reaction_type
            session.add(reaction)
            session.flush()
            return reaction.id

    @classmethod
    def get_all_reactions(
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
        comment_id: int,
    ) -> int:
        """Count all reactions for a comment

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
          comment_id: forge comment id
        """
        with transaction(commit=False) as session:
            reactions = (
                session.query(cls)
                .join(Comments, cls.comment_id == Comments.id)
                .join(MergeRequests, Comments.merge_request_id == MergeRequests.id)
                .filter(
                    Comments.comment_id == comment_id,
                    MergeRequests.mr_id == mr_id,
                    MergeRequests.project_id == project_id,
                    MergeRequests.job_id == job_id,
                )
                .all()
            )

            return reactions

    @classmethod
    def get_reactions_by_type(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        mr_id: int,
        project_id: int,
        job_id: int,
        comment_id: int,
        reaction_type: str,
    ) -> int:
        """Count all reactions for a comment
        of a given type

        Args:
          mr_id: forge id
          project_id: forge project id
          job_id: forge job id
          comment_id: forge comment id
          reaction_type: str like "thumb-up"
        """
        with transaction(commit=False) as session:
            reactions = (
                session.query(cls)
                .join(Comments, cls.comment_id == Comments.id)
                .join(MergeRequests, Comments.merge_request_id == MergeRequests.id)
                .filter(
                    Comments.comment_id == comment_id,
                    MergeRequests.mr_id == mr_id,
                    MergeRequests.project_id == project_id,
                    MergeRequests.job_id == job_id,
                    Reactions.reaction_type == reaction_type,
                )
                .all()
            )

            return reactions
