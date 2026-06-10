from typing import Sequence
import backoff
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    String,
    ForeignKey,
    select,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, joinedload
from sqlalchemy.exc import OperationalError
from logdetective.constants import EMBEDDING_VECTOR_SIZE
from logdetective.server.database.base import Base, transaction, DB_MAX_RETRIES


class AnnotatedSnippets(Base):
    """Store annotated snippet"""

    __tablename__ = "annotated_snippets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    text: Mapped[str] = mapped_column(String)
    annotation: Mapped[str] = mapped_column(String)
    text_embedding: Mapped[list] = mapped_column(VECTOR(EMBEDDING_VECTOR_SIZE))

    source_artifact_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("annotated_artifacts.id"),
        nullable=False,
        unique=False,  # Multiple snippets per artifact
        index=True,
        comment="Source artifact of the annotated snippet"
    )

    source_artifact: Mapped["AnnotatedArtifacts"] = relationship("AnnotatedArtifacts")

    __table_args__ = (
        # An approximate index for faster search
        # The Hierarchical Navigable Small Worlds algorithm may not find the exact closest match
        # but it is much faster than alternatives. The building takes some time and resources.
        # For our purposes, with < 100K records, the requirements shouldn't be too onerous.
        Index(
            "ix_annotated_snippets_text_embedding",
            "text_embedding",
            postgresql_with={  # Defaults from pgvector docs
                "m": 16,  # Number of connections to neighbors
                "ef_construction": 64  # Closest neighbors to keep during build
            },
            postgresql_using="hnsw",  # Search algorithm
            postgresql_ops={
                "text_embedding": "vector_l2_ops"
            }
        ),
    )

    @classmethod
    async def get_by_snippet_embedding(
        cls,
        embedding_vector: list[float],
        top_k: int = 5
    ) -> Sequence["AnnotatedSnippets"]:
        """Return closest matches of given embedding by l2 norm."""
        query = (
            select(cls)
            .options(
                joinedload(cls.source_artifact)
                .joinedload(AnnotatedArtifacts.annotated_build)
            )
            .order_by(cls.text_embedding.l2_distance(embedding_vector))
            .limit(top_k)
        )
        async with transaction(commit=False) as session:
            query_result = await session.execute(query)
            snippets = query_result.unique().scalars().all()

            return snippets

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    async def create(
        cls,
        text: str,
        annotation: str,
        text_embedding: list,
        source_artifact_id: int,
    ) -> int:
        """Create new annotated snippet with linked source artifact."""
        async with transaction(commit=True) as session:
            snippet = cls()
            snippet.text = text
            snippet.annotation = annotation
            snippet.source_artifact_id = source_artifact_id
            snippet.text_embedding = text_embedding
            session.add(snippet)
            await session.flush()
            return snippet.id


class AnnotatedArtifacts(Base):
    """Store annotated artifacts"""

    __tablename__ = "annotated_artifacts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    annotated_build_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("annotated_builds.id"),
        nullable=False,
        unique=False,  # Multiple artifacts per build
        index=True,
        comment="Source build of the artifact"
    )

    annotated_build: Mapped["AnnotatedBuilds"] = relationship("AnnotatedBuilds")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    async def create(cls, annotated_build_id: int) -> int:
        """Create annotated artifact with linked build."""
        async with transaction(commit=True) as session:
            annotated_artifact = cls()
            annotated_artifact.annotated_build_id = annotated_build_id
            session.add(annotated_artifact)
            await session.flush()
            return annotated_artifact.id


class AnnotatedBuilds(Base):
    """Store annotated build data"""

    __tablename__ = "annotated_builds"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    problem: Mapped[str] = mapped_column(String, comment="Full problem description")
    solution: Mapped[str] = mapped_column(String, comment="Full solution to the issue.")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    async def create(cls, problem: str, solution: str) -> int:
        """Create annotated build"""
        async with transaction(commit=True) as session:
            annotated_build = cls()
            annotated_build.problem = problem
            annotated_build.solution = solution
            session.add(annotated_build)
            await session.flush()
            return annotated_build.id
