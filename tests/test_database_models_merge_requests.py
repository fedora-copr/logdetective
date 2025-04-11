import pytest

from sqlalchemy.exc import IntegrityError

from test_helpers import (
    DatabaseFactory,
)

from logdetective.server.database.models import MergeRequests, Comments, Reactions


def test_create_and_get_MergeRequests():
    with DatabaseFactory().make_new_db() as _:
        id_ = MergeRequests.create(mr_id=123, project_id=456, job_id=11)
        assert id_ == 1
        with pytest.raises(IntegrityError):
            MergeRequests.create(mr_id=123, project_id=456, job_id=11)
        id_ = MergeRequests.create(mr_id=123, project_id=456, job_id=22)
        assert id_ == 2

        mr = MergeRequests.get_by_details(123, 456, 11)
        assert mr.id == 1

        mr = MergeRequests.get_by_details(123, 456, 1)
        assert mr is None

        mr = MergeRequests.get_by_id(2)
        assert mr.job_id == 22

        mr = MergeRequests.get_by_id(3)
        assert mr is None


def test_create_and_get_Comments():
    with DatabaseFactory().make_new_db() as _:
        mr_id = MergeRequests.create(mr_id=123, project_id=456, job_id=11)
        assert mr_id == 1
        comment_id = Comments.create(
            mr_id=123,
            project_id=456,
            job_id=11,
            comment_id=789,
        )
        assert comment_id == 1
        comment = Comments.get_by_id(comment_id)
        assert comment.merge_request_id == 1

        # create a new mr (implicitly) if it does not exist
        comment_id = Comments.create(
            mr_id=123,
            project_id=456,
            job_id=21,
            comment_id=7890,
        )
        assert comment_id == 2
        comment = Comments.get_by_id(comment_id)
        assert comment.merge_request_id == 2

        # no more than 1 comment for 1 job
        with pytest.raises(IntegrityError):
            Comments.create(
                mr_id=123,
                project_id=456,
                job_id=21,
                comment_id=7891,
            )

        comment = Comments.get_by_details(123, 456, 11, 789)
        assert comment.id == 1
        assert comment.merge_request_id == 1

        comment = Comments.get_latest_comment(123, 456, 21)
        assert comment.id == 2

        comment = Comments.get_or_create(
            mr_id=123,
            project_id=456,
            job_id=22,
            comment_id=7892,
        )
        assert comment.id == 3


def test_create_and_get_Reactions():
    with DatabaseFactory().make_new_db() as _:
        id_ = Reactions.add(
            mr_id=123,
            project_id=456,
            job_id=11,
            comment_id=789,
            reaction_type="thumb_up",
        )
        assert id_ == 1
        id_ = Reactions.add(
            mr_id=123,
            project_id=456,
            job_id=11,
            comment_id=789,
            reaction_type="thumb_down",
        )
        assert id_ == 2
        id_ = Reactions.add(
            mr_id=123,
            project_id=456,
            job_id=11,
            comment_id=789,
            reaction_type="thumb_down",
        )
        assert id_ == 3
        # reaction for another comment
        id_ = Reactions.add(
            mr_id=123,
            project_id=456,
            job_id=21,
            comment_id=7890,
            reaction_type="thumb_down",
        )
        assert id_ == 4

        reactions = Reactions.get_all_reactions(123, 456, 11, 789)
        assert len(reactions) == 3

        reactions = Reactions.get_reactions_by_type(123, 456, 11, 789, "thumb_down")
        assert len(reactions) == 2
