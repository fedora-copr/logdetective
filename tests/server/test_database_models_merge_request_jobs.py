import pytest

from sqlalchemy.exc import IntegrityError

from tests.server.test_helpers import (
    DatabaseFactory,
)

from logdetective.server.database.models import (
    Forge,
    GitlabMergeRequestJobs,
    Comments,
    Reactions,
)


def test_create_and_get_GitlabMergeRequestJobs():
    with DatabaseFactory().make_new_db() as _:
        forge = Forge.gitlab_com
        id_ = GitlabMergeRequestJobs.create(
            forge, project_id=123, mr_iid=456, job_id=11
        )
        assert id_
        prev_id = id_

        # no same project/mr/job within same forge
        # but we can have same project/mr/job in different forges
        with pytest.raises(IntegrityError):
            GitlabMergeRequestJobs.create(forge, project_id=123, mr_iid=456, job_id=11)
        id_ = GitlabMergeRequestJobs.create(
            Forge.gitlab_cee_redhat_com, project_id=123, mr_iid=456, job_id=11
        )
        assert id_ > prev_id
        prev_id = id_

        id_ = GitlabMergeRequestJobs.create(
            forge, project_id=123, mr_iid=456, job_id=22
        )
        assert id_ > prev_id
        prev_id = id_

        # job_id is unique within the gitlab instance,
        # can't be associated with two project_ids or two mr_iids
        with pytest.raises(IntegrityError):
            GitlabMergeRequestJobs.create(forge, project_id=124, mr_iid=456, job_id=11)
        with pytest.raises(IntegrityError):
            GitlabMergeRequestJobs.create(forge, project_id=123, mr_iid=457, job_id=11)

        mr = GitlabMergeRequestJobs.get_by_details(forge, 123, 456, 11)
        assert mr.id == 1

        mr = GitlabMergeRequestJobs.get_by_details(forge, 123, 456, 1)
        assert mr is None


def test_create_and_get_Comments():
    with DatabaseFactory().make_new_db() as _:
        forge = Forge.gitlab_com
        mr_id = GitlabMergeRequestJobs.create(
            forge, project_id=123, mr_iid=456, job_id=11
        )
        assert mr_id == 1
        comment_db_id = Comments.create(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=11,
            comment_id="789",
        )
        assert comment_db_id == 1
        comment = Comments.get_by_id(comment_db_id)
        assert comment.merge_request_job_id == 1

        # create a new mr (implicitly) if it does not exist
        comment_db_id = Comments.create(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=21,
            comment_id="7890",
        )
        assert comment_db_id == 2
        comment = Comments.get_by_id(comment_db_id)
        assert comment.merge_request_job_id == 2

        # no more than 1 comment for 1 job
        with pytest.raises(IntegrityError):
            Comments.create(
                forge,
                project_id=123,
                mr_iid=456,
                job_id=21,
                comment_id="7891",
            )

        # no more than 1 comment_id for the same forge
        with pytest.raises(IntegrityError):
            Comments.create(
                forge,
                project_id=111,
                mr_iid=222,
                job_id=33,
                comment_id="7890",
            )

        comment = Comments.get_by_gitlab_id(forge, "789")
        assert comment.id == 1
        assert comment.merge_request_job_id == 1

        # one more comment for the same merge request
        comment_db_id = Comments.create(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=23,
            comment_id="7892",
        )
        assert comment_db_id

        comment = Comments.get_latest_comment(forge, 123, 456)
        assert comment.id == comment_db_id

        comments = Comments.get_mr_comments(forge, 123, 456)
        assert len(comments) == 3

        # Try to get a comment on an MR that doesn't exist
        comment = Comments.get_latest_comment(forge, 123, 457)
        assert comment is None

        # Try to create a comment associated with a job of very
        # high ID ( > 31 bits)
        Comments.create(
            forge=forge,
            project_id=111,
            mr_iid=222,
            job_id=30000000000,
            comment_id="7893",
        )

        comment = Comments.get_or_create(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=22,
            comment_id="7893",
        )
        assert comment.id


def test_create_and_get_Reactions():
    with DatabaseFactory().make_new_db() as _:
        forge = Forge.gitlab_com
        db_id = Reactions.create_or_update(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=11,
            comment_id="789",
            reaction_type="thumb_up",
            count=1,
        )
        assert db_id == 1
        db_id = Reactions.create_or_update(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=11,
            comment_id="789",
            reaction_type="thumb_down",
            count=1,
        )
        assert db_id == 2

        # update reaction count
        db_id = Reactions.create_or_update(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=11,
            comment_id="789",
            reaction_type="thumb_down",
            count=3,
        )
        assert db_id == 2

        # reaction for another comment
        db_id = Reactions.create_or_update(
            forge,
            project_id=123,
            mr_iid=456,
            job_id=21,
            comment_id="7890",
            reaction_type="thumb_down",
            count=3,
        )
        assert db_id == 3

        # same reaction in a new forge creates new entry
        db_id = Reactions.create_or_update(
            Forge.gitlab_cee_redhat_com,
            project_id=123,
            mr_iid=456,
            job_id=21,
            comment_id="7890",
            reaction_type="thumb_down",
            count=3,
        )
        assert db_id == 4

        reactions = Reactions.get_all_reactions(forge, 123, 456, 11, "789")
        assert len(reactions) == 2

        reaction = Reactions.get_reaction_by_type(
            forge, 123, 456, 11, "789", "thumb_down"
        )
        assert reaction.count == 3

        reaction = Reactions.get_reaction_by_type(forge, 123, 456, 11, "789", "thumbs")
        assert reaction is None
