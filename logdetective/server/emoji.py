import asyncio

from typing import List
from collections import Counter

import gitlab

from logdetective.server.models import TimePeriod
from logdetective.server.database.models import (
    Comments,
    Reactions,
    GitlabMergeRequestJobs,
    Forge,
)
from logdetective.server.config import LOG


async def collect_emojis(gitlab_conn: gitlab.Gitlab, period: TimePeriod):
    """
    Collect emoji feedback from logdetective comments saved in database.
    Check only comments created in the last given period of time.
    """
    comments = await Comments.get_since(period.get_period_start_time()) or []
    comments_for_gitlab_connection = [
        comment for comment in comments if comment.forge == gitlab_conn.url
    ]
    await collect_emojis_in_comments(comments_for_gitlab_connection, gitlab_conn)


async def collect_emojis_for_mr(
    project_id: int, mr_iid: int, gitlab_conn: gitlab.Gitlab
):
    """
    Collect emoji feedback from logdetective comments in the specified MR.
    """
    comments = []
    try:
        url = Forge(gitlab_conn.url)
    except ValueError as ex:
        LOG.exception("Attempt to use unrecognized Forge `%s`", gitlab_conn.url)
        raise ex
    mr_jobs = await GitlabMergeRequestJobs.get_by_mr_iid(url, project_id, mr_iid) or []

    comments = [await Comments.get_by_mr_job(mr_job) for mr_job in mr_jobs]
    # Filter all cases when no comments were found. This shouldn't happen if the database
    # is in good order. But checking for it can't hurt.
    comments = [comment for comment in comments if isinstance(comment, Comments)]

    await collect_emojis_in_comments(comments, gitlab_conn)


async def collect_emojis_in_comments(  # pylint: disable=too-many-locals
    comments: List[Comments], gitlab_conn: gitlab.Gitlab
):
    """
    Collect emoji feedback from specified logdetective comments.
    """
    projects = {}
    merge_requests = {}
    for comment in comments:
        mr_job_db = await GitlabMergeRequestJobs.get_by_id(comment.merge_request_job_id)
        if not mr_job_db:
            continue
        try:
            if mr_job_db.id not in projects:
                project = await asyncio.to_thread(
                    gitlab_conn.projects.get, mr_job_db.project_id
                )

                projects[mr_job_db.id] = project
            else:
                project = projects[mr_job_db.id]
            merge_request_iid = mr_job_db.mr_iid
            if merge_request_iid not in merge_requests:
                merge_request = await asyncio.to_thread(
                    project.mergerequests.get, merge_request_iid
                )

                merge_requests[merge_request_iid] = merge_request
            else:
                merge_request = merge_requests[merge_request_iid]

            discussion = await asyncio.to_thread(
                merge_request.discussions.get, comment.comment_id
            )

            # Get the ID of the first note
            if "notes" not in discussion.attributes or len(discussion.attributes["notes"]) == 0:
                LOG.warning(
                    "No notes were found in comment %s in merge request %d",
                    comment.comment_id,
                    merge_request_iid,
                )
                continue

            note_id = discussion.attributes["notes"][0]["id"]
            note = await asyncio.to_thread(merge_request.notes.get, note_id)

        # Log warning with full stack trace, in case we can't find the right
        # discussion, merge request or project.
        # All of these objects can be lost, and we shouldn't treat as an error.
        # Other exceptions are raised.
        except gitlab.GitlabError as e:
            if e.response_code == 404:
                LOG.warning(
                    "Couldn't retrieve emoji counts for comment %s due to GitlabError",
                    comment.comment_id, exc_info=True)
                continue
            LOG.error("Error encountered while processing emoji counts for GitLab comment %s",
                      comment.comment_id, exc_info=True)
            raise

        emoji_counts = Counter(emoji.name for emoji in note.awardemojis.list())

        # keep track of not updated reactions
        # because we need to remove them
        old_emojis = [
            reaction.reaction_type
            for reaction in await Reactions.get_all_reactions(
                comment.forge,
                mr_job_db.project_id,
                mr_job_db.mr_iid,
                mr_job_db.job_id,
                comment.comment_id,
            )
        ]
        for key, value in emoji_counts.items():
            await Reactions.create_or_update(
                comment.forge,
                mr_job_db.project_id,
                mr_job_db.mr_iid,
                mr_job_db.job_id,
                comment.comment_id,
                key,
                value,
            )
            if key in old_emojis:
                old_emojis.remove(key)

        # not updated reactions has been removed, drop them
        await Reactions.delete(
            comment.forge,
            mr_job_db.project_id,
            mr_job_db.mr_iid,
            mr_job_db.job_id,
            comment.comment_id,
            old_emojis,
        )
