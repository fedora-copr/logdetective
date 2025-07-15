import asyncio

from typing import List, Callable
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
    comments = Comments.get_since(period.get_period_start_time()) or []
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
    mr_jobs = GitlabMergeRequestJobs.get_by_mr_iid(url, project_id, mr_iid) or []

    comments = [Comments.get_by_mr_job(mr_job) for mr_job in mr_jobs]
    await collect_emojis_in_comments(comments, gitlab_conn)


async def _handle_gitlab_operation(func: Callable, *args):
    """
    It handles errors for the specified GitLab operation.
    After executing it in a separate thread.
    """
    try:
        return await asyncio.to_thread(func, *args)
    except (gitlab.GitlabError, gitlab.GitlabGetError) as e:
        log_msg = f"Error during GitLab operation {func}{args}: {e}"
        if "Not Found" in str(e):
            LOG.error(log_msg)
        else:
            LOG.exception(log_msg)
    except Exception as e:  # pylint: disable=broad-exception-caught
        LOG.exception(
            "Unexpected error during GitLab operation %s(%s): %s", func, args, e
        )


async def collect_emojis_in_comments(  # pylint: disable=too-many-locals
    comments: List[Comments], gitlab_conn: gitlab.Gitlab
):
    """
    Collect emoji feedback from specified logdetective comments.
    """
    projects = {}
    merge_requests = {}
    for comment in comments:
        mr_job_db = GitlabMergeRequestJobs.get_by_id(comment.merge_request_job_id)
        if not mr_job_db:
            continue
        if mr_job_db.id not in projects:
            project = await _handle_gitlab_operation(
                gitlab_conn.projects.get, mr_job_db.project_id
            )
            if not project:
                continue
            projects[mr_job_db.id] = project
        else:
            project = projects[mr_job_db.id]
        merge_request_iid = mr_job_db.mr_iid
        if merge_request_iid not in merge_requests:
            merge_request = await _handle_gitlab_operation(
                project.mergerequests.get, merge_request_iid
            )
            if not merge_request:
                continue
            merge_requests[merge_request_iid] = merge_request
        else:
            merge_request = merge_requests[merge_request_iid]

        discussion = await _handle_gitlab_operation(
            merge_request.discussions.get, comment.comment_id
        )
        if not discussion:
            continue

        # Get the ID of the first note
        note_id = discussion.attributes["notes"][0]["id"]
        note = await _handle_gitlab_operation(merge_request.notes.get, note_id)
        if not note:
            continue

        emoji_counts = Counter(emoji.name for emoji in note.awardemojis.list())

        # keep track of not updated reactions
        # because we need to remove them
        old_emojis = [
            reaction.reaction_type
            for reaction in Reactions.get_all_reactions(
                comment.forge,
                mr_job_db.project_id,
                mr_job_db.mr_iid,
                mr_job_db.job_id,
                comment.comment_id,
            )
        ]
        for key, value in emoji_counts.items():
            Reactions.create_or_update(
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
        Reactions.delete(
            comment.forge,
            mr_job_db.project_id,
            mr_job_db.mr_iid,
            mr_job_db.job_id,
            comment.comment_id,
            old_emojis,
        )
