import asyncio

from typing import List
from collections import Counter

import gitlab

from logdetective.server.models import TimePeriod
from logdetective.server.database.models import (
    Comments,
    Reactions,
    GitlabMergeRequestJobs,
)


async def collect_emojis(gitlab_conn: gitlab.Gitlab, period: TimePeriod):
    """
    Collect emoji feedback from logdetective comments saved in database.
    Check only comments created in the last given period of time.
    """
    comments = Comments.get_since(period.get_period_start_time())
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
    mr_jobs = GitlabMergeRequestJobs.get_by_mr_iid(gitlab_conn.url, project_id, mr_iid)
    comments = [Comments.get_by_mr_job(mr_job) for mr_job in mr_jobs]
    await collect_emojis_in_comments(comments, gitlab_conn)


async def collect_emojis_in_comments(  # pylint: disable=too-many-locals
    comments: List[Comments], gitlab_conn: gitlab.Gitlab
):
    """
    Collect emoji feedback from specified logdetective comments.
    """
    projects = {}
    mrs = {}
    for comment in comments:
        mr_job_db = GitlabMergeRequestJobs.get_by_id(comment.merge_request_job_id)
        if mr_job_db.id not in projects:
            projects[mr_job_db.id] = project = await asyncio.to_thread(
                gitlab_conn.projects.get, mr_job_db.project_id
            )
        else:
            project = projects[mr_job_db.id]
        mr_iid = mr_job_db.mr_iid
        if mr_iid not in mrs:
            mrs[mr_iid] = mr = await asyncio.to_thread(
                project.mergerequests.get, mr_iid
            )
        else:
            mr = mrs[mr_iid]

        discussion = mr.discussions.get(comment.comment_id)

        # Get the ID of the first note
        note_id = discussion.attributes["notes"][0]["id"]
        note = mr.notes.get(note_id)

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
