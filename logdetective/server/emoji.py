import asyncio
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
    Collect emoji feedback from logdetective comments in MRs.
    Check only MRs comments created on the last given period of time.
    """
    comments = Comments.get_since(period.get_period_start_time())
    comments_for_gitlab_connection = [
        comment for comment in comments if comment.forge == gitlab_conn.url
    ]
    projects = {}
    mrs = {}
    for comment in comments_for_gitlab_connection:
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

        note = mr.notes.get(comment.comment_id)
        emoji_counts = Counter(emoji.name for emoji in note.awardemojis.list())

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
