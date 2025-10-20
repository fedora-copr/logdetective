import os
import datetime

import gitlab
import responses
from responses import _recorder
from sqlalchemy import select
import pytest

from logdetective.server.models import TimePeriod
from logdetective.server.database.models import (
    GitlabMergeRequestJobs,
    Comments,
    Reactions,
    Forge,
)
from logdetective.server.emoji import collect_emojis, collect_emojis_for_mr

from tests.server.test_helpers import (
    DatabaseFactory,
)

# the token is needed only when registering responses
# export your own private token to be able to re-register responses
gitlab_conn = gitlab.Gitlab(
    url="https://gitlab.com/", private_token=os.environ.get("LOGDETECTIVE_TOKEN")
)

COLLECT_EMOJIS_RESPONSES = "tests/server/data/test_collect_emojis.yaml"
COLLECT_EMOJIS_FOR_MR_RESPONSES = "tests/server/data/test_collect_emojis_for_mr.yaml"
EMOJI_REMOVED_RESPONSES = "tests/server/data/test_emoji_removed.yaml"
COLLECT_EMOJIS_RESPONSES_WITH_404 = (
    "tests/server/data/test_collect_emojis_with_404.yaml"
)


async def populate_db_with_comments_for_libtiff_mr_26():
    """Mock db using data related with the following MR
    https://gitlab.com/redhat/centos-stream/rpms/libtiff/-/merge_requests/26
    """
    async with DatabaseFactory().make_new_db() as session_factory:
        async with session_factory() as db_session:
            db_session.add(
                GitlabMergeRequestJobs(
                    id=11,
                    forge=Forge.gitlab_com,
                    project_id=23667077,
                    mr_iid=26,
                    job_id=1,
                )
            )
            await db_session.commit()
            db_session.add(
                Comments(
                    forge=Forge.gitlab_com,
                    merge_request_job_id=11,
                    comment_id="1ba1f46903f2b4e0573d514ed6d2cae29ab040f6",  # note id 2462509330
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                )
            )
            await db_session.commit()

            db_session.add(
                GitlabMergeRequestJobs(
                    id=12,
                    forge=Forge.gitlab_com,
                    project_id=23667077,
                    mr_iid=26,
                    job_id=2,
                )
            )
            await db_session.commit()
            db_session.add(
                Comments(
                    forge=Forge.gitlab_com,
                    merge_request_job_id=12,
                    comment_id="91090f16d27d4ad053f39990c5a47a592565d38f",  # note id 2464715549
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                )
            )
            await db_session.commit()

            yield db_session


async def _test_collect_emojis():
    query = select(Reactions)
    async for db_session in populate_db_with_comments_for_libtiff_mr_26():
        await collect_emojis(gitlab_conn, TimePeriod(hours=1))
        query_result = await db_session.execute(query)
        reactions = query_result.scalars().all()
        assert len(reactions) == 2
        types = [reaction.reaction_type for reaction in reactions]
        assert "thumbsup" in types
        assert "thumbsdown" in types


@_recorder.record(file_path=COLLECT_EMOJIS_RESPONSES)
async def record_responses_for_collect_emojis():
    await _test_collect_emojis()


@responses.activate
@pytest.mark.asyncio
async def test_collect_emojis():
    responses._add_from_file(file_path=COLLECT_EMOJIS_RESPONSES)
    await _test_collect_emojis()


async def _test_collect_emojis_for_mr():
    query = select(Reactions)
    async for db_session in populate_db_with_comments_for_libtiff_mr_26():
        await collect_emojis_for_mr(23667077, 26, gitlab_conn)
        query_result = await db_session.execute(query)
        reactions = query_result.scalars().all()
        assert len(reactions) == 2
        types = [reaction.reaction_type for reaction in reactions]
        assert "thumbsup" in types
        assert "thumbsdown" in types


@_recorder.record(file_path=COLLECT_EMOJIS_FOR_MR_RESPONSES)
async def record_responses_for_collect_emojis_for_mr():
    await _test_collect_emojis_for_mr()


@responses.activate
@pytest.mark.asyncio
async def test_collect_emojis_for_mr():
    responses._add_from_file(file_path=COLLECT_EMOJIS_FOR_MR_RESPONSES)
    await _test_collect_emojis_for_mr()


async def _test_emoji_removed():
    query = select(Reactions)
    async for db_session in populate_db_with_comments_for_libtiff_mr_26():
        await collect_emojis(gitlab_conn, TimePeriod(hours=1))
        query_result = await db_session.execute(query)
        reactions = query_result.scalars().all()
        assert len(reactions) == 2
        types = [reaction.reaction_type for reaction in reactions]
        assert "thumbsup" in types
        assert "thumbsdown" in types

        await collect_emojis(gitlab_conn, TimePeriod(hours=1))
        query_result = await db_session.execute(query)
        reactions = query_result.scalars().all()
        assert len(reactions) == 1
        types = [reaction.reaction_type for reaction in reactions]
        assert "thumbsup" in types
        assert "thumbsdown" not in types


@_recorder.record(file_path=EMOJI_REMOVED_RESPONSES)
@pytest.mark.asyncio
async def record_responses_for_emoji_removed():
    await _test_emoji_removed()


@responses.activate
@pytest.mark.asyncio
async def test_emoji_removed():
    responses._add_from_file(file_path=EMOJI_REMOVED_RESPONSES)
    await _test_emoji_removed()


@pytest.mark.asyncio
@responses.activate
async def test_collect_emojis_with_404():
    # To re-create the test use COLLECT_EMOJS_RESPONSES and
    # update status with 404
    # body with '404 Discussion Not Found'
    # content type with text/plain
    # for response with url
    # https://gitlab.com/api/v4/projects/23667077/merge_requests/
    # 26/discussions/1ba1f46903f2b4e0573d514ed6d2cae29ab040f6

    responses._add_from_file(file_path=COLLECT_EMOJIS_RESPONSES_WITH_404)
    query = select(Reactions)
    async for db_session in populate_db_with_comments_for_libtiff_mr_26():
        await collect_emojis(gitlab_conn, TimePeriod(hours=1))
        query_result = await db_session.execute(query)
        reactions = query_result.scalars().all()
        assert len(reactions) == 1
        types = [reaction.reaction_type for reaction in reactions]
        assert "thumbsup" not in types
        assert "thumbsdown" in types


if __name__ == "__main__":
    # call the module to re-record responses for tests
    # responses are recorded with the wrong content-type
    # substitute text/plain with application/json
    record_responses_for_collect_emojis()
    record_responses_for_collect_emojis_for_mr()
    record_responses_for_emoji_removed()
