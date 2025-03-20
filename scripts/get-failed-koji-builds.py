#!/usr/bin/python3
"""
Print failed builds from Koji that are followed by successfull one.
With the link to failed taske and link to commits.
To ease debuging why the build fails.
"""

import koji
import rpm
from datetime import datetime, timedelta

DATE = (datetime.now() - timedelta(days=30)).isoformat()
MAX_BUILDS = 200


def get_failed_builds(session, max_builds=100):
    """
    Retrieves the last `max_builds` failed builds from Fedora's Koji instance.

    :param max_builds: The maximum number of failed builds to retrieve.
    :return: A list of dictionaries containing build information.
    """
    # Use the correct Koji API method to search for builds
    builds = session.listBuilds(
        state=koji.BUILD_STATES["FAILED"], completeBefore=None, completeAfter=DATE
    )

    # Limit the number of builds returned
    builds = builds[-max_builds:]

    # Format the results
    failed_builds = []
    for build in builds:
        task_id = build.get("task_id")
        if task_id is None:
            continue
        task = session.getTaskInfo(build.get("task_id"))
        if task and task.get("method") == "build":
            failed_builds.append(
                {
                    "build_id": build.get("build_id"),
                    "package_name": build.get("package_name"),
                    "package_id": build.get("package_id"),
                    "version": build.get("version"),
                    "release": build.get("release"),
                    "completion_time": build.get("completion_time"),
                    "task_id": task,
                }
            )
            print(
                f"Build ID: {build['build_id']}, Package: {build['package_name']}, Version: {build['version']}, Release: {build['release']}"
            )
            has_newer_build(
                session,
                build["package_id"],
                build["completion_time"],
                build["version"],
                build["release"],
                task,
            )

    return failed_builds


def has_newer_build(session, packageID, completeAfter, version, release, task):
    builds = session.listBuilds(
        packageID=packageID,
        state=koji.BUILD_STATES["COMPLETE"],
        completeAfter=completeAfter,
    )
    builds = builds[:MAX_BUILDS]
    for build in builds:
        version_result = rpm.labelCompare(build["version"], version)
        release_result = rpm.labelCompare(build["release"], release)
        if (version_result > 0) or (version_result == 0 and release_result > 0):
            print(
                f"  COMPLETED: Build ID: {build['build_id']}, Package: {build['package_name']}, Version: {build['version']}, Release: {build['release']}"
            )
            print(
                f"  Koji task: https://koji.fedoraproject.org/koji/taskinfo?taskID={task['id']}"
            )
            print(
                f"  Check https://src.fedoraproject.org/rpms/{build['package_name']}/commits/rawhide"
            )
            return


koji_url = "https://koji.fedoraproject.org/kojihub"
# Initialize a Koji client session
session = koji.ClientSession(koji_url)

get_failed_builds(session, max_builds=MAX_BUILDS)
