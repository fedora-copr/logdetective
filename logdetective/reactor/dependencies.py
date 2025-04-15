import aiohttp
import jinja2

from typing import NamedTuple

from fastapi import Request


class HttpConnections(NamedTuple):
    """Connections established by reactor to
    - gitlab
    - logdetective server
    """

    gitlab: aiohttp.ClientSession
    logdetective: aiohttp.ClientSession


async def get_http_connections(request: Request) -> HttpConnections:
    """Get reactor's connections"""
    return HttpConnections(
        gitlab=request.app.gitlab_http, logdetective=request.app.logdetective_http
    )


async def get_jinja_env(request: Request) -> jinja2.Environment:
    """Get reactor's jinja template"""
    return request.app.jinja_env
