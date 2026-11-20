"""Suite downloads enforce address policy through real aiohttp requests."""

import ipaddress
import socket

import aiohttp
from aiohttp import web
import pytest
import pytest_asyncio

from logdetective.artifacts import fetch_suite_artifacts
from logdetective.config import SERVER_CONFIG
from logdetective.models import TestingFarmSuiteRequest as SuiteRequest


@pytest_asyncio.fixture
async def artifact_transport(monkeypatch):
    """Dial a local server only after the real connector has checked addresses.

    DNS supplies controlled public/private addresses. Remapping the final socket
    destination avoids external traffic without bypassing resolver/connector or
    aiohttp redirect handling. A blocked address must never reach this transport.
    """
    requests = []
    connections = []

    async def serve(request):
        requests.append((request.method, request.path))
        if request.path == "/redirect" and request.method == "GET":
            raise web.HTTPFound("http://127.0.0.1/forbidden")
        return web.Response(text="error in log\n")

    application = web.Application()
    application.router.add_route("*", "/{path:.*}", serve)
    runner = web.AppRunner(application)
    await runner.setup()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    site = web.SockSite(runner, listener)
    await site.start()

    async def resolve(_self, host, port=0, family=socket.AF_INET):
        address = "192.168.1.2" if host == "trusted.example" else "8.8.8.8"
        return [{"hostname": host, "host": address, "port": port,
                 "family": socket.AF_INET, "proto": socket.IPPROTO_TCP, "flags": 0}]

    connect = aiohttp.TCPConnector._wrap_create_connection

    async def dial(self, *args, addr_infos, **kwargs):
        connections.extend(info[4][0] for info in addr_infos)
        return await connect(self, *args, addr_infos=[(
            socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port),
        )], **kwargs)

    monkeypatch.setattr(aiohttp.ThreadedResolver, "resolve", resolve)
    monkeypatch.setattr(aiohttp.TCPConnector, "_wrap_create_connection", dial)
    try:
        yield requests, connections
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,enabled,allowlist,cidrs,success",
    [
        ("http://127.0.0.1/log", True, [], [], False),
        ("http://[::1]/log", True, [], [], False),
        ("http://[::ffff:127.0.0.1]/log", True, [], [], False),
        ("http://public.example/log", True, [], [], True),
        ("http://8.8.8.8/log", True, [], [], True),
        ("http://public.example/log", True, [], ["8.8.8.0/24"], False),
        ("http://trusted.example/log", True, ["trusted.example"], [], True),
        ("http://trusted.example/log", True, [], [], False),
        ("http://trusted.example/log", True, ["trusted.example"], ["192.168.1.0/24"], False),
        ("http://127.0.0.1/log", False, [], ["127.0.0.0/8"], True),
    ],
)
async def test_suite_download_address_policy(
    monkeypatch, artifact_transport, *, url, enabled, allowlist, cidrs, success,
):
    """Configured policy governs actual HEAD/GET traffic, not only construction."""
    monkeypatch.setattr(SERVER_CONFIG.general, "block_localhost_urls", enabled)
    policy = SERVER_CONFIG.general.ssrf_protection
    monkeypatch.setattr(policy, "allowed_hostnames", allowlist)
    monkeypatch.setattr(policy, "blocked_cluster_cidrs", [ipaddress.ip_network(c) for c in cidrs])
    payload = SuiteRequest(suite={"plan": "/plan"}, cases=[{
        "case_id": "c1", "name": "/test", "result": "failed",
        "artifacts": [{"name": "log", "url": url}],
    }])
    result = await fetch_suite_artifacts(payload, 100)
    requests, connections = artifact_transport
    if success:
        assert result.artifacts == {"case/c1/log": "error in log\n"}
        assert requests == [("HEAD", "/log"), ("GET", "/log")]
        assert connections
        assert not result.warnings
    else:
        assert not result.artifacts
        assert result.errors["c1"].error.code == "artifact_error"
        assert result.warnings[0].code == "artifact_unavailable"
        assert not requests
        assert not connections


@pytest.mark.asyncio
async def test_suite_redirect_to_literal_is_blocked(monkeypatch, artifact_transport):
    """An allowed HEAD/GET cannot redirect the worker into a loopback service."""
    monkeypatch.setattr(SERVER_CONFIG.general, "block_localhost_urls", True)
    payload = SuiteRequest(suite={"plan": "/plan", "state": "error"}, cases=[], shared_artifacts=[{
        "name": "log", "url": "http://public.example/redirect",
    }])
    result = await fetch_suite_artifacts(payload, 100)
    requests, connections = artifact_transport
    assert requests == [("HEAD", "/redirect"), ("GET", "/redirect")]
    assert connections == ["8.8.8.8"]
    assert not result.artifacts
    assert result.warnings[0].source_file == "shared/log"
