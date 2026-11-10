from unittest.mock import AsyncMock, patch
import ipaddress
import socket
import aiohttp
import pytest
from logdetective.utils import SSRFProtectedConnector, SSRFProtectedResolver


def dns_answer(address):
    """Return one aiohttp resolver result for an IPv4 or IPv6 address."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return {
        "hostname": "test", "host": address, "port": 80,
        "family": family, "proto": 6, "flags": 0,
    }


def mock_dns(*addresses):
    """Mock the parent resolver without starting getaddrinfo executor threads."""
    return patch.object(
        aiohttp.ThreadedResolver,
        "resolve",
        new=AsyncMock(return_value=[dns_answer(address) for address in addresses]),
    )


@pytest.mark.asyncio
async def test_resolver_allows_public_ip():
    """Test that public IPs are allowed by the resolver."""
    resolver = SSRFProtectedResolver()
    with mock_dns("8.8.8.8"):
        assert await resolver.resolve("google.com", 80)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "private_ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # AWS Metadata
        "::1",  # IPv6 Loopback
        "fe80::1",  # IPv6 Link-local
    ],
)
async def test_resolver_blocks_private_ips(private_ip):
    """Test that various private/local IPs are blocked."""
    resolver = SSRFProtectedResolver()

    with mock_dns(private_ip):
        with pytest.raises(
            socket.gaierror,
            match=f"resolved to internal IP: {private_ip}.",
        ):
            await resolver.resolve("malicious.local", 80)


@pytest.mark.asyncio
async def test_session_integration():
    """Test that a session using the resolver actually blocks requests."""

    # In a real scenario, we'd use the custom pool manager logic
    # but we can test the resolve logic specifically.
    resolver = SSRFProtectedResolver()

    with mock_dns("127.0.0.1"):
        with pytest.raises(socket.gaierror):
            # We simulate the call that the custom pool manager would make
            await resolver.resolve("localhost", 80)


@pytest.mark.asyncio
async def test_multi_ip_resolution_blocking():
    """Test that if a host resolves to multiple IPs and one is private, it is blocked."""
    resolver = SSRFProtectedResolver()

    with mock_dns("8.8.8.8", "10.0.0.1"):
        with pytest.raises(socket.gaierror, match="resolved to internal IP: 10.0.0.1"):
            await resolver.resolve("mixed-records.com", 80)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("address", "allowed"),
    [
        ("192.168.1.2", True),
        ("172.16.0.1", True),
        ("fc00::1", True),
        ("8.8.8.8", True),
        ("10.95.255.255", True),
        ("10.96.0.0", False),
        ("10.111.255.255", False),
        ("10.112.0.0", True),
        ("172.30.0.1", False),
        ("10.128.0.1", False),
        ("10.0.0.1", False),
        ("100.64.0.1", False),
        ("100.88.0.1", False),
        ("127.0.0.1", False),
        ("169.254.169.254", False),
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("240.0.0.1", False),
        ("192.0.2.1", False),
        ("::1", False),
        ("fe80::1", False),
        ("2001:db8::1", False),
        ("::ffff:192.168.1.2", True),
        ("::ffff:10.96.0.1", False),
        ("::ffff:127.0.0.1", False),
    ],
)
async def test_allowed_hostname_address_policy(address, allowed):
    """Hostname exceptions only admit ordinary private and public addresses."""
    resolver = SSRFProtectedResolver(allowed_hostnames=["Trusted.Example."])
    with mock_dns(address):
        if allowed:
            assert await resolver.resolve("TRUSTED.EXAMPLE", 80)
        else:
            with pytest.raises(socket.gaierror):
                await resolver.resolve("trusted.example", 80)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hostname", ["sub.trusted.example", "untrusted.example", "trusted.example.evil"]
)
async def test_allowlist_requires_exact_hostname(hostname):
    """Neither subdomains nor suffix lookalikes inherit trust."""
    resolver = SSRFProtectedResolver(allowed_hostnames=["trusted.example"])
    with mock_dns("192.168.1.2"):
        with pytest.raises(socket.gaierror):
            await resolver.resolve(hostname, 80)


@pytest.mark.asyncio
async def test_allowed_hostname_mixed_dns_answers_are_rejected():
    """One unsafe answer rejects the entire DNS response."""
    resolver = SSRFProtectedResolver(allowed_hostnames=["trusted.example"])
    with mock_dns("192.168.1.2", "10.96.0.1"):
        with pytest.raises(socket.gaierror):
            await resolver.resolve("trusted.example", 80)


@pytest.mark.asyncio
async def test_custom_cluster_cidrs_replace_defaults():
    """Configured ranges are not appended to the defaults."""
    resolver = SSRFProtectedResolver(
        allowed_hostnames=["trusted.example"],
        blocked_cluster_cidrs=[ipaddress.ip_network("192.168.1.0/24")],
    )
    with mock_dns("10.96.0.1"):
        assert await resolver.resolve("trusted.example", 80)
    with mock_dns("192.168.1.0"):
        with pytest.raises(socket.gaierror):
            await resolver.resolve("trusted.example", 80)
    with mock_dns("192.168.2.1"):
        assert await resolver.resolve("trusted.example", 80)


@pytest.mark.asyncio
async def test_connector_checks_ip_literal_urls():
    """aiohttp bypasses DNS for literals, so the connector checks them itself."""
    resolver = SSRFProtectedResolver(allowed_hostnames=["trusted.example"])
    connector = SSRFProtectedConnector(resolver)
    try:
        for address in ("127.0.0.1", "10.96.0.1", "::ffff:127.0.0.1"):
            with pytest.raises(socket.gaierror):
                await connector._resolve_host(address, 80)  # pylint: disable=protected-access
        assert await connector._resolve_host("8.8.8.8", 80)  # pylint: disable=protected-access
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_session_blocks_ip_literal_url_before_connecting():
    """A real session receives the check-specific rejection for loopback IPs."""
    connector = SSRFProtectedConnector(SSRFProtectedResolver())
    async with aiohttp.ClientSession(connector=connector) as session:
        with pytest.raises(aiohttp.ClientConnectorError) as error:
            await session.get("http://127.0.0.1:9/")
    assert isinstance(error.value.os_error, socket.gaierror)
    assert "resolved to internal IP: 127.0.0.1" in str(error.value.os_error)


@pytest.mark.asyncio
async def test_redirect_target_is_checked_by_same_connector():
    """Each new hostname in an aiohttp redirect receives its own DNS check."""
    resolver = SSRFProtectedResolver(allowed_hostnames=["trusted.example"])
    connector = SSRFProtectedConnector(resolver)
    try:
        with mock_dns("192.168.1.2"):
            assert await connector._resolve_host("trusted.example", 80)  # pylint: disable=protected-access
        with mock_dns("127.0.0.1"):
            with pytest.raises(socket.gaierror):
                await connector._resolve_host("redirect.example", 80)  # pylint: disable=protected-access
    finally:
        await connector.close()
