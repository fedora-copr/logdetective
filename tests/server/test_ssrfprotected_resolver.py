from unittest.mock import patch
import socket
import pytest
from requests import exceptions
from logdetective.server.utils import SSRFProtectedResolver


@pytest.mark.asyncio
async def test_resolver_allows_public_ip():
    """Test that public IPs are allowed by the resolver."""
    resolver = SSRFProtectedResolver()
    # Mocking getaddrinfo to return a public IP (Google DNS)
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))
        ]

        # This should not raise an exception
        try:
            await resolver.resolve("google.com", 80)
        except exceptions.ConnectionError:
            pytest.fail("SSRFProtectedResolver blocked a public IP")


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

    family = socket.AF_INET6 if ":" in private_ip else socket.AF_INET

    sockaddr = (private_ip, 80, 0, 0) if family is socket.AF_INET6 else (private_ip, 80)

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [(family, socket.SOCK_STREAM, 6, "", sockaddr)]
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

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ]

        with pytest.raises(socket.gaierror):
            # We simulate the call that the custom pool manager would make
            await resolver.resolve("localhost", 80)


@pytest.mark.asyncio
async def test_multi_ip_resolution_blocking():
    """Test that if a host resolves to multiple IPs and one is private, it is blocked."""
    resolver = SSRFProtectedResolver()

    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        # Returns one public and one private IP
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80)),
        ]

        with pytest.raises(socket.gaierror, match="resolved to internal IP: 10.0.0.1"):
            await resolver.resolve("mixed-records.com", 80)
