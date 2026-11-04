from time import perf_counter

import pytest

from logdetective.utils import sanitize_artifact


def test_sanitize_identifiers():
    """Test that every type of personal identifier gets sanitized."""
    test_lines = [
        "email to sanitize: example-mail@arbitrary.domain.com",
        "pubkey-abcdef0123456789abcdef0123456789aaaaaaaa",
        "gpg-pubkey-deadbeef-01234567",
        "RSA key 0000ABCD1234FFFF0123456701234567",
        "RSA key FEEDDEADBEEFCAFE",
        "GPG Fingerprint: 1234 ABCD 1234 ABCD 1234 ABCD 1234 ABCD 5678 EF90",
        "GPG fingerprint: 1234ABCD1234ABCD1234ABCD1234ABCD5678EF90",
    ]

    # by doing this, we also check if the order of lines isn't somehow changed
    redacted_lines = sanitize_artifact("\n".join(test_lines)).split("\n")

    assert "copr-team@redhat.com" in redacted_lines[0]
    assert "example-mail@arbitrary.domain.com" not in redacted_lines[0]

    assert f"pubkey-{'f' * 40}" in redacted_lines[1]
    assert "abcdef0123456789abcdef0123456789aaaaaaaa" not in redacted_lines[1]

    assert f"pubkey-{'f' * 40}" in redacted_lines[2]
    assert all(i not in redacted_lines[2] for i in ["deadbeef", "01234567"])

    assert "0000ABCD1234FFFF0123456701234567" not in redacted_lines[3]
    assert f"RSA key {'F' * 40}" in redacted_lines[3]

    assert "FEEDDEADBEEFCAFE" not in redacted_lines[4]
    assert f"RSA key {'F' * 40}" in redacted_lines[4]

    assert all(i not in redacted_lines[5] for i in ["1234", "ABCD", "5678", "EF90"])
    assert all(i in redacted_lines[5] for i in ["GPG Fingerprint", " FFFF" * 10])

    assert all(i not in redacted_lines[6] for i in ["1234", "ABCD", "5678", "EF90"])
    assert all(i in redacted_lines[6] for i in ["GPG Fingerprint", " FFFF" * 10])


def test_do_not_sanitize_identifiers():
    """Testing things that should not be sanitized from logs."""
    test_lines = [
        "root@12.34.56.78",
        "git hash deadbee or 1234567",
        "git checkout 0d5375d54b258ec63edd1fb5d58c37d58ce8be8b",
    ]
    redacted_lines = sanitize_artifact("\n".join(test_lines)).split("\n")

    assert "root@12.34.56.78" in redacted_lines[0]
    assert "copr-team" not in redacted_lines[0]

    assert all(i in redacted_lines[1] for i in ["deadbee", "1234567"])
    assert all(i not in redacted_lines[1] for i in ["ffff", "FFFF"])

    assert "0d5375d54b258ec63edd1fb5d58c37d58ce8be8b" in redacted_lines[2]
    assert all(i not in redacted_lines[2] for i in ["ffff", "FFFF"])


def test_sanitize_all_emails():
    """Every line of the log has an email that should be replaced."""
    test_lines = [
        "onename+something@some-domain.org",
        "jdoe@redhat.com",
        "nickname2456@coolest.domain.ever",
        "~/Documents/project$ ./script ran by someone.important@gmail.com",
        "CentOS (CentOS Official Signing Key) <security@centos.org>",
        "ALLCAPS@EMAIL.COM",
        # commit headers are only in specfiles, not in the logs themselves,
        # but still we can check if we catch the email
        "- commit header: Mon Jan 1 2020 John Doe <jdoe@domain.com>",
    ]

    redacted_lines = sanitize_artifact("\n".join(test_lines)).split("\n")

    assert "copr-team@redhat.com" in redacted_lines[0]
    assert all(i not in redacted_lines[0] for i in ["onename", "something", "some-domain"])

    assert "copr-team@redhat.com" in redacted_lines[1]
    assert "jdoe" not in redacted_lines[1]

    assert "copr-team@redhat.com" in redacted_lines[2]
    assert all(i not in redacted_lines[0] for i in ["nickname2456", "coolest", "domain"])

    assert "copr-team@redhat.com" in redacted_lines[3]
    assert all(i not in redacted_lines[0] for i in ["someone", "important", "gmail"])

    assert "<copr-team@redhat.com>" in redacted_lines[4]
    assert all(i not in redacted_lines[0] for i in ["security", "centos"])

    assert "copr-team@redhat.com" in redacted_lines[5]
    assert all(i not in redacted_lines[0] for i in ["ALLCAPS", "EMAIL"])

    assert "<copr-team@redhat.com>" in redacted_lines[6]
    assert all(i not in redacted_lines[0] for i in ["jdoe", "domain"])


def test_do_not_sanitize_emails():
    """Nothing in the log should be considered an email."""
    test_lines = [
        "warning: root@localhost or root@127.0.0.1 should not be sanitized as emails",
        "bind-9.11.36/bin/tests/system/dlz/ns1/dns-root/com/example/dns.d/@/",
        "sed -e 's/@MAJOR_VERSION@/2/' -e 's/@MINOR_VERSION@/11/' ../udisks/udisksversion.h.in",
        "\"project_owner\": \"@python\",",
        "@pytest.mark.parametrize(",
        "GIT_AUTHOR_DATE=@1000000000",
        "something something invalid/character@ema/il.com",
        "Not found: /nut-2.8.3-build/BUILDROOT/usr/lib/systemd/system/enphase-monitor@.service",
        "func.func @inline_func_with_file_locations(%arg0: i32)",
        "36:  func.func private @replaced_foo() attributes {sym.new_name = \"replaced_foo\"}",
        "-e 's,@PACKAGE_URL\\@,http://avahi.org/",
        "static const G@Type@Value values[] = {",
        "this script was ran as admin@InternalService or user@192.168.1.1",
        "this looks like an email, but is not: systemd-mkswap@dev-sdy3.service",
        "same here serial-getty@ttyS0.service",
        "this is some journal entry, not a mail: system@0005ebbfd4385848-2e5dff5354ab9bcf.journal",
    ]

    redacted_lines = sanitize_artifact("\n".join(test_lines)).split("\n")

    assert all("copr-team@redhat.com" not in line for line in redacted_lines)

    assert all(i in redacted_lines[0] for i in ["root@localhost", "root@127.0.0.1"])
    assert all(i in redacted_lines[1] for i in ["bind-9.11.36", "example/dns.d/@/"])
    assert all(i in redacted_lines[2] for i in [
        "s/@MAJOR_VERSION@/2/", "s/@MINOR_VERSION@/11/", "udisks/udisksversion.h.in"
    ])
    assert all(i in redacted_lines[3] for i in ["\"@python\""])
    assert all(i in redacted_lines[4] for i in ["@pytest.mark.parametrize"])
    assert all(i in redacted_lines[5] for i in ["GIT_AUTHOR_DATE=@1000000000"])
    assert all(i in redacted_lines[6] for i in ["invalid/character@ema/il.com"])
    assert all(i in redacted_lines[7] for i in ["nut-2.8.3-build", "enphase-monitor@.service"])
    assert all(i in redacted_lines[8] for i in ["func.func", "@inline_func_with_file_locations"])
    assert all(i in redacted_lines[9] for i in ["func.func", "@replaced_foo"])
    assert all(i in redacted_lines[10] for i in ["@PACKAGE_URL", "http://avahi.org/"])
    assert all(i in redacted_lines[11] for i in ["G@Type@Value"])
    assert all(i in redacted_lines[12] for i in ["admin@InternalService", "user@192.168.1.1"])
    assert all(i in redacted_lines[13] for i in ["systemd-mkswap@dev-sdy3.service"])
    assert all(i in redacted_lines[14] for i in ["serial-getty@ttyS0.service"])
    assert all(i in redacted_lines[15] for i in [
        "system@0005ebbfd4385848-2e5dff5354ab9bcf.journal"
    ])


@pytest.mark.parametrize(
    "suffix",
    [
        "automount", "device", "mount", "path", "scope", "service",
        "slice", "socket", "swap", "target", "timer",
    ],
)
def test_systemd_unit_suffixes_remain_visible(suffix: str) -> None:
    """Systemd instances and templates remain readable beside real emails."""
    artifact = f"foo@bar.{suffix} foo@.{suffix} alice@example.com"
    expected = f"foo@bar.{suffix} foo@.{suffix} copr-team@redhat.com"
    assert sanitize_artifact(artifact) == expected


def test_service_suffix_exempts_entire_candidate() -> None:
    """A dotted systemd instance is retained under the unit suffix rule."""
    assert sanitize_artifact("alice@example.com.service") == "alice@example.com.service"


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        ("a.b+tag(at)sub.example.org", "copr-team@redhat.com"),
        ("UPPER[AT]EXAMPLE.COM", "copr-team@redhat.com"),
        ("<name@domain.com>", "<copr-team@redhat.com>"),
        ("-name@domain.com", "-copr-team@redhat.com"),
        ("name@domain.com.extra", "copr-team@redhat.com"),
        ("one@a.com two@b.org", "copr-team@redhat.com copr-team@redhat.com"),
        ("a@x.com+second@y.org", "copr-team@redhat.comcopr-team@redhat.com"),
        ("one@a.com.+[at]a.com", "copr-team@redhat.comcopr-team@redhat.com"),
        ("name@.domain.com", "copr-team@redhat.com"),
        ("name@-domain.com", "copr-team@redhat.com"),
        ("name@domain-.com", "copr-team@redhat.com"),
        ("name@domain..com", "copr-team@redhat.com"),
        ("name@domain.toolong", "copr-team@redhat.com"),
        ("name@domain.12", "copr-team@redhat.com"),
        ("name@domain.com-.invalid", "copr-team@redhat.com"),
        ("name@domain_with_underscore.com", "copr-team@redhat.com"),
        ("example@invalid-domain.42", "copr-team@redhat.com"),
        ("name(at)domain-.com.", "copr-team@redhat.com."),
        ("name@localhost", "name@localhost"),
        ("@domain.com", "@domain.com"),
    ],
)
def test_email_boundaries_and_domains(artifact: str, expected: str) -> None:
    """Redact dotted address candidates while preserving separator boundaries."""
    assert sanitize_artifact(artifact) == expected


@pytest.mark.parametrize(
    "artifact",
    [
        "a" * 32_768 + "@invalid",
        "a." * 16_384 + "@invalid",
        "Fingerprint:" + " " * 32_768 + "Z",
    ],
)
def test_adversarial_sanitization_finishes_promptly(artifact: str) -> None:
    """Long failed candidates must not monopolize an analysis worker."""
    started = perf_counter()
    assert sanitize_artifact(artifact) == artifact
    assert perf_counter() - started < 2.0
