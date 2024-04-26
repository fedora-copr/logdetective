#!/usr/bin/env python
import argparse
import subprocess
import sys


NAME = "logdetective"


def call_poetry(bump_rule):
    """Call poetry tool to bump the version."""
    # Bump the version based on the bump rule
    # See poetry version --help
    ret = subprocess.run(["poetry", "version", "-s", bump_rule], capture_output=True, check=True)

    return ret.stdout.decode().strip()


def bump_version(target_version=None):
    """Bump pyproject.toml version."""
    if target_version:
        version = call_poetry(target_version)
    else:
        version = call_poetry("patch")

    print(f"Version bumped to {version}")

    return f"{version}"


def bump_patch_version(version):
    """Add +1 to the current version."""
    major, minor, patch = map(int, version.strip().strip('"').split("."))

    # Increment patch version
    patch += 1

    return f"{major}.{minor}.{patch}"


def commit(version):
    """Create a git commit for the new version."""
    subprocess.run(["git", "add", "./pyproject.toml"], check=True)
    subprocess.run(["git", "commit", "-m", f"New version {version}"], check=True)


def create_tag(version):
    """Create a git tag for the new version."""
    subprocess.run(["git", "tag", f"{NAME}-{version}"], check=True)


def run_checks():
    """Run checks on the code and check lock file."""
    ok = True
    print("Running poetry lock check")
    try:
        subprocess.run(["poetry", "check", "--lock"],
                       check=True,
                       stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError:
        print("Poetry lock file is not up to date!", file=sys.stderr)
        ok = False

    print("Running tests")
    try:
        subprocess.run(["tox"],
                       check=True,
                       stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError:
        print("Tests failed!", file=sys.stderr)
        ok = False

    return ok


if __name__ == "__main__":
    # Parse arguments
    parse = argparse.ArgumentParser("bumpver.py",
                                    description="Bump version of the logdetective project.")
    parse.add_argument("-t", "--tag", action="store_true",
                       dest="tag", help="Create also a git tag.")
    parse.add_argument("--version", action="store", default=None,
                       dest="version", help="""Set a specific version. Without this argument the
                       patch version bump will be automatically used.""")
    parse.add_argument("--no-check", action="store_false",
                       default=True, dest="check",
                       help="Disable checks on the code and lock file before commit.")
    args = parse.parse_args()

    # Run checks before version bump
    if args.check:
        if not run_checks():
            print("\nChecks returned error!", file=sys.stderr)
            print("If you want to bypass checks use '--no-check' argument.")
            sys.exit(1)

    # Bump the project version
    new_version = bump_version(args.version)
    commit(new_version)

    if args.tag:
        create_tag(new_version)
