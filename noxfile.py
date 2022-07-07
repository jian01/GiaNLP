"""Entrypoint for nox."""

import tempfile
import nox


@nox.session(
    python="3.7"
)
def tests_tensorflow_23(session):
    """Run all tests."""
    session.install("tensorflow==2.3.0")
    session.install(".")
    session.install("-r", "./requirements.txt")

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.8"
)
def tests_tensorflow_24(session):
    """Run all tests."""
    session.install("tensorflow==2.4.0")
    session.install(".")
    session.install("-r", "./requirements.txt")

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.9"
)
def tests_tensorflow_25(session):
    """Run all tests."""
    session.install("tensorflow==2.5.0")
    session.install(".")
    session.install("-r", "./requirements.txt")

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.9"
)
def tests_tensorflow_26(session):
    """Run all tests."""
    session.install("tensorflow==2.6.0")
    session.install(".")
    session.install("-r", "./requirements.txt")

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.9"
)
def tests_tensorflow_27(session):
    """Run all tests."""
    session.install("tensorflow==2.7.0")
    session.install(".")
    session.install("-r", "./requirements.txt")

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.9"
)
def tests_tensorflow_28(session):
    """Run all tests."""
    session.install("tensorflow==2.8.0")
    session.install(".")
    session.install("-r", "./requirements.txt")

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.9"
)
def tests_tensorflow_latest(session):
    """Run all tests."""
    session.install("tensorflow")
    session.install(".")
    session.install("-r", "./requirements.txt")

    cmd = ["pytest", "--cov=.", "--cov-report", "xml", "--cov-fail-under=97"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(reuse_venv=True, python="3.9")
def cop(session):
    """Run all pre-commit hooks."""
    session.install(".")
    session.install("-r", "./requirements.txt")

    session.run("pre-commit", "install")
    session.run("pre-commit", "run")


@nox.session(reuse_venv=True, python="3.9")
def test_sphinx_build(session):
    """Build docs with sphinx."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        session.install(".")
        session.install("sphinx")
        session.run("sphinx-build", "-W", "-E", "-n", "-b", "html", "docs", tmpdirname)
