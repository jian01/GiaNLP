"""Entrypoint for nox."""

import tempfile
import nox

@nox.session(
    python="3.9"
)
def tests_tensorflow_28(session):
    """Run all tests."""
    session.install("tensorflow==2.8.4", "protobuf<=3.20.1", silent=False)
    session.install(".", silent=False)
    session.install("-r", "./requirements.txt", silent=False)

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.10"
)
def tests_tensorflow_29(session):
    """Run all tests."""
    session.install("tensorflow==2.9.3", silent=False)
    session.install(".", silent=False)
    session.install("-r", "./requirements.txt", silent=False)

    cmd = ["pytest"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(
    python="3.10"
)
def tests_tensorflow_10(session):
    """Run all tests."""
    session.install("tensorflow==2.10.1", silent=False)
    session.install(".", silent=False)
    session.install("-r", "./requirements.txt", silent=False)

    cmd = ["pytest", "--cov=.", "--cov-report", "xml:/tmp/coverage.xml"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)

@nox.session(reuse_venv=True, python="3.7")
def cop(session):
    """Run all pre-commit hooks."""
    session.install(".", silent=False)
    session.install("-r", "./requirements.txt", silent=False)

    session.run("pre-commit", "install")
    session.run("pre-commit", "run")


@nox.session(reuse_venv=True, python="3.9")
def test_sphinx_build(session):
    """Build docs with sphinx."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        session.install(".", silent=False)
        session.install("-r", "./requirements.txt", silent=False)
        session.install("-r", "./docs/requirements.txt", silent=False)
        session.run("sphinx-build", "-E", "-n", "-b", "html", "docs", tmpdirname)
