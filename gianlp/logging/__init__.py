"""
Logging module
"""

import logging as _logging
import sys as _sys
from logging import INFO

_logger = None


def get_logger() -> _logging:
    """Return logger instance. 'Inspired' 👀 by tensorflow get_logger

    :return: A python logger instance"""
    global _logger

    if _logger:
        return _logger

    logger = _logging.getLogger("nlp_builder")

    if not _logging.getLogger().handlers:  # pragma: no cover
        # Determine whether we are in an interactive environment
        _interactive = False
        try:
            # This is only defined in interactive shells.
            if _sys.ps1:
                _interactive = True
        except AttributeError:
            # Even now, we may be in an interactive shell with `python -i`.
            _interactive = _sys.flags.interactive

        # If we are in an interactive environment (like Jupyter), set loglevel
        # to INFO and pipe the output to stdout.
        if _interactive:
            logger.setLevel(INFO)
            _logging_target = _sys.stdout
        else:
            _logging_target = _sys.stderr

        # Add the output handler.
        _handler = _logging.StreamHandler(_logging_target)
        _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
        logger.addHandler(_handler)

    _logger = logger
    return _logger


def warning(msg: str) -> None:
    """
    Logs a warning message
    :param msg: the message to log
    """
    logger = get_logger()
    logger.warning(msg)
