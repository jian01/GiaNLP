"""
Configuration module
"""

import os
from multiprocessing import current_process


def is_ipython() -> bool:
    """
    Determines weather we are in a ipython environment
    :return: True if in ipython
    """
    try:  # pragma: no cover
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return True
    except NameError:
        return False


if is_ipython():
    __default_processes = 1
else:
    __default_processes = os.cpu_count()


def set_default_jobs(processes: int) -> None:
    """
    Set default jobs for parallelization where available
    :param processes: the amount of processes
    """
    global __default_processes
    __default_processes = processes


def get_default_jobs() -> int:
    """
    Gets the current default jobs for parallelization
    :return: the processes that will be used in parallel operations
    """
    if current_process().daemon:
        return 1
    return __default_processes
