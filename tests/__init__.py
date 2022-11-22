"""
Test module
"""
from gianlp.config import set_default_jobs
from tests.utils import ensure_reproducibility

set_default_jobs(1)
ensure_reproducibility(42)

__version__ = "utils"
