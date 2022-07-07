"""
NLP Builder module
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise ModuleNotFoundError("tensorflow or tensorflow-gpu >=2.3.0 is needed")

if not tf.__version__ >= '2.3.0':
    raise ModuleNotFoundError(f"tensorflow or tensorflow-gpu >=2.3.0 is needed. You have version {tf.__version__}.")

if hasattr(tf, "get_logger"):
    tf.get_logger().setLevel("ERROR")

import sys

if "absl.logging" in sys.modules:
    import absl.logging

    absl.logging.set_verbosity("error")
    absl.logging.set_stderrthreshold("error")

from gianlp.logging import warning

warning("The NLP builder disables all tensorflow-related logging")

__version__ = "0.0.1"
