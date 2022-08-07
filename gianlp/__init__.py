"""
GiaNLP module
"""
import os

_TENSORFLOW_REQUIRED_MIN_VERSION = "2.3.0"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise ModuleNotFoundError(f"tensorflow or tensorflow-gpu >={_TENSORFLOW_REQUIRED_MIN_VERSION} is needed")

if tf.__version__ < _TENSORFLOW_REQUIRED_MIN_VERSION:
    raise ModuleNotFoundError(
        f"tensorflow or tensorflow-gpu >={_TENSORFLOW_REQUIRED_MIN_VERSION} is needed. "
        f"You have version {tf.__version__}."
    )

if hasattr(tf, "get_logger"):
    tf.get_logger().setLevel("ERROR")

import sys

if "absl.logging" in sys.modules:
    import absl.logging

    absl.logging.set_verbosity("error")
    absl.logging.set_stderrthreshold("error")

from gianlp.logging import warning

warning("GiaNLP disables all tensorflow-related logging")

__version__ = "0.0.1"
