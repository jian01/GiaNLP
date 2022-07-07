"""
Health utils for models
"""

from typing import Dict

import h5py
import numpy as np
import tensorflow as tf

from gianlp.logging import warning


def get_dependencies_signature() -> Dict[str, str]:
    """
    Get a dict containing all the versions of each important module for the library
    :return: a dict of versions
    """
    return {'tensorflow': tf.__version__, 'h5py': h5py.__version__,
            'numpy': np.version.version}


def warn_for_dependencies_signature(signature: Dict[str, str]) -> None:
    """
    Warns for dependencies mismatch
    :param signature: a signature of dependencies versions
    """
    if signature['tensorflow'] != tf.__version__:
        if signature['tensorflow'] < tf.__version__:
            warning(f"The tensorflow version used for serialization "
                    f"is {signature['tensorflow']} and higher than your"
                    f" tensorflow version ({tf.__version__}). "
                    f"This may cause a lot of hard-to-debug "
                    f"issues and is not recommended.")
        else:
            warning(f"Your tensorflow version ({tf.__version__}) is higher "
                    f"than the version used to serialize the model."
                    f"Tensorflow guarantees serialization backward compatibility, "
                    f"but there are some known issues.")
    if signature['h5py'] != h5py.__version__:
        warning(f"Your h5py version ({h5py.__version__}) differs from "
                f"the version used for serialization ({signature['h5py']}). "
                f"This may cause an issue.")
    if signature['numpy'] != np.version.version:
        warning(f"Your numpy version ({np.version.version}) differs from "
                f"the version used for serialization ({signature['numpy']}). "
                f"This may cause an issue.")
