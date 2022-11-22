"""
Utils for testing
"""
import os
import random
from typing import List, Any, Generator

import numpy as np
import tensorflow as tf
from tensorflow.random import set_seed as set_tf_seed

from gianlp.utils import Sequence

LOREM_IPSUM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Vestibulum eget nisl interdum, hendrerit ante quis, dignissim "
    "mauris. Aenean mauris nisi, tempor sed lobortis in, facilisis "
    "non ante. Maecenas volutpat, mauris a pharetra luctus, tellus "
    "enim porttitor velit, ac finibus mauris ipsum eget turpis. "
    "Cras commodo felis a ligula tincidunt scelerisque. Donec non "
    "imperdiet quam. Vivamus ac vulputate ligula. Duis malesuada "
    "sem enim, lacinia hendrerit purus hendrerit quis. Donec a "
    "turpis eu justo egestas tincidunt sed ac lorem. Curabitur non "
    "vehicula sem. Ut euismod dignissim nunc eget lacinia.\nDuis "
    "magna urna, finibus a tincidunt at, cursus eu odio. "
    "Pellentesque at nibh egestas, consequat lacus pulvinar, "
    "ultricies tellus. Curabitur porttitor consequat semper. Duis "
    "ultricies odio id dolor pulvinar, id auctor arcu imperdiet. "
    "Donec eu imperdiet lorem, eu euismod elit. Proin sed sodales "
    "sapien, et tristique mi. Maecenas at purus sed quam porttitor "
    "interdum. Cras pulvinar sed mauris gravida molestie. Maecenas "
    "sit amet nisl erat.\nCras ultrices, nisi in auctor mattis, "
    "risus lacus blandit quam, mollis scelerisque sem urna sit amet "
    ""
    "lorem. Etiam non lectus in ipsum porttitor laoreet. Vivamus a "
    "diam suscipit erat rhoncus lobortis. Aenean viverra, nibh non "
    "viverra lobortis, odio tellus pretium urna, eget luctus est "
    "nisi et turpis. Nullam aliquet blandit sodales. Etiam "
    "molestie, dolor at pretium vulputate, metus turpis volutpat "
    "diam, ac viverra orci nunc non dui. Praesent pellentesque leo "
    "ante, et mollis sem viverra pellentesque. Integer quis pretium "
    ""
    "lorem. Nulla vitae lectus lacinia arcu feugiat finibus eu eu "
    "turpis. Ut fermentum viverra est, tempor euismod ligula "
    "convallis auctor. Suspendisse laoreet eleifend dapibus. Mauris "
    ""
    "placerat nulla non magna scelerisque, vitae condimentum sapien "
    ""
    "pharetra.\nAenean eget erat a sapien sollicitudin faucibus sit "
    ""
    "amet a lorem. Fusce tempor dignissim facilisis. Aenean rutrum "
    "nec nibh id porta. Duis rutrum commodo felis, tempus semper "
    "tellus. Aliquam venenatis dui dictum neque congue, nec finibus "
    ""
    "odio pulvinar. Morbi lacinia erat pulvinar congue tincidunt. "
    "Curabitur imperdiet semper laoreet. Aenean mauris nulla, "
    "malesuada in lacinia at, faucibus sit amet nulla. Fusce "
    "facilisis sed libero a ultricies. Donec tincidunt efficitur "
    "massa, sed viverra nisl congue eu. In hac habitasse platea "
    "dictumst. Nulla ultricies laoreet pharetra. Etiam luctus felis "
    ""
    "quis posuere varius.\nUt quis risus sit amet purus elementum "
    "aliquam. Vestibulum sed aliquet odio. Nunc ultricies tempor "
    "enim, vehicula sagittis felis suscipit sed. Quisque ut "
    "fermentum velit. Nam quis nisi purus. Fusce ligula lectus, "
    "suscipit sit amet gravida et, porta eu metus. Sed tempus, "
    "lectus a pellentesque cursus, massa orci sollicitudin nibh, "
    "sed porta leo neque eget libero. Praesent vel ornare purus. "
    "Fusce vel magna eget mauris interdum rhoncus. Etiam vitae "
    "tincidunt lectus, non luctus lacus. Suspendisse et dictum "
    "erat. Donec nisl libero, imperdiet ut diam ac, sollicitudin "
    "faucibus nulla. Cras non magna vel dolor blandit scelerisque."
)


def set_seed(seed: int) -> None:
    """
    Sets seed for all randomness in python

    :param seed: the seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    set_tf_seed(seed)
    np.random.seed(seed)


def ensure_reproducibility(seed: int) -> None:
    """
    Ensures the tensorflow encironment is reproducible
    :param seed: the seed
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def read_sms_spam_dset():
    """
    Reads and returns sms spam dataset
    :return: a tuple containing a list with the text and a list with the labels
    """
    texts = []
    labels = []
    with open("tests/resources/SMSSpamCollection.txt", "r") as file:
        for line in file:
            if line:
                line = line.split("\t")
                texts.append(line[1])
                labels.append((1 if line[0] == "spam" else 0))
    return texts, labels


def accuracy(labels: List[int], preds: List[int]) -> float:
    """
    Computes accuracy

    :param labels: the true labels
    :param preds: predicted labels
    :return:
    """
    return (np.asarray(labels) == np.asarray(preds)).sum() / len(labels)


def split_tokenizer(text: str) -> List[str]:
    """
    Simple tokenizer
    :param text: the text
    :return: the text splited by spaces
    """
    return text.split(" ")


def newline_chunker(text: str) -> List[str]:
    """
    Simple newline chunker
    :param text: the text
    :return: the text splited by newlines
    """
    return text.split("\n")


def dot_chunker(text: str) -> List[str]:
    """
    Simple dot chunker
    :param text: the text
    :return: the text splited by dots
    """
    return text.split(".")


def generator_from_list(l: List[Any]) -> Generator[Any, None, None]:
    """
    Makes a generator from a list

    :param l: the list
    :return: a generator
    """
    for elem in l:
        yield elem


class SequenceFromList(Sequence):
    """
    Makes a sequence object form a list
    """

    def __init__(self, l: List[Any]):
        """

        :param l: the list to use
        """
        self.l = l

    def __len__(self) -> int:
        return len(self.l)

    def __getitem__(self, index: int) -> Any:
        return self.l[index]
