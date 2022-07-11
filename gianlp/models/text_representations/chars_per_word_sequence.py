"""
Module for char sequence per word input
"""

from typing import List, Optional, Callable

# pylint: disable=no-name-in-module
from tensorflow.keras.models import Model

# pylint: enable=no-name-in-module

from gianlp.models.text_representations.char_embedding_sequence import CharEmbeddingSequence
from gianlp.models.text_representations.per_chunk_sequencer import PerChunkSequencer
from gianlp.models.text_representations.text_representation import TextRepresentation


class CharPerWordEmbeddingSequence(TextRepresentation):
    """
    Char per word sequence. A wrapper for instantiating a per chunk sequencer.

    :var _chunker: function used for chunking the texts
    :var _sequencer: text input use for sequencing each chunk
    :var _chunking_maxlen: the maximum length in chunks for a text
    """

    _keras_model: Optional[Model]
    _chunker: Callable[[str], List[str]]
    _sequencer: TextRepresentation
    _chunking_maxlen: int

    def __new__(
        cls,
        tokenizer: Callable[[str], List[str]],
        embedding_dimension: int = 256,
        word_maxlen: int = 30,
        char_maxlen: int = 12,
        min_freq_percentile: int = 5,
        random_state: int = 42,
    ):
        """

        :param tokenizer: a tokenizer function that transforms each string into a list of string tokens
                        the function must support serialization through pickle
        :param embedding_dimension: The char embedding dimension
        :param word_maxlen: the max length for word sequences
        :param char_maxlen: the max length for chars within a word
        :param min_freq_percentile: minimum percentile of the frequency for keeping a char.
                                    If a char has a frequency lower than this percentile it
                                    would be treated as unknown.
        :param random_state: random seed
        :returns: a PerChunkSequencer with your tokenizer and a CharEmbeddingSequence as sequencer
        """
        char_embedding = CharEmbeddingSequence(
            embedding_dimension=embedding_dimension,
            sequence_maxlen=char_maxlen,
            min_freq_percentile=min_freq_percentile,
            random_state=random_state,
        )
        return PerChunkSequencer(char_embedding, tokenizer, chunking_maxlen=word_maxlen)
