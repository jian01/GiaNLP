"""
Tests internal model outputs wrapper
"""

import unittest

import numpy as np

from gianlp.types import ModelOutputsWrapper


class TestTextInputWrapper(unittest.TestCase):
    """
    Tests internal model outputs wrapper
    """

    def test_slice_simple_output(self) -> None:
        """
        Test indexing with a slice having simple outputs
        """
        outputs = ModelOutputsWrapper(np.asarray([0, 1, 1, 0] * 20))
        self.assertEqual(outputs[-2:].to_model_outputs().tolist(), [1, 0])

    def test_list_index_simple_output(self) -> None:
        """
        Test indexing with a list having simple outputs
        """
        outputs = ModelOutputsWrapper(np.asarray([0, 1, 1, 0] * 20))
        self.assertEqual(outputs[[0, 2, 4, 6, 8, 10]].to_model_outputs().tolist(), [0, 1] * 3)

    def test_add_simple_outputs(self) -> None:
        """
        Add two simple outputs
        """
        out1 = ModelOutputsWrapper(np.asarray([0, 1, 1, 0] * 20))
        out2 = ModelOutputsWrapper(np.asarray([1, 0, 0, 1] * 20))
        concat = out1[[0, 2, 4, 6, 8, 10]] + out2[[0, 2, 4, 6, 8, 10]]
        self.assertEqual(concat.to_model_outputs().tolist(), [0, 1] * 3 + [1, 0] * 3)

    def test_add_multiple_outputs_different_length_fails(self) -> None:
        """
        Add two multiple outputs with different length raises error
        """
        out1 = ModelOutputsWrapper([np.asarray([0, 1] * 20), np.asarray([0, 1] * 20), np.asarray([0, 1] * 20)])
        out2 = ModelOutputsWrapper([np.asarray([1, 0] * 20), np.asarray([0, 1] * 20)])
        with self.assertRaises(ValueError):
            out1 + out2

    def test_add_multiple_output_with_simple_fails(self) -> None:
        """
        Add multiple output with simple fails
        """
        out1 = ModelOutputsWrapper(np.asarray([0, 1] * 20))
        out2 = ModelOutputsWrapper([np.asarray([1, 0] * 20), np.asarray([0, 1] * 20)])
        with self.assertRaises(ValueError):
            out1 + out2

    def test_cannot_double_wrap(self) -> None:
        """
        Wrapping a wrapper throws exception
        """
        out = ModelOutputsWrapper(np.asarray([0, 1] * 20))
        with self.assertRaises(ValueError):
            ModelOutputsWrapper(out)

    def test_invalid_type_key(self) -> None:
        """
        Key of invalid type throws exception
        """
        out = ModelOutputsWrapper(np.asarray([0, 1] * 20))
        with self.assertRaises(KeyError):
            out["asd"]
