"""
Tests internal text input wrapper
"""

import unittest
from gianlp.types import TextsInputWrapper


class TestTextInputWrapper(unittest.TestCase):
    """
    Tests internal text input wrapper
    """

    def test_fails_when_nesting_text_wrappers(self) -> None:
        """
        Tests error when initializing with another wrapper
        """

        with self.assertRaises(ValueError):
            TextsInputWrapper(TextsInputWrapper(["asd", "bcd"]))

    def test_simple_input_slice(self) -> None:
        """
        Tests simple input slicing
        """
        wrapper = TextsInputWrapper([str(i) for i in range(10)])
        self.assertEqual(wrapper[3:7].to_texts_inputs(), ["3", "4", "5", "6"])

    def test_multi_input_slice(self) -> None:
        """
        Tests multiple input slicing
        """
        wrapper = TextsInputWrapper({"t1": [str(i) for i in range(10)], "t2": [str(i) for i in range(10, 20)]})
        self.assertEqual(wrapper[3:7].to_texts_inputs(), {"t1": ["3", "4", "5", "6"], "t2": ["13", "14", "15", "16"]})

    def test_multi_input_int_key_error(self) -> None:
        """
        Tests error when indexing multi input by int
        """
        wrapper = TextsInputWrapper({"t1": [str(i) for i in range(10)], "t2": [str(i) for i in range(10, 20)]})
        with self.assertRaises(KeyError):
            wrapper[0]

    def test_simple_input_str_key_error(self) -> None:
        """
        Tests error when indexing simple input by str
        """
        wrapper = TextsInputWrapper([str(i) for i in range(10)])
        with self.assertRaises(KeyError):
            wrapper["t1"]

    def test_add_multi_with_simple_error(self) -> None:
        """
        Test error when adding multi text and simple text inputs
        """
        wrapper1 = TextsInputWrapper([str(i) for i in range(10)])
        wrapper2 = TextsInputWrapper({"t1": [str(i) for i in range(10)], "t2": [str(i) for i in range(10, 20)]})
        with self.assertRaises(ValueError):
            wrapper1 + wrapper2

    def test_add_multis_with_different_keys_error(self) -> None:
        """
        Test error when adding multi text inputs with different text names
        """
        wrapper1 = TextsInputWrapper({"t1": [str(i) for i in range(10)], "t3": [str(i) for i in range(10, 20)]})
        wrapper2 = TextsInputWrapper({"t1": [str(i) for i in range(10)], "t2": [str(i) for i in range(10, 20)]})
        with self.assertRaises(ValueError):
            wrapper1 + wrapper2
