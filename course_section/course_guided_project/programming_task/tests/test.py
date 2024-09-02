import os
import unittest
from unittest.mock import patch

from invisible_main import say_bye
from main import invoke_say_bye


class TestAddFunction(unittest.TestCase):
    test_cases = [2, 4, 20, 1900, 400, 2004, 2000, 2001]

    @staticmethod
    def correct_invoke_say_bye(how_many_times: int):
        return os.linesep.join([say_bye() for _ in range(how_many_times)])

    @patch('builtins.print')
    def test_parametrized(self, mock_print):
        for how_many_times in self.test_cases:
            with self.subTest(how_many_times):
                correct_solution = self.correct_invoke_say_bye(how_many_times)
                invoke_say_bye(how_many_times)
                mock_print.assert_called_once_with(correct_solution)
                mock_print.reset_mock()
