import unittest
import os
import numpy as np
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import filter_utils.filter_function as f


class TestFilterFunction(unittest.TestCase):
    def test1(self):
        # test symmetry
        self.assertTrue(f.get_function_at(-1) == f.get_function_at(1))

        self.assertTrue(f.get_function_at(f.S) == 0)
        self.assertTrue(f.get_function_at(-f.S) == 0)
        self.assertTrue(f.get_function_at(-2.0*f.S) == 0)
        self.assertTrue(f.get_function_at(f.S - 0.01) != 0)
        self.assertTrue(f.get_function_at(-f.S+ 0.01) != 0)



if __name__ == "__main__":
    unittest.main()

