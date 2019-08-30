import unittest
import os
import numpy as np
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from filter_utils.filters import *


class TestIsFilter(unittest.TestCase):
    def test1(self):
        with self.assertRaises(TypeError):
            check_is_filter(1)
        with self.assertRaises(ValueError):
            check_is_filter((1,2,3))
        check_is_filter((np.zeros(3), 1))
        with self.assertRaises(ValueError):
            check_is_filter((np.zeros(3), 3))


class TestApplyFilter(unittest.TestCase):
    def test1(self):
        pos = np.random.randint(0,4)
        filt = (np.zeros(4), pos)
        filt[0][pos] = 1.0

        signal = np.arange(0, 10, dtype=np.float32)
        print("pos = ", pos)
        filtered = apply_filter(filt, signal)
        print("Signal = {}, filtered = {}".format(signal, filtered))
        self.assertTrue(np.array_equal(filtered, signal))

        signal = np.arange(0, 30, dtype=np.float32).reshape((10,3))
        axis = np.random.randint(0,2)

        filtered = apply_filter(filt, signal, axis=axis)
        print("Signal = {}, filtered = {}".format(signal, filtered))
        self.assertTrue(np.array_equal(filtered, signal))


    def test_energy(self):
        i = 2000
        signal = np.random.randn(i)
        filt = low_pass_filter(0.25, 10)
        # 0.25 is half of (nyquist == 0.5) so should preserve half the
        # energy of the signal.
        old_energy = (signal ** 2).sum()
        new_energy = (apply_filter(filt, signal) ** 2).sum()
        proportion = new_energy / old_energy
        print("Proportion, should be 0.5, is: ", proportion)
        assert(proportion > 0.4 and proportion < 0.6)

    def test_energy2(self):
        i = 2000
        signal = np.random.randn(i)
        filt = high_pass_filter(0.5 * 0.333, 10)

        old_energy = (signal ** 2).sum()
        new_energy = (apply_filter(filt, signal) ** 2).sum()
        proportion = new_energy / old_energy
        print("Proportion, should be 0.666, is: ", proportion)
        assert(proportion > 0.55 and proportion < 0.77)



if __name__ == "__main__":
    unittest.main()

