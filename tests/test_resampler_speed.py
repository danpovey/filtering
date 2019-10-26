import unittest
import os
import numpy as np
import sys
import torch
import math
import matplotlib.pyplot as plt
import time
import librosa  # For comparison of speed

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import lilfilter.resampler as res


class TestSpeed(unittest.TestCase):
    def test(self):
        ## NOTE: for comparison with librosa we fix the num_zeros to 64 which is
        ## the default with librosa's "kaiser_best" resampling.

        signal = torch.randn((2, 10000), dtype=torch.float32)


        r_down = res.Resampler(2, 1, dtype=torch.float32, num_zeros = 64)
        r_up = res.Resampler(1, 2, dtype=torch.float32, num_zeros = 64)
        begin = time.perf_counter()

        print("our torch-filter filter size = {}", r_down.weights.shape)
        for i in range(10):
            b = r_down.resample(signal)
            c = r_up.resample(b)
            print("T:b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        print("Elapsed time for our torch-based resampler: {}".format(time.perf_counter() - begin))

        begin = time.perf_counter()
        signal = np.asfortranarray(signal.numpy())
        for i in range(10):
            b = librosa.core.resample(signal, 2, 1)
            c = librosa.core.resample(b, 1, 2)
            print("R:b.shape = {}, c.shape = {}".format(b.shape, c.shape))
        print("Elapsed time for librosa resampler: {}".format(time.perf_counter() - begin))


if __name__ == "__main__":
    unittest.main()
