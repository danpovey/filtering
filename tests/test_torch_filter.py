import unittest
import os
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import lilfilter.filters as F
import lilfilter.torch_filter as T


class TestTorchFilter(unittest.TestCase):
    def test1(self):
        filt = F.gaussian_filter(5.0)
        t = T.SymmetricFirFilter(filt)
        len = 500
        a = torch.randn(1, len)
        plt.plot(torch.arange(len), a.squeeze(0))
        b = t.apply(a)
        plt.plot(torch.arange(len), b.squeeze(0))
        plt.show()


if __name__ == "__main__":
    unittest.main()

