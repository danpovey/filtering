import unittest
import os
import numpy as np
import sys
import torch
import math
import matplotlib.pyplot as plt

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import filter_utils.resampler as r


class TestResampler(unittest.TestCase):
    def test_constructor_and_forward(self):
        # test constructor doesn't crash with various args,
        # and that the forward computation does not crash.

        signal = torch.randn((2, 500), dtype=torch.float32)

        a = r.Resampler(5)
        b = a.downsample(signal)
        c = a.upsample(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        a = r.Resampler(4)
        b = a.downsample(signal)
        c = a.upsample(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        a = r.Resampler(10)
        b = a.downsample(signal)
        c = a.upsample(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        a = r.Resampler(10, full_padding = False)
        b = a.downsample(signal)
        c = a.upsample(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        signal = torch.randn((2, 500), dtype=torch.float64)
        a = r.Resampler(10, double_precision = True)
        b = a.downsample(signal)
        c = a.upsample(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

    def test_energy(self):
        # Testing that the process of downsampling and then upsampling again has the expected effect on
        # the energy of a sinusoid that is comfortably below the filter cutoff, i.e. the
        # round trip process should preserve that energy.

        for N in range(2, 9):
            a = r.Resampler(N, full_padding = True, num_zeros = 4)
            omega = math.pi / (1.5 * N)
            len = 500
            #signal = torch.zeros((1, len), dtype=torch.float32)
            signal = torch.cos(torch.arange(len, dtype=torch.float32)*omega).unsqueeze(0)
            window_func = 0.5*torch.cos((torch.arange(len, dtype=torch.float32) - len/2) * (2*math.pi / len)) + 0.5
            signal = signal * window_func

            #print(window_func)

            print("N = ", N)
            input_energy = (signal * signal).sum().item()
            print("Energy of input signal is ", input_energy)
            s = a.downsample(signal)
            #  The factor of 1/N in the expected energy is because there are N times
            #  fewer points than the input.
            print("Energy of output signal is ", (s * s).sum().item(), ", compare ", input_energy / N)
            t = a.upsample(s)
            print("Energy of reconstructed signal is ", (t * t).sum().item())


            plt.plot(torch.arange(len), signal.squeeze(0))
            plt.plot(torch.arange(t.shape[-1]), t.squeeze(0))
            plt.show()


            print("Length of input signal is {}, downsampled {}, reconstructed {}".format(
                    signal.shape[-1], s.shape[-1], t.shape[-1]))


            min_len = min(t.shape[1], signal.shape[1])
            print("min_len = ", min_len)

            print("Direction match of original with reconstructed signal is ",
                  ((((signal[:,0:min_len] * t[:,0:min_len]).sum()**2) / ((t * t).sum() * (signal * signal).sum())) ** 0.5).item())
            print("Energy ratio = ", (t * t).sum().item() / (signal * signal).sum().item())





if __name__ == "__main__":
    unittest.main()

