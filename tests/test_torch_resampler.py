import unittest
import os
import numpy as np
import sys
import torch
import math
import matplotlib.pyplot as plt

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import lilfilter.torch_resampler as r


class TestResampler(unittest.TestCase):
    def test_constructor_and_forward(self):
        # test constructor doesn't crash with various args,
        # and that the forward computation does not crash.

        a = torch.randn((2, 500), dtype=torch.float32)

        res = r.Resampler(5, 4, torch.float32)
        b = res.resample(a)
        print("a.shape = {}, b.shape = {}".format(a.shape, b.shape))

        res2 = r.Resampler(5, 4, torch.float64)
        a = a.double()
        b = res2.resample(a)
        print("a.shape = {}, b.shape = {}".format(a.shape, b.shape))

    def test_energy(self):
        # Testing that the process of downsampling and then upsampling again has the expected effect on
        # the energy of a sinusoid that is comfortably below the filter cutoff, i.e. the
        # round trip process should preserve that energy.

        for pair in [ (2,3), (4,1), (5,4), (7,8) ]:
            n1, n2 = pair

            a = r.Resampler(n1, n2, dtype=torch.float32)

            nyquist = math.pi * min(n2 / n1, 1)
            omega = 0.95 * nyquist  # enough less than nyquist that energy should be preserved.
            length = 500
            signal = torch.cos(torch.arange(length, dtype=torch.float32)*omega).unsqueeze(0)
            window_func = 0.5*torch.cos((torch.arange(length, dtype=torch.float32) - length/2) * (2*math.pi / length)) + 0.5
            signal = signal * window_func

            input_energy = (signal * signal).sum().item()
            print("Energy of input signal is ", input_energy)
            s = a.resample(signal)


            b = r.Resampler(n2, n1, dtype=torch.float32)
            t = b.resample(s)

            length = min(t.shape[1], signal.shape[1])

            sig1 = signal[:,:length]
            sig2 = t[:,:length]


            prod1 = (sig1 * sig1).sum()
            prod2 = (sig2 * sig2).sum()
            prod3 = (sig1 * sig2).sum()


            print("The following numbers should be the same: {},{},{}".format(
                prod1, prod2, prod3))

            r1 = prod1 / prod2
            r2 = prod2 / prod3
            assert( abs(r1-1.0) < 0.001 and abs(r2-1.0) < 0.001)


            #plt.plot(np.arange(length), sig1.squeeze(0).numpy())
            #plt.plot(np.arange(length), sig2.squeeze(0).numpy())
            #plt.show()

            print("Length of input signal is {}, downsampled {}, reconstructed {}".format(
                    signal.shape[-1], s.shape[-1], t.shape[-1]))




if __name__ == "__main__":
    unittest.main()
