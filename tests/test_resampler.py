import unittest
import os
import numpy as np
import sys
import torch
import math
import matplotlib.pyplot as plt

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import lilfilter
import librosa

class TestResampler(unittest.TestCase):
    def test_constructor_and_forward(self):
        # test constructor doesn't crash with various args,
        # and that the forward computation does not crash.

        a = torch.randn((2, 500), dtype=torch.float32)

        res = lilfilter.Resampler(5, 4, torch.float32)
        b = res.resample(a)
        print("a.shape = {}, b.shape = {}".format(a.shape, b.shape))

        res2 = lilfilter.Resampler(5, 4, torch.float64)
        a = a.double()
        b = res2.resample(a)
        print("a.shape = {}, b.shape = {}".format(a.shape, b.shape))

    def test_energy(self):
        # Testing that the process of downsampling and then upsampling again has the expected effect on
        # the energy of a sinusoid that is comfortably below the filter cutoff, i.e. the
        # round trip process should preserve that energy.

        for pair in [ (2,3), (4,1), (5,4), (7,8) ]:
            n1, n2 = pair
            print("n1,n2 = {},{}".format(n1, n2))

            nyquist = math.pi * min(n2 / n1, 1)
            omega = 0.85 * nyquist  # enough less than nyquist that energy should be preserved.
            length = 500
            signal = torch.cos(torch.arange(length, dtype=torch.float32)*omega).unsqueeze(0)
            window_func = 0.5*torch.cos((torch.arange(length, dtype=torch.float32) - length/2) * (2*math.pi / length)) + 0.5
            signal = signal * window_func

            input_energy = (signal * signal).sum().item()
            print("Energy of input signal is ", input_energy)

            s = lilfilter.resample(signal, n1, n2)
            s_rosa = torch.tensor(librosa.core.resample(signal.numpy(), n1, n2))

            t = lilfilter.resample(s, n2, n1)
            t_rosa = torch.tensor(librosa.core.resample(s_rosa.numpy(), n2, n1))

            length = min(t.shape[1], signal.shape[1])

            sig1 = signal[:,:length]
            sig2 = t[:,:length]
            prod1 = (sig1 * sig1).sum()
            prod2 = (sig2 * sig2).sum()
            prod3 = (sig1 * sig2).sum()

            length_rosa = min(t_rosa.shape[1], signal.shape[1])
            sig1_rosa = signal[:,:length_rosa]
            sig2_rosa = t_rosa[:,:length_rosa]
            prod1_rosa = (sig1_rosa * sig1_rosa).sum()
            prod2_rosa = (sig2_rosa * sig2_rosa).sum()
            prod3_rosa = (sig1_rosa * sig2_rosa).sum()


            print("The following [lilfilter] numbers should be the same: {},{},{}".format(
                prod1, prod2, prod3))

            r1 = prod1 / prod2
            r2 = prod2 / prod3
            assert( abs(r1-1.0) < 0.001 and abs(r2-1.0) < 0.001)

            print("The following [librosa] numbers should be the same: {},{},{}".format(
                prod1_rosa, prod2_rosa, prod3_rosa))

            r1_rosa = prod1_rosa / prod2_rosa
            r2_rosa = prod2_rosa / prod3_rosa
            #assert( abs(r1_rosa-1.0) < 0.001 and abs(r2_rosa-1.0) < 0.001)


            #plt.plot(np.arange(length), sig1.squeeze(0).numpy())
            #plt.plot(np.arange(length), sig2.squeeze(0).numpy())
            #plt.show()

            print("Length of input signal is {}, downsampled {}, reconstructed {}, librosa-reconstructed".format(
                    signal.shape[-1], s.shape[-1], t.shape[-1], t_rosa.shape[-1]))




if __name__ == "__main__":
    unittest.main()
