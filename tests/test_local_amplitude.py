import unittest
import os
import numpy as np
import sys
import torch
import math
import matplotlib.pyplot as plt

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import filter_utils.multistreamer as m
import filter_utils.local_amplitude as local


class TestLocalAmplitude(unittest.TestCase):
    def test_constructor_and_compute(self):
        # this is build on an older test for Multistreamer

        signal = torch.randn((2, 5000), dtype=torch.float32)
        l = local.LocalAmplitudeComputer()

        a = m.Multistreamer()
        b = a.split(signal)
        print("b sum = ", b.sum())
        c = a.merge(b)
        print("b sum = ", b.sum())
        d = l.compute(b)
        print("b.shape = {}, c.shape = {}, d.shape = {}".format(b.shape, c.shape, d.shape))
        print("b sum = ", b.sum())
        print("partial sum = ", b[0,0,5,:].sum().item())
        plt.plot(torch.arange(b.shape[-1]), b[0,0,5,:])
        print("d sum = ", d.sum())
        plt.plot(torch.arange(d.shape[-1]), d[0,5,:])
        plt.show()


        k = l.compute(torch.ones(1,2,1,2000) * 5.0)
        # sqrt(50) = sqrt(5^2 + 5^2).
        print("k is ", k)
        assert (k - math.sqrt(50.0)).abs().sum().item() < 0.1

        a = m.Multistreamer(4)
        b = a.split(signal)
        c = a.merge(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        a = m.Multistreamer(5)
        b = a.split(signal)
        c = a.merge(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        a = m.Multistreamer(full_padding = False)
        b = a.split(signal)
        c = a.merge(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

        signal = torch.randn((2, 500), dtype=torch.float64)
        a = m.Multistreamer(double_precision = True)
        b = a.split(signal)
        c = a.merge(b)
        print("b.shape = {}, c.shape = {}".format(b.shape, c.shape))

    def test_energy(self):
        # Testing that the forward computation has the expected effect on
        # the energy of a sinusoid.
        #
        # Note on why there is a fact of 1/(2*N) on the energy:
        #  The factor of 1/N is because there are N times fewer points than
        #  the input.
        #
        # There is a factor of 1/2 on the amplitude because: firstly, the input
        # sinusoid can be viewed as a sum of two complex exponentials which
        # are conjugates of each other, and we only keep one of them.
        # Each has 1/2 the amplitude, so the energy factor is 1/4.  But
        # then each complex exponential has a real and an imaginary part,
        # so when we sum the energy over these two parts the 1/4
        # turns into a 1/2.

        for N in range(1, 9):
            a = m.Multistreamer(N, full_padding = True)
            omega = math.pi / 2
            len = 50
            #signal = torch.zeros((1, len), dtype=torch.float32)
            signal = torch.randn((1, len), dtype=torch.float32)
            window_func = 0.5*torch.cos((torch.arange(len, dtype=torch.float32) - len/2) * (2*math.pi / len)) + 0.5
            signal = signal * window_func
            #print(window_func)

            print("N = ", N)
            input_energy = (signal * signal).sum().item()
            print("Energy of input signal is ", input_energy)
            s = a.split(signal)
            print("Energy of output signal is ", (s * s).sum().item(), ", compare ", input_energy / (2*N))
            t = a.merge(s)
            print("Energy of reconstructed signal is ", (t * t).sum().item())

            min_len = min(t.shape[1], signal.shape[1])
            print("min_len = ", min_len)

            print("Direction match of original with reconstructed signal is ",
                  ((((signal[:,0:min_len] * t[:,0:min_len]).sum()**2) / ((t * t).sum() * (signal * signal).sum())) ** 0.5).item())
            print("Energy ratio = ", (t * t).sum().item() / (signal * signal).sum().item())

            #plt.plot(signal[0,:])
            #plt.plot(t[0,:])
            #plt.show()

            arr = []
            for n in range(N):
                real_energy = (s[0,0,n,:] ** 2).sum().item()
                im_energy = (s[0,1,n,:] ** 2).sum().item()
                arr.append('%d=%.2f+%.2fi' % (n ,real_energy, im_energy))

            print("Division of energy by N and by (real,im) is: ",
                  " ".join(arr))




if __name__ == "__main__":
    unittest.main()

