# To be run with python3

"""
This module defines an object that can be used for signal resampling.
It has a torch dependency because it does the resampling via 1d convolution.
"""


import numpy as np
from . import filters
import math
import torch


def gcd(a, b):
    """ Return the greatest common divisor of a and b"""
    assert isinstance(a, int) and isinstance(b, int)
    if b == 0:
        return a
    else:
        return gcd(b, a % b)

class Resampler:
    """
    This object should ideally be initialized once and used many times,
    but the construction time shouldn't be excessive.
    Please read the documentation carefully!
    """

    def __init__(self,
                 input_sr, output_sr,
                 num_zeros = 64, cutoff_ratio = 0.95,
                 double_precision = False):
        """
        This creates an object that can apply a symmetric FIR filter
        based on torch.nn.functional.conv1d.

        Args:
          input_sr:  The input sampling rate, AS A SMALL INTEGER..
              does not have to be the real sampling rate but should
              have the correct ratio with output_sr.
          output_sr:  The output sampling rate, AS A SMALL INTEGER.
              It is the ratio with the input sampling rate that is
              important here.
          num_zeros: The number of zeros per side in the (sinc*hanning-window)
              filter function.  More->more accurate, but 64 is already
              quite a lot.

        You can think of this algorithm as dividing up the signals
        (input,output) into blocks where there are `input_sr` input
        samples and `output_sr` output samples.  Then we treat it
        using convolutional code, imagining there are `input_sr`
        input channels and `output_sr` output channels per time step.

        """
        d = gcd(input_sr, output_sr)
        input_sr, output_sr = input_sr / d, output_sr / d


        # We work out the size of the filter patch in the convolution as
        # follows... note, this is a little approximate.
        #
        # We want `num_zeros` zeros of the sinc function on either side.
        # Approximating cutoff_ratio == 1, the sinc function has a zero
        # once per sample
        # of whichever of the input or output signal has the lower sampling
        # rate

 approximately have a zero every sample
        # at which




        filters.check_is_filter(filter)
        (f, i) = filter
        filt_len = f.shape[0]
        assert filt_len == i * 2 + 1  # check it's symmetric
        dtype = (torch.float64 if double_precision else torch.float32)
        # the shape is (out_channels, in_channels, width),
        # where out_channels and in_channels are both 1.
        self.filt = torch.tensor(f, dtype=dtype).view(1, 1, filt_len)
        self.padding = i

    def apply(self, input):
        """
        Apply the FIR filter, and return a result of the same shape

        Args:
         input: a torch.Tensor with dtype torch.float64 if double_precision=True was
         supplied to the constructor, else torch.float32.
         There must be 2 axes, interpreted as (minibatch_size, sequence_length)

        Return:  Returns a torch.Tensor with the same dtype and dim as the
        input.
        """

        # input.unsqueeze(1) changes dim from (minibatch_size, sequence_length) to
        # (minibatch_size, num_channels=1, sequence_length)
        # the final squeeze(1) removes the num_channels=1 axis
        return torch.nn.functional.conv1d(input.unsqueeze(1), self.filt,
                                         padding=self.padding).squeeze(1)



