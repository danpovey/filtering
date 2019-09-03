# To be run with python3

"""
This module defines an object that can be used for upsampling and downsampling
of signals.  Note: unlike ./filters.py, this object has a torch dependency.
(It uses ./filters.py for initialization though.)
"""


import numpy as np
from . import filters
import math
import torch

class SymmetricFirFilter:
    """
    This class is used for applying symmetric FIR filters using torch 1d
    convolution.
    """

    def __init__(self, filter,
                 double_precision = False):
        """
        This creates an object that can apply a symmetric FIR filter
        based on torch.nn.functional.conv1d.

        Args:
        filter:  A filter as defined in ./filters.py.  Expected to be
               symmetric, i.e. its (i*2)+1 must equal its filter
               length.
        double_precision:  If true, we'll use float64 for the filter; else float32.

        padding:  Must be 'zero' or 'reflect'.  If 'zero', the output is
          as if we padded the signal with zeros to get the same length.
          If 'reflect', it's as if we reflected the signal at 0.5 of a
          sample past the first and last sample.
        """
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



