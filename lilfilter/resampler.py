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

class Resampler:

    def __init__(self, N, num_zeros = 7,
                 filter_cutoff_ratio = 0.95,
                 full_padding = False,
                 double_precision = False):
        """
        This creates an object which can be used for both upsampling and
        downsampling of signals.  This involves creating a low-pass filter with
        the appropriate cutoff.

        Args:
             N (int):   The downsampling or upsampling  ratio.  For example,
                     4 would mean we downsample or upsample by a factor of 4.
                     Must be > 1.

             num_zeros (int): The number of zeros in the filter function..
                     a larger number will give a sharper cutoff, but will be
                     slower.

             filter_cutoff_ratio (float):  Determines where we place the
                     cutoff of the filter used for upsampling and
                     downsampling, relative to the Nyquist of the lower
                     of the two frequencies.  Must be >0.5 and <1.0.

             full_padding (bool):  If true, will pad on each side with
                     (filter_width - 1) which ensures that a sufficiently-low-pass
                     signal that's upsampled and then downsampled will
                     undergo the round trip with minimal end effects.
                     If false, we pad with filter_width when downsampling,
                     which will give a signal length closer to
                     input_signal_length / N and enables easier
                     mapping of time offsets,(without worrying about time
                     offsets).

             double_precision:  If true, will use torch.float64 for the filter
                     (and expect this for the input); else will use torch.float32.
        """
        self.N = N
        if not (isinstance(N, int) and isinstance(num_zeros, int) and isinstance(filter_cutoff_ratio, float)):
            raise TypeError("One of the args has the wrong type")
        if N <= 1 or num_zeros < 2:
            raise ValueError("Require N > 1 and num_zeros > 1")
        if not (filter_cutoff_ratio > 0.5 and filter_cutoff_ratio < 1.0):
            raise ValueError("Invalid number for filter_cutoff_ratio: ",
                             filter_cutoff_ratio)

        self.dtype = (torch.float64 if double_precision else torch.float32)

        # f is a numpy array.  i is its central index, not really needed.
        (f, i) = filters.low_pass_filter(filter_cutoff_ratio / (N * 2),
                                         num_zeros = num_zeros)


        f_len = f.shape[0]

        # self.filter is a torch.Tensor whose dimension is interpreted
        # as (out_channels, in_channels, width) where out_channels and
        # in_channels are both 1.
        self.forward_filter = torch.tensor(f).view(1, 1, f_len).to(self.dtype)

        self.backward_filter = self.forward_filter * N

        if full_padding:
            self.padding = f_len - 1
        else:
            self.padding = (f_len - 1) // 2



    def downsample(self, input):
        """
        This downsamples the signal `input` and returns the result.
        Args:
         input (torch.Tensor): A Tensor with shape (minibatch_size, signal_length),
              and dtype torch.float64 if double_precision to constructor was true,
              else torch.float32.

        Return:
            Returns a torch.Tensor with shape (minibatch_size, reduced_signal_length).
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("Expected input to be torch.Tensor, got ",
                            type(input))
        if not (input.dtype == self.dtype):
            raise TypeError("Expected input tensor to have dtype {}, got {}".format(
                    self.dtype, input.dtype))

        # The squeeze and unsqueeze are to insert a dim for num_channels == 1.
        return torch.nn.functional.conv1d(input.unsqueeze(1),
                                          self.forward_filter,
                                          stride=self.N,
                                          padding=self.padding).squeeze(1)

    def upsample(self, input):
        """
        This upsamples the signal `input`.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError("Expected input to be torch.Tensor, got ",
                            type(input))
        if not (input.dtype == self.dtype):
            raise TypeError("Expected input tensor to have dtype {}, got {}".format(
                    self.dtype, input.dtype))

        # The squeeze and unsqueeze are to insert a dim for num_channels == 1.
        return torch.nn.functional.conv_transpose1d(input.unsqueeze(1),
                                                    self.backward_filter,
                                                    stride=self.N,
                                                    padding=self.padding).squeeze(1)


