# To be run with python3.  Caution: this module requires torch!

"""
This module defines an object called Normalizer that can be used to
normalize the output of class Multistreamer (from ./multistreamer.py),
with a view to making the data easier for neural nets to process.

The basic idea is that we compute a moving average of the amplitude of the
signal within each frequency band, and use that to normalize the signal.  (The
neural net will see both the normalized signal and the log of the normalization
factor).  The idea is that after possibly being modified by the nnet
(e.g. denoised), we then 'un-normalize' the signal with the same normalization
factor.

We also provide a factor that can be used as part of the objective function
if it's desired to put a greater weight on the louder frequency bands for
training purposes.
"""


import numpy as np
import cmath
import math
import torch
from . import filter_function
from . import filters
from . import torch_filter
from . import resampler

import matplotlib.pyplot as plt  # TEMP

class LocalAmplitudeComputer:
    """
    This class is a utility for computing the smoothed-over-time local amplitude
    of a signal, to be used in class Normalizer to compute a normalized form of
    the signal.
    """
    def __init__(self,
                 gaussian_stddev = 100.0,
                 epsilon = 1.0e-05,
                 block_size = 8,
                 double_precision = False):
        """
        Constructor.
        Args:
           gaussian_stddev (float):  This can be interpreted as a time constant measured
                    in samples; for instance, if the sampling rate of the signal
                    we are normalizing is 1kHz, gaussian_stddev = 1000 would mean
                    we're smoothing with approximately 1 second of data on each
                    side.
           epsilon (float):  A constant that is used to smooth the instantaneous
                    amplitude.  Dimensionally this is an amplitude.
           block_size  A number which should be substantially less than
                    gaussian_stddev.  We first sum the data over blocks and then
                    do convolutions, efficiency.  Any number >= 1 is OK but
                    numbers approaching gaussian_stddev may start to affect
                    the output
           double_precision  If true, create these filters in double precision
                    (float64), will require input to be double too.
        """
        if block_size < 1 or block_size >= gaussian_stddev / 2:
            raise ValueError("Invalid values block-size={}, gaussian-stddev={}".format(
                    block_size, gaussian_stddev))

        # reduced_stddev is the stddev after summing over blocks of samples
        # (which reduces the sampling rate by that factor).
        reduced_stddev = gaussian_stddev / block_size
        (f, i) = filters.gaussian_filter(reduced_stddev)
        # We'll be summing, not averaging over blocks, so we need
        # to correct for that factor.
        f *= (1.0 / block_size)

        self.epsilon = epsilon

        self.dtype = torch.float64 if double_precision else torch.float32

        self.gaussian_filter = torch_filter.SymmetricFirFilter(
            (f,i), double_precision = double_precision)


        self.block_size = block_size
        if block_size > 1:
            # num_zeros = 4 is a lower-than-normal width for the FIR filter since there
            # won't be frequencies near the Nyquist and we don't need a sharp cutoff.
            # filter_cutoff_ratio = 9 is to avoid aliasing effects with this less-precise
            # filter (default is 0.95).
            self.resampler = resampler.Resampler(block_size, num_zeros = 4,
                                                 filter_cutoff_ratio = 0.9,
                                                 double_precision = double_precision)


    def compute(self,
                input):
        """
        Computes and returns the local energy which is a smoothed version of the
        instantaneous amplitude.

        Args:
          input: a torch.Tensor with dimension
            (minibatch_size, 2, num_channels, signal_length)
            representing the (real, imaginary) parts of `num_channels`
            parallel frequency channels.  dtype should be
            torch.float32 if constructor had double_precision==False,
            else torch.float36.
        Returns:
           Returns a torch.Tensor with dimension (minibatch_size, num_channels,
            signal_length) containing the smoothed local amplitude.
        """
        if not isinstance(input, torch.Tensor) or input.dtype != self.dtype:
            raise TypeError("Expected input to be of type torch.Tensor with dtype=".format(
                            self.dtype))
        if len(input.shape) != 4 or input.shape[1] != 2:
            raise ValueError("Expected input to have 4 axes with the 2nd dim == 2, got {}".format(
                    input.shape))
        (minibatch_size, two, num_channels, signal_length) = input.shape


        # We really want shape (minibatch_size, num_channels, signal_length) for
        # instantaneous_amplitude, but we want another array of size (signal_length)
        # containing all ones, for purposes of normalization after applying the
        # Gaussian smoothing (to correct for end effects)..
        amplitudes = torch.empty(
            (minibatch_size * num_channels + 1), signal_length,
            dtype=self.dtype)

        # set the last row to all ones.
        amplitudes[minibatch_size*num_channels:,:] = 1

        instantaneous_amplitude = amplitudes[0:minibatch_size*num_channels,:].view(
            minibatch_size, num_channels, signal_length)
        instantaneous_amplitude.fill_(self.epsilon*self.epsilon)  # set to epsilon...
        instantaneous_amplitude += input[:,0,:,:] ** 2
        instantaneous_amplitude += input[:,1,:,:] ** 2
        instantaneous_amplitude.sqrt_()


        # summed_amplitudes has num-cols reduced by about self.block_size,
        # which will make convolution with a Gaussian easier.
        summed_amplitudes = self._block_sum(amplitudes)


        smoothed_amplitudes = self.gaussian_filter.apply(summed_amplitudes)
        assert smoothed_amplitudes.shape == summed_amplitudes.shape

        upsampled_amplitudes = self.resampler.upsample(smoothed_amplitudes)
        assert upsampled_amplitudes.shape[1] >= signal_length



        # Truncate to actual signal length (we may have a few extra samples at
        # the end.)  Remove the first self.block_size samples to avoid small
        # phase changes, not that it would really matter since the block
        # size will be << the gaussian stddev.
        upsampled_amplitudes = upsampled_amplitudes[:,:signal_length]

        n = minibatch_size*num_channels
        # The following corrects for constant factors, including a
        # 1/b factor that we missed when summing over blocks, and also for
        # edge effects so that we can interpret the Gaussian convolution as
        # an appropriately weighted average near the edges of the signal.
        # We took a signal of all-ones and put it through this process
        # as the last row of an n+1-row matrix, and we're using that
        # to normalize.
        # The shapes of the expressions below are, respectively:
        #   (minibatch_size*num_channels, signal_length) and (1, signal_length)
        upsampled_amplitudes[0:n,:] /= upsampled_amplitudes[n:,:]


        # the `contiguous()` below would not be necessary if PyTorch had been
        # more carefully implemented, since the shapes here are quite compatible
        # with zero-copy.  (Possibly it's not necessary even now, not 100%
        # sure.)
        return upsampled_amplitudes[0:n,:].contiguous().view(minibatch_size, num_channels,
                                                             signal_length)

    def _block_sum(self, amplitudes):
        """
        This internal function sums the input amplitudes over blocks
        (we do this before the Gaussian filtering to save compute).

        Args:
          amplitudes: a torch.Tensor with shape (n, s) with s being the
                  signal length and n being some combination of minibatch
                  and channel; dtype self.dtype
        Returns:
          returns a torch.Tensor with shape (n, t) where t = (s+2b-1)//b, where
          b is the block_size passed to the constructor.  Note that this means
          we are padding with two extra outputs, one zero-valued block at the
          start and also a partial block sum at the end.  This is necessary to
          ensure we have enough samples when we upsample the Gaussian-smoothed
          version of this.  It also means we get the amplitude sum for time t
          from a Gaussian centered at about t - block_size/2; this is harmless.
        """
        amplitudes = amplitudes.contiguous()
        b = self.block_size
        (n, s) = amplitudes.shape
        t = (s + 2 * b - 1) // b

        ans = torch.zeros((n, t), dtype=self.dtype)

        # make sure `amplitudes` is contiguous.

        # t_end will be t-1 if there is a partial block, otherwise t.
        t_whole = s // b      # the number of whole sums
        t_end = t_whole + 1
        s_whole = (s // b) * b

        # Sum over the b elements of each block.
        ans[:,1:t_end] += amplitudes[:,:s_whole].view(n, t_whole, b).sum(dim=-1)
        if t_end != t:
            # sum over the left-over columns, i.e. sum over k things where k ==
            # s % b
            ans[:,t_end] += amplitudes[:,s_whole:].sum(dim=-1)
        return ans





