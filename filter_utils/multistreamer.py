# To be run with python3.  Caution: this module requires torch!

"""
This module defines an object called Multistreamer that can be used to
separate a real-valued audio signal into multiple complex streams
(with each stream representing a single frequency band of the
original data), and then reconstruct the original signal from
the multiple streams.

The user chooses a number N >= 1 that represents the ratio between the input
sampling rate and the sampling rate of our multple complex streams.  (The number
of streams will be N+1 because of special cases at the zero frequency and the
Nyquist frequency; this is similar to how the real FFT works).

Since the input signal is real and the output is complex, this means that we're
encoding the signal with a redundancy of greater than 2.  (This kind of
redundancy is common whenever you use representations that have locality in both
time and frequency).
"""

# TODO: eventually add an option for PyTorch-based computation, while not
# requiring it as a dependency (back off to numpy).

import numpy as np
import cmath
import math
import torch
from . import filter_function



class Multistreamer:
    def __init__(self, N = 8,
                 S = 10,
                 double_precision = False,
                 full_padding = True):
        """
        Constructor.

        Args:
            N (int):  The number of streams we split the signal into.  We'll
                   split into N complex streams, and each stream will be sampled
                   N times slower than the input signal.  This gives us a
                   redundancy factor of 2, given that the input signal was
                   real and each complex number is formed from 2 reals.

            S (int):  S defines the width of support of the canonical (un-scaled)
                  version of the filter function we're using.  Currently the
                  only allowed values are 6 or 10; these are determined by
                  which values of S we have tabulated the filter function for
                  in ./filter_function.py.  The default of 10 gives more
                  accurate reconstruction, 6 would be faster but give less
                  accurate reconstruction (relative errors of, say, 10^-4 versus
                  10^-5, to give the ballpark estimate).

            double_precision (bool):  If true, the filters used will be double
                   precision.

            full_padding (bool):  If true, we'll pad the input signal with zeros as
                   needed to prevent extra reconstruction error at the ends
                   of the signal: specifically we'll pad with `filter_size`
                   samples; this will mean that the subsampled signal returned
                   from split() has a length slightly longer than
                   input_signal_length / N.
        """

        assert isinstance(N, int) and N >= 1 and isinstance(double_precision, bool)
        assert isinstance(full_padding, bool)

        self.N = N
        self.double_precision = double_precision
        self.full_padding = full_padding

        # Get the base filter function (before multiplying by e^(-i omega t),
        # where 0 <= omega <= pi is the base frequency of this band.

        if not S in filter_function.f.keys():
            raise ValueError("S = {} is not one of the allowed values {}".format(
                    S, filter_function.f.keys()))

        # the filter function is defined out to (S * N) to each side of
        # its center, but at those points (+- S * N) it is zero, so there is
        # no point including that in the representation, so we subtract,
        # rather than add, 1.
        filter_size = (S * N) * 2 - 1

        self.filter_offset = (S * N) - 1


        if full_padding:
            self.padding = 2 * S * N
        else:
            self.padding = S * N


        complex_dtype = (np.complex128 if double_precision else np.complex64)
        canonical_filter = np.empty((filter_size), dtype=complex_dtype)

        for i in range(filter_size):
            t_int = i - (S * N - 1)
            t = t_int / N
            # Divide the filter function by N when scaling it on the x-axis by
            # N, as we want the gain at its center frequency to still be one;
            # and that scales with the areal.
            canonical_filter[i] = filter_function.get_function_at(S, t) / N


        self.filter_matrix = np.empty((N, filter_size), dtype=complex_dtype)

        for b in range(N):
            # `band_center` is the center of the band as an angular frequency,
            # assuming the input signal was sampled at an angular frequency of 2pi
            # (i.e. at 1Hz).
            # Note: due to symmetry (since it's a real input signal), we only
            # use the postive complex frequencies and not their negative mirror
            # images.
            band_center = (b + 0.5) * math.pi / N



            for i in range(filter_size):
                # This 't' value is the time relative to the center of the
                # filter, at the original sampling rate, e.g. if the original
                # sampling rate it 1Hz it would be in seconds.
                t = i - (S * N - 1)

                f = canonical_filter[i]

                # The minus sign is similar in purpose to the minus sign in the forward version of
                # the fourier transform.  We'll take the conjugate when reconstructing.
                self.filter_matrix[b, i] = f * cmath.exp(-1.0j * band_center * t)



        dtype_real = torch.float64 if double_precision else torch.float32

        # (2*N, 1, filter_size) is interpreted as 1d-convolution terms as:
        # (num_channels_out, num_channels_in, patch-width) The order is all the
        # reals and then all the imaginary parts, which makes it easier later on
        # to reshape to view the real and imaginary parts as being two channels.
        self.filter_forward = torch.empty((2 * N, 1, filter_size),
                                           dtype=dtype_real)
        self.filter_forward[0:N,0,:]  = torch.tensor(self.filter_matrix.real)
        self.filter_forward[N:2*N,0,:] = torch.tensor(self.filter_matrix.imag)

        # We don't need to explicitly take the conjugate here, since the need
        # for it disappears when we do things in real arithmetic.  (The negation
        # of the imaginary part was to cancel out the factor i * i = -1).
        self.filter_backward = 2.0 * N * self.filter_forward



    def split(self, signal):
        """
        Demultiplex the signal `signal` into multiple streams.  This is the
        `forward` direction of the transform.

        Args:
           signal   A torch.Tensor of shape (minibatch_size, signal_length)

        Returns:
           Returns a torch.Tensor of shape (minibatch_size, 2, N, reduced_signal_length)
           which is intended to be interpreted as:
           (minibatch_size, num_channels=2, height=N, width=reduced_signal_length).

           where reduced_signal_length is about signal_length / N.
           The two channels may be interpreted as, respectively, as the real component
           and the imaginary component of N separate complex sequences for each
           input signal.  Each of the N sequences represents one frequency band
           of the input; the n'th sequence represents a frequency band centered at
           an angular frequency of (n + 0.5) pi / N assuming the input sequence
           was sampled at 1Hz, so potentially contained angular frequencies from -pi
           through pi.
        """


        if len(signal.shape) != 2:
            raise ValueError("Signal expected to have two axes (minibatch_size, signal_length); got".format(
                             signal.shape))
        dtype_real = torch.float64 if self.double_precision else torch.float32
        if signal.dtype != dtype_real:
            raise ValueError("Expected input signal to have dtype == {}, found {}".format(
                    dtype_real, signal.dtype))

        # Change signal from (minibatch_size, signal_length) to (minibatch_size,
        # 1, signal_length) where the 1 represents the number of input channels
        signal = signal.unsqueeze(1)

        conv_output = torch.nn.functional.conv1d(signal, self.filter_forward,
                                                 stride = self.N,
                                                 padding = self.padding)
        minibatch_size = conv_output.shape[0]
        reduced_signal_length = conv_output.shape[2]
        return conv_output.view(minibatch_size, 2, self.N, reduced_signal_length)


    def merge(self, split_signal):
        """
        This is the reverse of the `split` function; it reconstructs the signal from
        the output of `split`.

        Args:
           spilt_signal   A torch.Tensor of shape (minibatch_size, 2, N, reduced_signal_length);
                      see docs for split() for more information on what this represents.
        Returns:
           Returns a torch.Tensor of shape (minibatch_size, signal_length), where
           signal_length will be original signal_length truncated to the nearest
           multiple of N.
        """

        if len(split_signal.shape) != 4 or split_signal.shape[1] != 2 or split_signal.shape[2] != self.N:
            raise ValueError("Expected input to have 4 axes, of the form (x, 2, {}, y), got {}".format(
                    self.N, split_signal.shape))

        dtype_real = torch.float64 if self.double_precision else torch.float32
        if split_signal.dtype != dtype_real:
            raise ValueError("Expected input to have dtype {}, got instead {}".format(
                    split_signal.dtype, dtype_real))

        # actually minibatch_size, 2, N, reduced_signal_length; we checked above.
        (minibatch_size, _, _, reduced_signal_length) = split_signal.shape


        # So that we can apply 1d convolution code, we combine the dimensions 2
        # and N into a single dimension interpreted as the channel.  You can see
        # the shape of reshaped_split_signal as the args of view() below.
        reshaped_split_signal = split_signal.contiguous().view(
            minibatch_size, 2 * self.N, reduced_signal_length)


        # If you want to get back the same signal as you input (minus edge effects if
        # there was an input frame or two left over that didn't make up a whole stride),
        # you have to give it the same args as the forward one.  This isn't really
        # documented well, I figured it out experimentally.
        # Note: self.filter_backward is just self.filter_forward * 2N.
        signal = torch.nn.functional.conv_transpose1d(
            reshaped_split_signal,
            self.filter_backward,
            stride = self.N,
            padding = self.padding)

        # signal will be of shape:
        #   (minibatch_size, 1, signal_length)
        # where the 1 is the number of channels.  We remove this axis when we return it.
        return signal.squeeze(1)
