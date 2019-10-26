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
                 input_sr, output_sr, dtype,
                 num_zeros = 64, cutoff_ratio = 0.95):
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
          dtype:  The torch dtype to use for computations
          num_zeros: The number of zeros per side in the (sinc*hanning-window)
              filter function.  More is more accurate, but 64 is already
              quite a lot.

        You can think of this algorithm as dividing up the signals
        (input,output) into blocks where there are `input_sr` input
        samples and `output_sr` output samples.  Then we treat it
        using convolutional code, imagining there are `input_sr`
        input channels and `output_sr` output channels per time step.

        """
        assert isinstance(input_sr, int) and isinstance(output_sr, int)
        d = gcd(input_sr, output_sr)
        input_sr, output_sr = input_sr // d, output_sr // d

        assert dtype in [torch.float32, torch.float64]
        assert num_zeros > 3  # a reasonable bare minimum
        np_dtype = np.float32 if dtype == torch.float32 else np.float64

        # Define one 'block' of samples `input_sr` input samples
        # and `output_sr` output samples.  We can divide up
        # the samples into these blocks and have the blocks be
        #in correspondence.

        # The sinc function will have, on average, `zeros_per_block`
        # zeros per block.
        zeros_per_block = min(input_sr, output_sr) * cutoff_ratio

        # The convolutional kernel size will be n = (blocks_per_side*2 + 1),
        # i.e. we add that many blocks on each side of the central block.  The
        # window radius (defined as distance from center to edge)
        # is `blocks_per_side` blocks.  This ensures that each sample in the
        # central block can "see" all the samples in its window.
        #
        # Assuming the following division is not exact, adding 1
        # will have the same effect as rounding up.
        blocks_per_side = 1 + int(num_zeros / zeros_per_block)

        kernel_width = 2*blocks_per_side + 1

        # We want the weights as used by torch's conv1d code; format is
        #  (out_channels, in_channels, kernel_width)
        # https://pytorch.org/docs/stable/nn.functional.html
        weights = torch.tensor((output_sr, input_sr, kernel_width), dtype=dtype)

        # Computations involving time will be in units of 1 block.  Actually this
        # is the same as the `canonical` time axis since each block has input_sr
        # input samples, so it would be one of whatever time unit we are using
        window_radius_in_blocks = blocks_per_side


        # The `times` below will end up being the args to the sinc function.
        # For the shapes of the things below, look at the args to `view`.  The terms
        # below will get expanded to shape (output_sr, input_sr, kernel_width) through
        # broadcasting
        # We want it so that, assuming input_sr == output_sr, along the diagonal of
        # the central block we have t == 0.
        # The signs of the output_sr and input_sr terms need to be opposite.  The
        # sign that the kernel_width term needs to be will depend on whether it's
        # convolution or correlation, and the logic is tricky.. I will just find
        # which sign works.



        times = (
            np.arange(output_sr, dtype=np_dtype).reshape((output_sr, 1, 1)) / output_sr -
            np.arange(input_sr, dtype=np_dtype).reshape((1, input_sr, 1)) / input_sr -
            (np.arange(kernel_width, dtype=np_dtype).reshape((1, 1, kernel_width)) - blocks_per_side))


        def window_func(a):
            """
            window_func returns the Hann window on [-1,1], which is zero
            if a < -1 or a > 1, and otherwise 0.5 + 0.5 cos(a/pi).
            This is applied elementwise to a, which should be a NumPy array.

            The heaviside function returns (a > 0 ? 1 : 0).
            """
            return np.heaviside(1 - np.abs(a), 0.0) * (0.5 + 0.5 * np.cos(a * np.pi))


        # The weights below are a sinc function times a Hann-window function.
        #
        # multiplication by zeros_per_block can be seen as correctly normalizing
        # the sinc function (to compensate for scaling on the x-axis), so that
        # its integral is 1.
        #
        # division is by input_sr can be interpreted as normalizing the input
        # function correctly...if we view it as a stream of dirac deltas that's
        # passed through a low pass filter and want that to have the same
        # magnitude as the original input function, we need to divide by the
        # number of those deltas per unit time.
        weights = (np.sinc(times * zeros_per_block)
                   * window_func(times / window_radius_in_blocks)
                   * zeros_per_block / input_sr)

        self.input_sr = input_sr
        self.output_sr = output_sr
        self.padding = blocks_per_side
        self.weights = torch.tensor(weights, dtype=dtype)



    def resample(self, in_data):
        """
        Resample the data

        Args:
         input: a torch.Tensor with the same dtype as was passed to the
        constructor.
         There must be 2 axes, interpreted as (minibatch_size, sequence_length)

        Return:  Returns a torch.Tensor with the same dtype as the input, and
         dimension (minibatch_size, (sequence_length//input_sr)*output_sr),
         where input_sr and output_sr are the corresponding constructor args,
         modified to remove any common factors.


        """

        for x in [1]: #try:
            # input.unsqueeze(1) changes dim from (minibatch_size, sequence_length) to
            # (minibatch_size, num_channels=1, sequence_length)
            # the final squeeze(1) removes the num_channels=1 axis

            # round the input length

            (minibatch_size, seq_len) = in_data.shape
            num_blocks = seq_len // self.input_sr
            if num_blocks == 0:
                # TODO: pad with zeros.
                raise RuntimeError("Signal is too short to resample")
            in_data = in_data[:, 0:(num_blocks*self.input_sr)]  # Truncate input
            in_data = in_data.view(minibatch_size, num_blocks, self.input_sr)

            # Torch's conv1d actually expects input data with shape (minibatch,
            # in_channels, width) so we need to reshape (note: time is width).
            in_data = in_data.transpose(1, 2)

            out = torch.nn.functional.conv1d(in_data, self.weights,
                                             padding=self.padding)
            assert out.shape == (minibatch_size, self.output_sr, num_blocks)
            return out.transpose(1, 2).contiguous().view(minibatch_size, num_blocks * self.output_sr)
#except Exception as e:
#            print("Error: exception is {}".format(e))
