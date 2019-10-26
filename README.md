## lilfilter

Utilities for resampling and filtering audio data

This repository exports a Python package `lilfilter` containing certain
utilities for filtering and resampling audio data.


One quite-useful thing is class Resampler:
```
python3
>>> import lilfilter
>>> # ... let a be a Torch tensor of size (num_channels, num_samples)
>>> # that we want to downsample from 42.1kHz to 16kHz.  Note,
>>> # the sampling rates must be integers; only their ratio
>>> # matters.
>>> r = lilfilter.Resampler(42100, 16000, dtype=torch.float32)
>>> b = r.resample(a)
```

Another thing that's useful is class Multistreamer, which can turn a
signal into multiple parallel signals at a lower sampling rate, where
pairs of those signals represent the (real,complex) part of one
complex frequency band of the input.
```
>>> import lilfilter
>>> num_freq_bands = 8
>>> m = lilfilter.Multistreamer(num_freq_bands)
>>>
>>> # ... let a be a Torch tensor of size (num_channels, num_samples)
>>> # that we want to `demultiplex`.
>>>
>>> b = m.split(a)
>>> # now b is of size (num_channels, 2, num_freq_bands, num_samples/num_freq_bands)
>>> # (note: the dim of the last axis may be slightly different from that number).
>>> # You can in principle manipulate b somehow, e.g. do some kind of machine
>>> # learning with it, and then reconstruct to the original format:
>>>
>>> c = m.merge(b)
>>> # now c is of size (num_channels, 8*(num_samples/8)) and will be extremely
>>> # close to a.
```


