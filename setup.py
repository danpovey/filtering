import setuptools
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    python_requires='>=3.5',
    name="lilfilter",
    version="0.0.1",
    author="Daniel Povey",
    author_email="dpovey@gmail.com",
    description="Utilities for filtering and resampling signals",
    keywords="resampling,audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danpovey/filtering",
    packages=['lilfilter'],
    install_requires=[
        'numpy', 'torch'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
        "Operating System :: OS Independent",
    ],
)
