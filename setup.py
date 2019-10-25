import setuptools
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    python_requires='>3.0.0',
    name="lilfilter",
    version="0.0.1",
    author="Daniel Povey",
    author_email="dpovey@gmail.com",
    description="Utilities for filtering signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danpovey/filtering",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'torch'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
