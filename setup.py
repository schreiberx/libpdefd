#! /usr/bin/env python3


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(
    name="pdefd",
    version="0.0.1",
    author="Martin Schreiber",
    author_email="schreiberx@gmail.com",
    description="Finite difference PDE solver framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
#    url="https://github.com/pypa/sampleproject",
#    packages=setuptools.find_packages(),
    packages=['libpdefd'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

