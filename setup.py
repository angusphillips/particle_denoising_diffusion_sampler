#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

with open("requirements.txt", "r") as file:
    requirements = [line.strip() for line in file]

with open("README.md", "r") as file:
    long_description = file.read()

# with open("VERSION", "r") as file:
#     version = file.read().strip()

setup(
    name="diffusion-smc",
    # version=version,
    author="Angus Phillips, Michael Hutchinson, Hai-Dang Dau",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="",
    # license="Apache License 2.0",
    keywords="",
    install_requires=requirements,
    author_email="angus.phillips@stats.ox.ac.uk",
    packages=find_namespace_packages(include=["pdds"]),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
