#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_namespace_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="HISAN",
    author="Idiri Massinissa, Thayananthan Anithyan, Ghemmour Yacine, Lekhmamra Chihabeddine",
    author_email="massinissa.idiri@etu.sorbonne-universite.fr",
    description="Re-implementation of the Hierarchical Self-Attention Network for Action Localization in Videos (HISAN)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "tqdm", "pandas", "tensorflow",
                      "keras", "pathlib", "pickle", "shutil", "jupyter"],
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False
)
