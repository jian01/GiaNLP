"""
Library setup
"""
import sys
from os import path
from setuptools import find_packages, setup  # type: ignore

readme_file = path.join(path.dirname(path.abspath(__file__)), "README.md")
with open(readme_file) as f:
    readme = f.read()

license_file = path.join(path.dirname(path.abspath(__file__)), "LICENSE")
with open(license_file) as f:
    license_content = f.read()

# Library dependencies
INSTALL_REQUIRES = ["gensim>=4.0.0", "pandas", "tqdm"]

# Testing dependencies
TEST_REQUIRES = ["pytest", "pytest-cov", "tensorflow>=2.3.0"]

setup(
    name="GiaNLP",
    version="0.0.1",
    description="Natural Language Processing for humans",
    long_description_content_type="text/markdown",
    long_description=readme,
    license=license_content,
    author="Gianmarco Cafferata",
    author_email="gcafferata@fi.uba.ar",
    url="https://jian01.github.io/GiaNLP/",
    packages=["gianlp"],
    python_requires=">=3.5",
    setup_requires=["wheel"],
    install_requires=INSTALL_REQUIRES,
    tests_require=TEST_REQUIRES,
    test_suite="tests",
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License v2.0",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
