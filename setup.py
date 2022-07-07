"""
Library setup
"""

from setuptools import find_packages, setup  # type: ignore

# Library dependencies
INSTALL_REQUIRES = ["gensim==3.8.3", "pandas", "tqdm"]

# Testing dependencies
TEST_REQUIRES = ["pytest", "black", "pre-commit", "flake8", "mypy", "bandit",
                 "pytest-cov", "tensorflow>=2.3.0", "tqdm"]

setup(
    name="GiaNLP",
    version="0.0.1",
    description="Natural Language Processing for humans",
    author="Gianmarco Cafferata",
    author_email="gcafferata@fi.uba.ar",
    url="git@github.com:jian01/GiaNLP.git",
    packages=find_packages(),
    python_requires=">=3.5",
    setup_requires=["wheel"],
    install_requires=INSTALL_REQUIRES,
    extras_require={"test": TEST_REQUIRES},
    classifiers=[
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development",
    ],
)