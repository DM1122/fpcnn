[![Python Version](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DM1122/fpcnn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![GitHub repo size](https://img.shields.io/github/repo-size/DM1122/fpcnn)](https://github.com/DM1122/fpcnn)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/DM1122/fpcnn)](https://github.com/DM1122/fpcnn)


<img src="img/utat-logo.png" height="64">

# FPCNN
FINCH Predictive Coder Neural Network.

A full end-to-end implementation of the [CSNN](https://www.mdpi.com/2313-433X/6/6/38) state-of-the-art compression algorithm for hyperspectral image data. Developed by the [University of Toronto Aerospace Team](https://www.utat.ca/space-systems) :milky_way:

<p align="center"><img src="img/csnn.png" height="256"></p>

# Usage
Check out the interactive Google Colab [notebooks](https://colab.research.google.com/github/DM1122/fpcnn) to start tinkering.

# Contribution
## Setup
This section will take you through the procedure to configure a development environment for FPCNN. If you just want to play around with FPCNN, see the interactive [notebooks](https://colab.research.google.com/github/DM1122/fpcnn) on Google Colab.

procedure... WIP


## Commits
### Pre-Commit
This repo is configured to use pre-commit hooks. The pre-commit pipeline is configured as follows:

1. [**Isort**](https://pycqa.github.io/isort/): Sorts imports, so you don't have to.
1. [**Black**](https://black.readthedocs.io/en/stable/): The uncompromising code formatter.
1. [**Flakehell**](https://flakehell.readthedocs.io/): It’s a Flake8 wrapper to make it cool.

A successful commit therefore requires satisfying the syntactic rules put forth by isort, black, and flakehell.

### The 5 Rules of A Great Git Commit Message
<p align="center"><img src="https://imgs.xkcd.com/comics/git_commit.png" width="256"></p>

1. Write in the imperative
1. Capitalize first letter in the subject line 
1. Describe what was done and why, but not how
1. Limit subject line to 50 characters
1. End without a period


# TODO
* batch processsing
* wrapper on plotting utilities
* benchmarklib improvements to handle export of csv files
* improve readme
* prototyping notebooks (once google collab python version updates to 3.8)
* hyperparameter optimization toolkit
* discuss how to run isort, black, linter manually when pre-commit fails
* add about testing section. html pytest


isort . 
black .
flakehell lint
from the root directory



