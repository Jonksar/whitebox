"""
  --------------------------------------------------
  File Name : setup.py
  Creation Date : 2019-01-21 Mon 02:33 pm
  Last Modified : 2019-01-28 Mon 04:20 pm
  Created By : Joonatan Samuel
  --------------------------------------------------
"""

from setuptools import setup, find_packages

setup(
    name = 'whitebox',
    version = '0.1.0',
    url = 'https://github.com/Jonksar/whitebox',
    author = 'Joonatan Samuel',
    author_email = 'joonatan.samuel@gmail.com',
    description = 'Machine learning tools and frameworks that are mostly built on sklearn and other open source machine learning frameworks.',
    packages = find_packages(),
    install_requires = [
        "seaborn",
        "matplotlib",
        "numpy",
        "pandas",
        "termcolor",
        "imageio",
        "sklearn",
        "sklearn_pandas",
<<<<<<< Updated upstream
=======
        "gensim",
        "annoy",
        "wget",
        "nltk",
>>>>>>> Stashed changes
    ]
)
