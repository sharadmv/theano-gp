from setuptools import setup, find_packages

setup(
    name = "gp",
    version = "0.0.1",
    author = "Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    install_requires=['theanify'],
    packages=find_packages('.')
)
