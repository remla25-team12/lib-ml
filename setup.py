from setuptools import setup, find_packages

setup(
    name="libml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "scikit-learn",
    ],
    description="Preprocessing logic for the restaurant sentiment model",
    url="https://github.com/remla25-team12/lib-ml",
)