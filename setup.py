from setuptools import setup, find_packages

# Configuration is in pyproject.toml
setup(
    packages=find_packages(exclude=('tests', 'exp_notebooks', '_static')),
)

