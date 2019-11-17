import os
import sys

from setuptools import find_packages, setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

setup(
    name='bayesfem',
    version='0.0.1',
    author='Daniel Tait',
    author_email='tait.djk@gmail.com',
    description=('Bayesian inference for PDEs using the FEM', ),
    long_description=read('README.md'),
    license='BSD',
    packages=find_packages(),
    install_requires='tensorflow>=2.0',
)
