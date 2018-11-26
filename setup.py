#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages
import os

setup_path = os.path.abspath(os.path.dirname(__file__))

setup(name='fast_adversarial',
      version='0.1',
      url='https://github.com/jeromerony/fast_adversarial',
      maintainer='Jerome Rony, Luiz G. Hafemann',
      maintainer_email='jerome.rony@gmail.com',
      description='Implementation of gradient-based attacks and defenses for adversarial examples',
      author='Jerome Rony, Luiz G. Hafemann',
      author_email='jerome.rony@gmail.com',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      python_requires='>=3.6',
      install_requires=[
          'torch>=0.4.1',
          'torchvision>=0.2.1',
          'tqdm>=4.23.4',
          'visdom>=0.1.8',
          'foolbox>=1.7.0',
      ],
      packages=find_packages())
