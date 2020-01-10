# Copyright 2018 The TF-Slim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup script for TF-Slim.

This script will install TF-Slim as a Python module.

See: https://github.com/google-research/tf-slim

"""

from os import path
from setuptools import find_packages
from setuptools import setup

here = path.abspath(path.dirname(__file__))

install_requires = ['absl-py >= 0.2.2',]
tests_require = ['nose']

tf_slim_description = (
    'TensorFlow-Slim: A lightweight library for defining, training and '
    'evaluating complex models in TensorFlow')


setup(
    name='tf_slim',
    version='1.0',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite='nose.collector',
    description=tf_slim_description,
    long_description=tf_slim_description,
    url='https://github.com/google-research/tf-slim',  # Optional
    author='The TF-Slim Team',  # Optional
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    project_urls={  # Optional
        'Documentation': 'https://github.com/google-research/tf-slim',
        'Bug Reports': 'https://github.com/google-research/tf-slim/issues',
        'Source': 'https://github.com/google-research/tf-slim',
    },
    license='Apache 2.0',
    keywords='tensorflow tf-slim python machine learning'
)
