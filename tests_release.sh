# Copyright 2021 The TF-Slim Authors.
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

#!/usr/bin/env bash

# Creates a pip package for tf-slim after executing unit tests.
#
# Script is assumed to run from within a docker where the version of tensorflow
# to test against is already installed.
#
# bash tests_release.sh
#
# Example usage with docker:
#   # Test and build with latest
#   docker run --rm -v $(pwd):/workspace \
#     --workdir /workspace tensorflow/tensorflow:latest  \
#     bash tests_release.sh
#
#   # Test and build with nightly
#   docker run --rm -v $(pwd):/workspace \
#     --workdir /workspace tensorflow/tensorflow:nightly  \
#     bash tests_release.sh


# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x


run_tests() {
  echo "run_tests"
  TMP=$(mktemp -d)

  # Run the tests
  python setup.py test

  # Install tf_slim package.
  WHEEL_PATH=${TMP}/wheel/$1
  ./pip_pkg.sh ${WHEEL_PATH}/

  pip install ${WHEEL_PATH}/tf_slim*.whl

  # Move away from repo directory so "import tf_agents" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tf_slim')

  # Copies wheel out of tmp to root of repo so it can be more easily uploaded
  # to pypi as part of the stable release process.
  cp ${WHEEL_PATH}/tf_slim*.whl ./

}

# Build and run tests.
run_tests


