# Copyright 2019 The TF-Slim Authors.
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

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs, which are replicated to sponge.
set -x


run_tests() {
  echo "run_tests $1"
  TMP=$(mktemp -d)
  # Create and activate a virtualenv to specify python version and test in
  # isolated environment. Note that we don't actually have to cd'ed into a
  # virtualenv directory to use it; we just need to source bin/activate into the
  # current shell.
  VENV_PATH=${TMP}/virtualenv/$1
  virtualenv -p "$1" "${VENV_PATH}"
  source ${VENV_PATH}/bin/activate


  # TensorFlow isn't a regular dependency because there are many different pip
  # packages a user might have installed.
  pip install tensorflow

  # Run the tests
  python setup.py test

  # Install the tf_slim package
  WHEEL_PATH=${TMP}/wheel/$1
  ./pip_pkg.sh ${WHEEL_PATH}/

  pip install ${WHEEL_PATH}/tf_slim*.whl

  # Move away from repo directory so "import tf_slim" refers to the
  # installed wheel and not to the local fs.
  (cd $(mktemp -d) && python -c 'import tf_slim')

  # Deactivate virtualenv
  deactivate
}

# Test on Python2.7
run_tests "python2.7"
# Test on Python3.6
run_tests "python3.6"

