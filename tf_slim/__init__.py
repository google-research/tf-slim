# coding=utf-8
# Copyright 2016 The TF-Slim Authors. All Rights Reserved.
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
"""Slim is an interface to TF functions, examples and models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member,wildcard-import
from tf_slim import evaluation
from tf_slim import learning
from tf_slim import model_analyzer
from tf_slim import queues
from tf_slim import summaries
from tf_slim.data import data_decoder
from tf_slim.data import data_provider
from tf_slim.data import dataset
from tf_slim.data import dataset_data_provider
from tf_slim.data import parallel_reader
from tf_slim.data import prefetch_queue
from tf_slim.data import tfexample_decoder

# TODO(b/135606235): Delete non-slim imports
# -- from tensorflow.contrib import losses
# from tensorflow import losses
# -- from tensorflow.contrib import metrics
# from tensorflow import metrics
# -- from tensorflow.contrib.framework.python.ops.arg_scope import *
# -- from tensorflow.contrib.framework.python.ops.variables import *
from tf_slim.ops.arg_scope import *
from tf_slim.ops.variables import *
from tf_slim.layers.layers import *
from tf_slim.layers.initializers import *
from tf_slim.layers.regularizers import *
from tensorflow.python.util.all_util import make_all  # pylint:disable=g-direct-tensorflow-import
# pylint: enable=unused-import,line-too-long,g-importing-member,wildcard-import


from tensorflow.compat.v1 import disable_eager_execution
disable_eager_execution()

__all__ = make_all(__name__)
