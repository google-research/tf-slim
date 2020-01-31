# coding=utf-8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Ops for evaluation metrics and summary statistics.

See the
[Contrib Metrics](https://tensorflow.org/api_guides/python/contrib.metrics)
guide.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_slim.metrics.classification import accuracy
from tf_slim.metrics.classification import f1_score
from tf_slim.metrics.histogram_ops import auc_using_histogram
from tf_slim.metrics.metric_ops import aggregate_metric_map
from tf_slim.metrics.metric_ops import aggregate_metrics
from tf_slim.metrics.metric_ops import auc_with_confidence_intervals
from tf_slim.metrics.metric_ops import cohen_kappa
from tf_slim.metrics.metric_ops import count
from tf_slim.metrics.metric_ops import precision_at_recall
from tf_slim.metrics.metric_ops import precision_recall_at_equal_thresholds
from tf_slim.metrics.metric_ops import recall_at_precision
from tf_slim.metrics.metric_ops import sparse_recall_at_top_k
from tf_slim.metrics.metric_ops import streaming_accuracy
from tf_slim.metrics.metric_ops import streaming_auc
from tf_slim.metrics.metric_ops import streaming_concat
from tf_slim.metrics.metric_ops import streaming_covariance
from tf_slim.metrics.metric_ops import streaming_curve_points
from tf_slim.metrics.metric_ops import streaming_dynamic_auc
from tf_slim.metrics.metric_ops import streaming_false_negative_rate
from tf_slim.metrics.metric_ops import streaming_false_negative_rate_at_thresholds
from tf_slim.metrics.metric_ops import streaming_false_negatives
from tf_slim.metrics.metric_ops import streaming_false_negatives_at_thresholds
from tf_slim.metrics.metric_ops import streaming_false_positive_rate
from tf_slim.metrics.metric_ops import streaming_false_positive_rate_at_thresholds
from tf_slim.metrics.metric_ops import streaming_false_positives
from tf_slim.metrics.metric_ops import streaming_false_positives_at_thresholds
from tf_slim.metrics.metric_ops import streaming_mean
from tf_slim.metrics.metric_ops import streaming_mean_absolute_error
from tf_slim.metrics.metric_ops import streaming_mean_cosine_distance
from tf_slim.metrics.metric_ops import streaming_mean_iou
from tf_slim.metrics.metric_ops import streaming_mean_relative_error
from tf_slim.metrics.metric_ops import streaming_mean_squared_error
from tf_slim.metrics.metric_ops import streaming_mean_tensor
from tf_slim.metrics.metric_ops import streaming_pearson_correlation
from tf_slim.metrics.metric_ops import streaming_percentage_less
from tf_slim.metrics.metric_ops import streaming_precision
from tf_slim.metrics.metric_ops import streaming_precision_at_thresholds
from tf_slim.metrics.metric_ops import streaming_recall
from tf_slim.metrics.metric_ops import streaming_recall_at_k
from tf_slim.metrics.metric_ops import streaming_recall_at_thresholds
from tf_slim.metrics.metric_ops import streaming_root_mean_squared_error
from tf_slim.metrics.metric_ops import streaming_sensitivity_at_specificity
from tf_slim.metrics.metric_ops import streaming_sparse_average_precision_at_k
from tf_slim.metrics.metric_ops import streaming_sparse_average_precision_at_top_k
from tf_slim.metrics.metric_ops import streaming_sparse_precision_at_k
from tf_slim.metrics.metric_ops import streaming_sparse_precision_at_top_k
from tf_slim.metrics.metric_ops import streaming_sparse_recall_at_k
from tf_slim.metrics.metric_ops import streaming_specificity_at_sensitivity
from tf_slim.metrics.metric_ops import streaming_true_negatives
from tf_slim.metrics.metric_ops import streaming_true_negatives_at_thresholds
from tf_slim.metrics.metric_ops import streaming_true_positives
from tf_slim.metrics.metric_ops import streaming_true_positives_at_thresholds
