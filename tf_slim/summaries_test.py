# coding=utf-8
# coding=utf-8
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf_slim.summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tempfile
import tensorflow.compat.v1 as tf
from tf_slim import summaries
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.summary import summary_iterator
# pylint:enable=g-direct-tensorflow-import


def setUpModule():
  tf.disable_eager_execution()


class SummariesTest(test.TestCase):

  def assert_scalar_summary(self, output_dir, names_to_values):
    """Asserts that the given output directory contains written summaries.

    Args:
      output_dir: The output directory in which to look for even tfiles.
      names_to_values: A dictionary of summary names to values.
    """
    # The events file may have additional entries, e.g. the event version
    # stamp, so have to parse things a bit.
    output_filepath = glob.glob(os.path.join(output_dir, '*'))
    self.assertEqual(len(output_filepath), 1)

    events = summary_iterator.summary_iterator(output_filepath[0])
    summaries_list = [e.summary for e in events if e.summary.value]
    values = []
    for item in summaries_list:
      for value in item.value:
        values.append(value)
    saved_results = {v.tag: v.simple_value for v in values}
    for name in names_to_values:
      self.assertAlmostEqual(names_to_values[name], saved_results[name])

  def testScalarSummaryIsPartOfCollectionWithNoPrint(self):
    tensor = array_ops.ones([]) * 3
    name = 'my_score'
    prefix = 'eval'
    op = summaries.add_scalar_summary(tensor, name, prefix, print_summary=False)
    self.assertIn(op, ops.get_collection(ops.GraphKeys.SUMMARIES))

  def testScalarSummaryIsPartOfCollectionWithPrint(self):
    tensor = array_ops.ones([]) * 3
    name = 'my_score'
    prefix = 'eval'
    op = summaries.add_scalar_summary(tensor, name, prefix, print_summary=True)
    self.assertIn(op, ops.get_collection(ops.GraphKeys.SUMMARIES))

  def verify_scalar_summary_is_written(self, print_summary):
    value = 3
    tensor = array_ops.ones([]) * value
    name = 'my_score'
    prefix = 'eval'
    summaries.add_scalar_summary(tensor, name, prefix, print_summary)

    output_dir = tempfile.mkdtemp('scalar_summary_no_print_test')
    summary_op = summary.merge_all()

    summary_writer = summary.FileWriter(output_dir)
    with self.cached_session() as sess:
      new_summary = sess.run(summary_op)
      summary_writer.add_summary(new_summary, 1)
      summary_writer.flush()

    self.assert_scalar_summary(output_dir, {
        '%s/%s' % (prefix, name): value
    })

  def testScalarSummaryIsWrittenWithNoPrint(self):
    self.verify_scalar_summary_is_written(print_summary=False)

  def testScalarSummaryIsWrittenWithPrint(self):
    self.verify_scalar_summary_is_written(print_summary=True)


if __name__ == '__main__':
  test.main()
