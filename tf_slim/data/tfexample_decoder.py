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
"""Contains the TFExampleDecoder its associated helper classes.

The TFExampleDecode is a DataDecoder used to decode TensorFlow Example protos.
In order to do so each requested item must be paired with one or more Example
features that are parsed to produce the Tensor-based manifestation of the item.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
from tf_slim.data import data_decoder
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
# pylint:enable=g-direct-tensorflow-import


@six.add_metaclass(abc.ABCMeta)
class ItemHandler(object):
  """Specifies the item-to-Features mapping for tf.parse_example.

  An ItemHandler both specifies a list of Features used for parsing an Example
  proto as well as a function that post-processes the results of Example
  parsing.
  """

  def __init__(self, keys):
    """Constructs the handler with the name of the tf.train.Feature keys to use.

    Args:
      keys: the name of the TensorFlow Example Feature.
    """
    if not isinstance(keys, (tuple, list)):
      keys = [keys]
    self._keys = keys

  @property
  def keys(self):
    return self._keys

  @abc.abstractmethod
  def tensors_to_item(self, keys_to_tensors):
    """Maps the given dictionary of tensors to the requested item.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      the final tensor representing the item being handled.
    """
    pass


class ItemHandlerCallback(ItemHandler):
  """An ItemHandler that converts the parsed tensors via a given function.

  Unlike other ItemHandlers, the ItemHandlerCallback resolves its item via
  a callback function rather than using prespecified behavior.
  """

  def __init__(self, keys, func):
    """Initializes the ItemHandler.

    Args:
      keys: a list of TF-Example keys.
      func: a function that takes as an argument a dictionary from `keys` to
        parsed Tensors.
    """
    super(ItemHandlerCallback, self).__init__(keys)
    self._func = func

  def tensors_to_item(self, keys_to_tensors):
    return self._func(keys_to_tensors)


class BoundingBox(ItemHandler):
  """An ItemHandler that concatenates a set of parsed Tensors to Bounding Boxes.
  """

  def __init__(self, keys=None, prefix=''):
    """Initialize the bounding box handler.

    Args:
      keys: A list of four key names representing the ymin, xmin, ymax, mmax
      prefix: An optional prefix for each of the bounding box keys.
        If provided, `prefix` is appended to each key in `keys`.

    Raises:
      ValueError: if keys is not `None` and also not a list of exactly 4 keys
    """
    if keys is None:
      keys = ['ymin', 'xmin', 'ymax', 'xmax']
    elif len(keys) != 4:
      raise ValueError('BoundingBox expects 4 keys but got {}'.format(
          len(keys)))
    self._prefix = prefix
    self._keys = keys
    self._full_keys = [prefix + k for k in keys]
    super(BoundingBox, self).__init__(self._full_keys)

  def tensors_to_item(self, keys_to_tensors):
    """Maps the given dictionary of tensors to a concatenated list of bboxes.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      [num_boxes, 4] tensor of bounding box coordinates,
        i.e. 1 bounding box per row, in order [y_min, x_min, y_max, x_max].
    """
    sides = []
    for key in self._full_keys:
      side = keys_to_tensors[key]
      if isinstance(side, sparse_tensor.SparseTensor):
        side = side.values
      side = array_ops.expand_dims(side, 0)
      sides.append(side)

    bounding_box = array_ops.concat(sides, 0)
    return array_ops.transpose(bounding_box)


class Tensor(ItemHandler):
  """An ItemHandler that returns a parsed Tensor."""

  def __init__(self, tensor_key, shape_keys=None, shape=None, default_value=0):
    """Initializes the Tensor handler.

    Tensors are, by default, returned without any reshaping. However, there are
    two mechanisms which allow reshaping to occur at load time. If `shape_keys`
    is provided, both the `Tensor` corresponding to `tensor_key` and
    `shape_keys` is loaded and the former `Tensor` is reshaped with the values
    of the latter. Alternatively, if a fixed `shape` is provided, the `Tensor`
    corresponding to `tensor_key` is loaded and reshape appropriately.
    If neither `shape_keys` nor `shape` are provided, the `Tensor` will be
    returned without any reshaping.

    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      shape_keys: Optional name or list of names of the TF-Example feature in
        which the tensor shape is stored. If a list, then each corresponds to
        one dimension of the shape.
      shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
        reshaped accordingly.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if both `shape_keys` and `shape` are specified.
    """
    if shape_keys and shape is not None:
      raise ValueError('Cannot specify both shape_keys and shape parameters.')
    if shape_keys and not isinstance(shape_keys, list):
      shape_keys = [shape_keys]
    self._tensor_key = tensor_key
    self._shape_keys = shape_keys
    self._shape = shape
    self._default_value = default_value
    keys = [tensor_key]
    if shape_keys:
      keys.extend(shape_keys)
    super(Tensor, self).__init__(keys)

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self._tensor_key]
    shape = self._shape
    if self._shape_keys:
      shape_dims = []
      for k in self._shape_keys:
        shape_dim = keys_to_tensors[k]
        if isinstance(shape_dim, sparse_tensor.SparseTensor):
          shape_dim = sparse_ops.sparse_tensor_to_dense(shape_dim)
        shape_dims.append(shape_dim)
      shape = array_ops.reshape(array_ops.stack(shape_dims), [-1])
    if isinstance(tensor, sparse_tensor.SparseTensor):
      if shape is not None:
        tensor = sparse_ops.sparse_reshape(tensor, shape)
      tensor = sparse_ops.sparse_tensor_to_dense(tensor, self._default_value)
    else:
      if shape is not None:
        tensor = array_ops.reshape(tensor, shape)
    return tensor


class LookupTensor(Tensor):
  """An ItemHandler that returns a parsed Tensor, the result of a lookup."""

  def __init__(self,
               tensor_key,
               table,
               shape_keys=None,
               shape=None,
               default_value=''):
    """Initializes the LookupTensor handler.

    See Tensor.  Simply calls a vocabulary (most often, a label mapping) lookup.

    Args:
      tensor_key: the name of the `TFExample` feature to read the tensor from.
      table: A tf.lookup table.
      shape_keys: Optional name or list of names of the TF-Example feature in
        which the tensor shape is stored. If a list, then each corresponds to
        one dimension of the shape.
      shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
        reshaped accordingly.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if both `shape_keys` and `shape` are specified.
    """
    self._table = table
    super(LookupTensor, self).__init__(tensor_key, shape_keys, shape,
                                       default_value)

  def tensors_to_item(self, keys_to_tensors):
    unmapped_tensor = super(LookupTensor, self).tensors_to_item(keys_to_tensors)
    return self._table.lookup(unmapped_tensor)


class BackupHandler(ItemHandler):
  """An ItemHandler that tries two ItemHandlers in order."""

  def __init__(self, handler, backup):
    """Initializes the BackupHandler handler.

    If the first Handler's tensors_to_item returns a Tensor with no elements,
    the second Handler is used.

    Args:
      handler: The primary ItemHandler.
      backup: The backup ItemHandler.

    Raises:
      ValueError: if either is not an ItemHandler.
    """
    if not isinstance(handler, ItemHandler):
      raise ValueError('Primary handler is of type %s instead of ItemHandler'
                       % type(handler))
    if not isinstance(backup, ItemHandler):
      raise ValueError('Backup handler is of type %s instead of ItemHandler'
                       % type(backup))
    self._handler = handler
    self._backup = backup
    super(BackupHandler, self).__init__(handler.keys + backup.keys)

  def tensors_to_item(self, keys_to_tensors):
    item = self._handler.tensors_to_item(keys_to_tensors)
    return control_flow_ops.cond(
        pred=math_ops.equal(math_ops.reduce_prod(array_ops.shape(item)), 0),
        true_fn=lambda: self._backup.tensors_to_item(keys_to_tensors),
        false_fn=lambda: item)


class SparseTensor(ItemHandler):
  """An ItemHandler for SparseTensors."""

  def __init__(self,
               indices_key=None,
               values_key=None,
               shape_key=None,
               shape=None,
               densify=False,
               default_value=0):
    """Initializes the Tensor handler.

    Args:
      indices_key: the name of the TF-Example feature that contains the ids.
        Defaults to 'indices'.
      values_key: the name of the TF-Example feature that contains the values.
        Defaults to 'values'.
      shape_key: the name of the TF-Example feature that contains the shape.
        If provided it would be used.
      shape: the output shape of the SparseTensor. If `shape_key` is not
        provided this `shape` would be used.
      densify: whether to convert the SparseTensor into a dense Tensor.
      default_value: Scalar value to set when making dense for indices not
        specified in the `SparseTensor`.
    """
    indices_key = indices_key or 'indices'
    values_key = values_key or 'values'
    self._indices_key = indices_key
    self._values_key = values_key
    self._shape_key = shape_key
    self._shape = shape
    self._densify = densify
    self._default_value = default_value
    keys = [indices_key, values_key]
    if shape_key:
      keys.append(shape_key)
    super(SparseTensor, self).__init__(keys)

  def tensors_to_item(self, keys_to_tensors):
    indices = keys_to_tensors[self._indices_key]
    values = keys_to_tensors[self._values_key]
    if self._shape_key:
      shape = keys_to_tensors[self._shape_key]
      if isinstance(shape, sparse_tensor.SparseTensor):
        shape = sparse_ops.sparse_tensor_to_dense(shape)
    elif self._shape:
      shape = self._shape
    else:
      shape = indices.dense_shape
    indices_shape = array_ops.shape(indices.indices)
    rank = indices_shape[1]
    ids = math_ops.cast(indices.values, dtypes.int64)
    indices_columns_to_preserve = array_ops.slice(
        indices.indices, [0, 0], array_ops.stack([-1, rank - 1]))
    new_indices = array_ops.concat(
        [indices_columns_to_preserve, array_ops.reshape(ids, [-1, 1])], 1)

    tensor = sparse_tensor.SparseTensor(new_indices, values.values, shape)
    if self._densify:
      tensor = sparse_ops.sparse_tensor_to_dense(tensor, self._default_value)
    return tensor


class Image(ItemHandler):
  """An ItemHandler that decodes a parsed Tensor as an image."""

  def __init__(self,
               image_key=None,
               format_key=None,
               shape=None,
               channels=3,
               dtype=dtypes.uint8,
               repeated=False,
               dct_method=''):
    """Initializes the image.

    Args:
      image_key: the name of the TF-Example feature in which the encoded image
        is stored.
      format_key: the name of the TF-Example feature in which the image format
        is stored.
      shape: the output shape of the image as 1-D `Tensor`
        [height, width, channels]. If provided, the image is reshaped
        accordingly. If left as None, no reshaping is done. A shape should
        be supplied only if all the stored images have the same shape.
      channels: the number of channels in the image.
      dtype: images will be decoded at this bit depth. Different formats
        support different bit depths.
          See tf.image.decode_image,
              tf.io.decode_raw,
      repeated: if False, decodes a single image. If True, decodes a
        variable number of image strings from a 1D tensor of strings.
      dct_method: An optional string. Defaults to empty string. It only takes
        effect when image format is jpeg, used to specify a hint about the
        algorithm used for jpeg decompression. Currently valid values
        are ['INTEGER_FAST', 'INTEGER_ACCURATE']. The hint may be ignored, for
        example, the jpeg library does not have that specific option.
    """
    if not image_key:
      image_key = 'image/encoded'
    if not format_key:
      format_key = 'image/format'

    super(Image, self).__init__([image_key, format_key])
    self._image_key = image_key
    self._format_key = format_key
    self._shape = shape
    self._channels = channels
    self._dtype = dtype
    self._repeated = repeated
    self._dct_method = dct_method

  def tensors_to_item(self, keys_to_tensors):
    """See base class."""
    image_buffer = keys_to_tensors[self._image_key]
    image_format = keys_to_tensors[self._format_key]

    if self._repeated:
      return map_fn.map_fn(lambda x: self._decode(x, image_format),
                           image_buffer, dtype=self._dtype)
    else:
      return self._decode(image_buffer, image_format)

  def _decode(self, image_buffer, image_format):
    """Decodes the image buffer.

    Args:
      image_buffer: The tensor representing the encoded image tensor.
      image_format: The image format for the image in `image_buffer`. If image
        format is `raw`, all images are expected to be in this format, otherwise
        this op can decode a mix of `jpg` and `png` formats.

    Returns:
      A tensor that represents decoded image of self._shape, or
      (?, ?, self._channels) if self._shape is not specified.
    """

    def decode_image():
      """Decodes a image based on the headers."""
      return math_ops.cast(
          image_ops.decode_image(image_buffer, channels=self._channels),
          self._dtype)

    def decode_jpeg():
      """Decodes a jpeg image with specified '_dct_method'."""
      return math_ops.cast(
          image_ops.decode_jpeg(
              image_buffer,
              channels=self._channels,
              dct_method=self._dct_method), self._dtype)

    def check_jpeg():
      """Checks if an image is jpeg."""
      # For jpeg, we directly use image_ops.decode_jpeg rather than decode_image
      # in order to feed the jpeg specify parameter 'dct_method'.
      return control_flow_ops.cond(
          image_ops.is_jpeg(image_buffer),
          decode_jpeg,
          decode_image,
          name='cond_jpeg')

    def decode_raw():
      """Decodes a raw image."""
      return parsing_ops.decode_raw(image_buffer, out_type=self._dtype)

    pred_fn_pairs = [(math_ops.logical_or(
        math_ops.equal(image_format, 'raw'),
        math_ops.equal(image_format, 'RAW')), decode_raw)]

    image = control_flow_ops.case(
        pred_fn_pairs, default=check_jpeg, exclusive=True)

    image.set_shape([None, None, self._channels])
    if self._shape is not None:
      image = array_ops.reshape(image, self._shape)

    return image


class BoundingBoxSequence(ItemHandler):
  """An ItemHandler that concatenates SparseTensors to Bounding Boxes.
  """

  def __init__(self, keys=None, prefix=None, return_dense=True,
               default_value=-1.0):
    """Initialize the bounding box handler.

    Args:
      keys: A list of four key names representing the ymin, xmin, ymax, xmax
        in the Example or SequenceExample.
      prefix: An optional prefix for each of the bounding box keys in the
        Example or SequenceExample. If provided, `prefix` is prepended to each
        key in `keys`.
      return_dense: if True, returns a dense tensor; if False, returns as
        sparse tensor.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if keys is not `None` and also not a list of exactly 4 keys
    """
    if keys is None:
      keys = ['ymin', 'xmin', 'ymax', 'xmax']
    elif len(keys) != 4:
      raise ValueError('BoundingBoxSequence expects 4 keys but got {}'.format(
          len(keys)))
    self._prefix = prefix
    self._keys = keys
    self._full_keys = [prefix + k for k in keys]
    self._return_dense = return_dense
    self._default_value = default_value
    super(BoundingBoxSequence, self).__init__(self._full_keys)

  def tensors_to_item(self, keys_to_tensors):
    """Maps the given dictionary of tensors to a concatenated list of bboxes.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      [time, num_boxes, 4] tensor of bounding box coordinates, in order
          [y_min, x_min, y_max, x_max]. Whether the tensor is a SparseTensor
          or a dense Tensor is determined by the return_dense parameter. Empty
          positions in the sparse tensor are filled with -1.0 values.
    """
    sides = []
    for key in self._full_keys:
      value = keys_to_tensors[key]
      expanded_dims = array_ops.concat(
          [math_ops.to_int64(array_ops.shape(value)),
           constant_op.constant([1], dtype=dtypes.int64)], 0)
      side = sparse_ops.sparse_reshape(value, expanded_dims)
      sides.append(side)
    bounding_boxes = sparse_ops.sparse_concat(2, sides)
    if self._return_dense:
      bounding_boxes = sparse_ops.sparse_tensor_to_dense(
          bounding_boxes, default_value=self._default_value)
    return bounding_boxes


class NumBoxesSequence(ItemHandler):
  """An ItemHandler that returns num_boxes per frame for a box sequence.

  `num_boxes` is inferred from a 2D SparseTensor decoded from a field in the
  SequenceExample. The SparseTensor is partially dense and only ragged along its
  second dimensions.

  The output is an int64 tf.Tensor of shape [time], which is solely determined
  by the tensor of the first key. However, if `check_consistency` is True, this
  function checks that `num_boxes` is consistent across all keys.
  """

  def __init__(self, keys=None, check_consistency=True):
    """Initialization.

    Args:
      keys: A list of keys of sparse tensors which have exactly 2 dimensions,
        with the 1st being the `time` and the 2nd the `boxes` per frame.
        key in `keys`.
      check_consistency: if True, check for consistency.

    Raises:
      ValueError: If keys is empty.
    """
    if not keys:
      raise ValueError('keys must not be empty.')
    self._check_consistency = check_consistency
    super(NumBoxesSequence, self).__init__(keys)

  def tensors_to_item(self, keys_to_tensors):
    """Maps the given dictionary of tensors to a num_boxes tensor.

    If check_consistency is True: raises runtime error in Tensorflow when the
    consistency is violated across tensors.

    Args:
      keys_to_tensors: A mapping of TF-Example keys to parsed tensors.

    Returns:
      [time] tf.Tensor containing the number of boxes per frame.

    Raises:
      ValueError: If any of the keyed tensors is not sparse or exactly 2
        dimensional.
    """
    def _compute_num_boxes(tensor):
      """Compute num_boxes from a single 2D tensor."""
      if not isinstance(tensor, sparse_tensor.SparseTensor):
        raise ValueError('tensor must be of type tf.SparseTensor.')
      indices = tensor.indices
      dense_shape = tensor.dense_shape
      box_ids = indices[:, 1]
      box_ids = sparse_tensor.SparseTensor(
          indices=indices, values=box_ids, dense_shape=dense_shape)
      box_ids = sparse_ops.sparse_tensor_to_dense(box_ids, default_value=-1)
      # In the event that the parsed tensor is empty (perhaps due to a negative
      # example), we pad box_ids so that the resulting number of boxes is 0.
      num_boxes = math_ops.reduce_max(
          array_ops.pad(box_ids + 1, [[0, 0], [0, 1]]), axis=1)
      return num_boxes

    num_boxes = _compute_num_boxes(keys_to_tensors[self._keys[0]])
    asserts = []
    if self._check_consistency:
      for i in range(1, len(self._keys)):
        cur_num_boxes = _compute_num_boxes(keys_to_tensors[self._keys[i]])
        asserts.append(check_ops.assert_equal(num_boxes, cur_num_boxes))

    with ops.control_dependencies(asserts):
      return array_ops.identity(num_boxes)


class KeypointsSequence(ItemHandler):
  """An ItemHandler that concatenates SparseTensors to Keypoints.
  """

  def __init__(self, keys=None, prefix=None, return_dense=True,
               default_value=-1.0):
    """Initialize the keypoints handler.

    Args:
      keys: A list of two key names representing the y and x coordinates in the
        Example or SequenceExample.
      prefix: An optional prefix for each of the keypoint keys in the Example
        or SequenceExample. If provided, `prefix` is prepended to each key in
        `keys`.
      return_dense: if True, returns a dense tensor; if False, returns as
        sparse tensor.
      default_value: The value used when the `tensor_key` is not found in a
        particular `TFExample`.

    Raises:
      ValueError: if keys is not `None` and also not a list of exactly 2 keys
    """
    if keys is None:
      keys = ['y', 'x']
    elif len(keys) != 2:
      raise ValueError('KeypointsSequence expects 2 keys but got {}'.format(
          len(keys)))
    self._prefix = prefix
    self._keys = keys
    self._full_keys = [prefix + k for k in keys]
    self._return_dense = return_dense
    self._default_value = default_value
    super(KeypointsSequence, self).__init__(self._full_keys)

  def tensors_to_item(self, keys_to_tensors):
    """Maps the given dictionary of tensors to a concatenated list of keypoints.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      [time, num_keypoints, 2] tensor of keypoint coordinates, in order [y, x].
          Whether the tensor is a SparseTensor or a dense Tensor is determined
          by the return_dense parameter. Empty positions in the sparse tensor
          are filled with -1.0 values.
    """
    coordinates = []
    for key in self._full_keys:
      value = keys_to_tensors[key]
      expanded_dims = array_ops.concat(
          [math_ops.to_int64(array_ops.shape(value)),
           constant_op.constant([1], dtype=dtypes.int64)], 0)
      coordinate = sparse_ops.sparse_reshape(value, expanded_dims)
      coordinates.append(coordinate)
    keypoints = sparse_ops.sparse_concat(2, coordinates)
    if self._return_dense:
      keypoints = sparse_ops.sparse_tensor_to_dense(
          keypoints, default_value=self._default_value)
    return keypoints


class TFExampleDecoder(data_decoder.DataDecoder):
  """A decoder for TensorFlow Examples.

  Decoding Example proto buffers is comprised of two stages: (1) Example parsing
  and (2) tensor manipulation.

  In the first stage, the tf.io.parse_example function is called with a list of
  FixedLenFeatures and SparseLenFeatures. These instances tell TF how to parse
  the example. The output of this stage is a set of tensors.

  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.

  To perform this decoding operation, an ExampleDecoder is given a list of
  ItemHandlers. Each ItemHandler indicates the set of features for stage 1 and
  contains the instructions for post_processing its tensors for stage 2.
  """

  def __init__(self, keys_to_features, items_to_handlers):
    """Constructs the decoder.

    Args:
      keys_to_features: a dictionary from TF-Example keys to either
        tf.io.VarLenFeature or tf.io.FixedLenFeature instances. See tensorflow's
        parsing_ops.py.
      items_to_handlers: a dictionary from items (strings) to ItemHandler
        instances. Note that the ItemHandler's are provided the keys that they
        use to return the final item Tensors.
    """
    self._keys_to_features = keys_to_features
    self._items_to_handlers = items_to_handlers

  def list_items(self):
    """See base class."""
    return list(self._items_to_handlers.keys())

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-example.

    Args:
      serialized_example: a serialized TF-example tensor.
      items: the list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.

    Returns:
      the decoded items, a list of tensor.
    """
    example = parsing_ops.parse_single_example(serialized_example,
                                               self._keys_to_features)

    # Reshape non-sparse elements just once, adding the reshape ops in
    # deterministic order.
    for k in sorted(self._keys_to_features):
      v = self._keys_to_features[k]
      if isinstance(v, parsing_ops.FixedLenFeature):
        example[k] = array_ops.reshape(example[k], v.shape)

    if not items:
      items = self._items_to_handlers.keys()

    outputs = []
    for item in items:
      handler = self._items_to_handlers[item]
      keys_to_tensors = {key: example[key] for key in handler.keys}
      outputs.append(handler.tensors_to_item(keys_to_tensors))
    return outputs


class TFSequenceExampleDecoder(data_decoder.DataDecoder):
  """A decoder for TensorFlow SequenceExamples.

  Decoding SequenceExample proto buffers is comprised of two stages:
  (1) Example parsing and (2) tensor manipulation.

  In the first stage, the tf.parse_single_sequence_example function is called
  with a list of FixedLenFeatures and SparseLenFeatures. These instances tell TF
  how to parse the example. The output of this stage is a set of tensors.

  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.

  To perform this decoding operation, a SequenceExampleDecoder is given a list
  of ItemHandlers. Each ItemHandler indicates the set of features for stage 1
  and contains the instructions for post_processing its tensors for stage 2.
  """

  def __init__(self, keys_to_context_features, keys_to_sequence_features,
               items_to_handlers):
    """Constructs the decoder.

    Args:
      keys_to_context_features: a dictionary from TF-SequenceExample context
        keys to either tf.VarLenFeature or tf.FixedLenFeature instances.
        See tensorflow's parsing_ops.py.
      keys_to_sequence_features: a dictionary from TF-SequenceExample sequence
        keys to either tf.VarLenFeature or tf.FixedLenSequenceFeature instances.
        See tensorflow's parsing_ops.py.
      items_to_handlers: a dictionary from items (strings) to ItemHandler
        instances. Note that the ItemHandler's are provided the keys that they
        use to return the final item Tensors.

    Raises:
      ValueError: if the same key is present for context features and sequence
        features.
    """
    unique_keys = set()
    unique_keys.update(keys_to_context_features)
    unique_keys.update(keys_to_sequence_features)
    if len(unique_keys) != (
        len(keys_to_context_features) + len(keys_to_sequence_features)):
      # This situation is ambiguous in the decoder's keys_to_tensors variable.
      raise ValueError('Context and sequence keys are not unique. \n'
                       ' Context keys: %s \n Sequence keys: %s' %
                       (list(keys_to_context_features.keys()),
                        list(keys_to_sequence_features.keys())))

    self._keys_to_context_features = keys_to_context_features
    self._keys_to_sequence_features = keys_to_sequence_features
    self._items_to_handlers = items_to_handlers

  def list_items(self):
    """See base class."""
    return self._items_to_handlers.keys()

  def decode(self, serialized_example, items=None):
    """Decodes the given serialized TF-SequenceExample.

    Args:
      serialized_example: a serialized TF-SequenceExample tensor.
      items: the list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.

    Returns:
      the decoded items, a list of tensor.
    """

    context, feature_list = parsing_ops.parse_single_sequence_example(
        serialized_example, self._keys_to_context_features,
        self._keys_to_sequence_features)

    # Reshape non-sparse elements just once:
    for k in self._keys_to_context_features:
      v = self._keys_to_context_features[k]
      if isinstance(v, parsing_ops.FixedLenFeature):
        context[k] = array_ops.reshape(context[k], v.shape)

    if not items:
      items = self._items_to_handlers.keys()

    outputs = []
    for item in items:
      handler = self._items_to_handlers[item]
      keys_to_tensors = {
          key: context[key] if key in context else feature_list[key]
          for key in handler.keys
      }
      outputs.append(handler.tensors_to_item(keys_to_tensors))
    return outputs
