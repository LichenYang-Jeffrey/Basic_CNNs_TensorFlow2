#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Authors: Shenjian Zhao
#   Date: 2020/01/16 1:57 PM
#
# pylint: skip-file
"""Gradient compression algorithms."""

import tensorflow as tf


class Compressor(object):
  """Interface for compressing and decompressing a given tensor."""

  @staticmethod
  def compress(tensor):
    """Compresses a tensor and returns it with the context needed to decompress.
    """
    pass

  @staticmethod
  def decompress(tensor, ctx):
    """Decompress the tensor with the given context."""
    pass


class NoneCompressor(Compressor):
  """Default no-op compression."""

  @staticmethod
  def compress(tensor):
    """Returns the tensor unmodified."""
    return tensor, None

  @staticmethod
  def decompress(tensor, ctx):
    """Returns the tensor unmodified."""
    return tensor


class FP16Compressor(Compressor):
  """Compress all floating point gradients to 16-bit."""

  @staticmethod
  def compress(tensor):
    """Downcasts the tensor to 16-bit."""
    tensor_compressed = tensor
    if tensor.dtype.is_floating:
      # Only allow compression from other floating point types
      tensor_compressed = tf.cast(tensor, dtype=tf.float16)
    return tensor_compressed, tensor.dtype

  @staticmethod
  def decompress(tensor, ctx):
    """Upcasts the tensor to the initialization dtype."""
    tensor_decompressed = tensor
    dtype = ctx
    if dtype.is_floating:
      tensor_decompressed = tf.cast(tensor, dtype=dtype)
    return tensor_decompressed


class SaturateFP16Compressor(FP16Compressor):
  @staticmethod
  def compress(tensor):
    """Downcasts the tensor to 16-bit."""
    tensor_compressed = tensor
    if tensor.dtype.is_floating:
      # Only allow compression from other floating point types
      tensor_compressed = tf.saturate_cast(tensor, dtype=tf.float16)
    return tensor_compressed, tensor.dtype


class Compression(object):
  """Optional gradient compression algorithm used during allreduce."""

  """Do not compress the gradients. This is the default."""
  none = NoneCompressor

  """Compress all floating point gradients to 16-bit."""
  fp16 = FP16Compressor

  """Compress all floating point gradients to 16-bit with saturate cast."""
  saturate_fp16 = SaturateFP16Compressor
