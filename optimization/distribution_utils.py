import json
import os

from absl import logging
import tensorflow as tf


def get_num_gpus(num_gpus):
  """Treat num_gpus=-1 as 'use all'."""
  if (num_gpus != -1) and (num_gpus is not None):
    return num_gpus

  from tensorflow.python.client import device_lib  # pylint: disable=C0415
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])


def _collective_communication(all_reduce_alg):
  """Return a CollectiveCommunication based on all_reduce_alg.
  Args:
    all_reduce_alg (str): A string specifying which collective communication to
      pick, or None.
  Returns:
    tf.distribute.experimental.CollectiveCommunication object
  Raises:
    ValueError: if `all_reduce_alg` not in [None, 'ring', 'nccl']
  """
  collective_communication_options = {
    None: tf.distribute.experimental.CollectiveCommunication.AUTO,
    "ring": tf.distribute.experimental.CollectiveCommunication.RING,
    "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
  }
  if all_reduce_alg not in collective_communication_options:
    raise ValueError(
      "When used with `multi_worker_mirrored`, valid values for "
      "all_reduce_alg are ['ring', 'nccl'].  Supplied value: {}".format(
        all_reduce_alg))
  return collective_communication_options[all_reduce_alg]


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
  """Return a CrossDeviceOps based on all_reduce_alg and num_packs.
  Args:
    all_reduce_alg (str): a string specifying which cross device op to pick,
      or None.
    num_packs (int): an integer specifying number of packs for the cross device
      op.
  Returns:
    tf.distribute.CrossDeviceOps object or None.
  Raises:
    ValueError: if `all_reduce_alg` not in [None, 'nccl', 'hierarchical_copy'].
  """
  if all_reduce_alg is None:
    return None
  mirrored_all_reduce_options = {
    "nccl": tf.distribute.NcclAllReduce,
    "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce
  }
  if all_reduce_alg not in mirrored_all_reduce_options:
    raise ValueError(
      "When used with `mirrored`, valid values for all_reduce_alg are "
      "['nccl', 'hierarchical_copy'].  Supplied value: {}".format(
        all_reduce_alg))
  cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
  return cross_device_ops_class(num_packs=num_packs)


def configure_cluster(worker_hosts=None, task_index=-1):
  """Set multi-worker cluster spec in TF_CONFIG environment variable.
  Args:
    worker_hosts: comma-separated list of worker ip:port pairs.
    task_index:
  Returns:
    Number of workers in the cluster.
  """
  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  if tf_config:
    num_workers = (len(tf_config['cluster'].get('chief', [])) +
                   len(tf_config['cluster'].get('worker', [])))
  elif worker_hosts:
    workers = worker_hosts.split(',')
    num_workers = len(workers)
    if num_workers > 1 and task_index < 0:
      raise ValueError('Must specify task_index when number of workers > 1')
    task_index = 0 if num_workers == 1 else task_index
    os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
        'worker': workers
      },
      'task': {'type': 'worker', 'index': task_index}
    })
  else:
    num_workers = 1
  return num_workers


def get_distribution_strategy(distribution_strategy="mirrored",
                              num_gpus=0,
                              num_workers=1,
                              all_reduce_alg=None,
                              num_packs=1,
                              tpu_address=None):
  """Return a DistributionStrategy for running the model.
  Args:
    distribution_strategy: a string specifying which distribution strategy to
      use. Accepted values are 'off', 'one_device', 'mirrored',
      'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case insensitive.
      'off' means not to use Distribution Strategy; 'tpu' means to use
      TPUStrategy using `tpu_address`.
    num_gpus: Number of GPUs to run this model.
    num_workers: Number of workers to run this model.
    all_reduce_alg: Optional. Specifies which algorithm to use when performing
      all-reduce. For `MirroredStrategy`, valid values are "nccl" and
      "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are
      "ring" and "nccl".  If None, DistributionStrategy will choose based on
      device topology.
    num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`
      or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
    tpu_address: Optional. String that represents TPU to connect to. Must not
      be None if `distribution_strategy` is set to `tpu`.
  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is 'off' or 'one_device' and
      `num_gpus` is larger than 1; or `num_gpus` is negative or if
      `distribution_strategy` is `tpu` but `tpu_address` is not specified.
  """
  if num_gpus < 0:
    raise ValueError("`num_gpus` can not be negative.")

  distribution_strategy = distribution_strategy.lower()
  if distribution_strategy == "off":
    if num_gpus > 1:
      raise ValueError(
        "When {} GPUs and  {} workers are specified, distribution_strategy "
        "flag cannot be set to 'off'.".format(num_gpus, num_workers))
    return None

  if distribution_strategy == "tpu":
    # # When tpu_address is an empty string, we communicate with local TPUs.
    # cluster_resolver = tpu_lib.tpu_initialize(tpu_address)
    # return tf.distribute.experimental.TPUStrategy(cluster_resolver)
    raise NotImplementedError("tpu_address: {}.".format(tpu_address))

  if distribution_strategy == "multi_worker_mirrored":
    return tf.distribute.experimental.MultiWorkerMirroredStrategy(
      communication=_collective_communication(all_reduce_alg))

  if distribution_strategy == "one_device":
    if num_gpus == 0:
      return tf.distribute.OneDeviceStrategy("device:CPU:0")
    if num_gpus > 1:
      raise ValueError("`OneDeviceStrategy` can not be used for more than "
                       "one device.")
    return tf.distribute.OneDeviceStrategy("device:GPU:0")

  if distribution_strategy == "mirrored":
    if num_gpus == 0:
      devices = ["device:CPU:0"]
    else:
      devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    return tf.distribute.MirroredStrategy(
      devices=devices,
      cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))

  if distribution_strategy == "parameter_server":
    return tf.distribute.experimental.ParameterServerStrategy()

  raise ValueError(
    "Unrecognized Distribution Strategy: %r" % distribution_strategy)


# pylint: disable=missing-class-docstring
class DummyContextManager:

  def __enter__(self):
    pass

  def __exit__(self, *args):
    pass


# pylint: disable=missing-function-docstring
def get_strategy_scope(strategy):
  if strategy:
    strategy_scope = strategy.scope()
  else:
    strategy_scope = DummyContextManager()

  return strategy_scope


def broadcast_variables(trainable_variables, optimizer, hvd=None):
  """Broadcast all variables from rank 0 worker.

  Args:
    trainable_variables (list): List of model trainable variables.
    optimizer (Optimizer):
    hvd:

  """
  if hvd is None:
    return
  opt_vars = optimizer.variables()
  if hasattr(optimizer, "_optimizer"):
    # use mix fp16
    opt_vars.extend(optimizer._optimizer.variables())  # pylint: disable=W0212
  hvd.broadcast_variables(trainable_variables, root_rank=0)
  hvd.broadcast_variables(opt_vars, root_rank=0)
  logging.info(
    "{} trainable and {} optimizer variables are synchronized.".format(
      len(trainable_variables), len(opt_vars)))