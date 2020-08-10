import re

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend
from tensorflow.python.eager import context

from .loss_scale_manager import LossScaleManager


# pylint: disable=invalid-name
class AdamWeightDecayOptimizer(optimizer_v2.OptimizerV2):
  """Optimizer that implements the Adam with weight decay."""

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               exclude_from_weight_decay=None,
               name='AdamWeightDecayOptimizer',
               **kwargs):
    r"""Constructs a AdamWeightDecayOptimizer.
    Just adding the square of the weights to the loss function is *not* the
    correct way of using L2 regularization/weight decay with Adam,
    see https://openreview.net/pdf?id=rk6qdGgCZ.

    If amsgrad = False:
      initialize $m_0$ as 1st moment vector
      initialize $v_0$ as 2nd moment vector
      The update rule for $\theta$ with gradient $g$ uses an optimization
      described at the end of section 2 of the paper:
      $$lr_t = \mathrm{learning\_rate} *
        \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
      $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
      $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
      $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
    If amsgrad = True:
      initialize $m_0$ as 1st moment vector
      initialize $v_0$ as 2nd moment vector
      initialize $\hat{v}_0$ as 2nd moment vector
      The update rule for $\theta$ with gradient $g$ uses an optimization
      described at the end of section 2 of the paper:
      $$lr_t = \mathrm{learning\_rate} *
        \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
      $$m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
      $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g^2$$
      $$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$
      $$\theta_t = \theta_{t-1} - lr_t * m_t / (\sqrt{\hat{v}_t} + \epsilon)$$
    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.
    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    Usage:
    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> # The first step is `-learning_rate*sign(grad)`
    >>> var1.numpy()
    9.9
    Args:
      learning_rate (float):: A `Tensor`, floating point value, or a schedule
        that is a `tf.keras.optimizers.schedules.LearningRateSchedule`,
        or a callable that takes no arguments and returns the actual value to
        use, The learning rate. Defaults to 0.001.
      weight_decay_rate (float): A float value or a constant float tensor. The
        weight decay rate.
      beta_1 (float): A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates. Defaults to 0.9.
      beta_2 (float): A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use, The
        exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
      epsilon (float): A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      amsgrad (bool): Boolean. Whether to apply AMSGrad variant of this
        algorithm from the paper "On the Convergence of Adam and beyond".
        Defaults to `False`.
      exclude_from_weight_decay (list): Some variables should be exclude from
        weight decay, like `layer_norm` and `batch_norm`.
      name (str): Optional name for the operations created when applying
        gradients. Defaults to "AdamWeightDecayOptimizer".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """

    super(AdamWeightDecayOptimizer, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.weight_decay_rate = weight_decay_rate
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.amsgrad = amsgrad

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecayOptimizer, self)._prepare_local(var_device, var_dtype,
                                                         apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
          (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
      dict(
        lr=lr,
        epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
        weight_decay_rate=ops.convert_to_tensor_v2(
          self.weight_decay_rate, var_dtype),
        beta_1_t=beta_1_t,
        beta_1_power=beta_1_power,
        one_minus_beta_1_t=1 - beta_1_t,
        beta_2_t=beta_2_t,
        beta_2_power=beta_2_power,
        one_minus_beta_2_t=1 - beta_2_t))

  def set_weights(self, weights):
    """Restore optimizer params.

    Args:
      weights (list): list of tensors.

    """
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(AdamWeightDecayOptimizer, self).set_weights(weights)

  # pylint: disable=arguments-differ
  def _resource_apply_dense(self, grad, var, apply_state=None):
    """Apply dense updates to variables."""
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = m * coefficients["beta_1_t"] + m_scaled_g_values
    m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = v * coefficients["beta_2_t"] + v_scaled_g_values
    v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)

    if not self.amsgrad:  # pylint: disable=no-else-return
      v_sqrt = math_ops.sqrt(v_t)
      update = m_t / (v_sqrt + coefficients['epsilon'])
      if self._do_use_weight_decay(var.name):
        update += coefficients["weight_decay_rate"] * var
      var_update = state_ops.assign_sub(
        var, coefficients['lr'] * update, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat_t = math_ops.maximum(v_hat, v_t)
      with ops.control_dependencies([v_hat_t]):
        v_hat_t = state_ops.assign(
          v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = math_ops.sqrt(v_hat_t)
      update = m_t / (v_hat_sqrt + coefficients['epsilon'])
      if self._do_use_weight_decay(var.name):
        update += coefficients["weight_decay_rate"] * var
      var_update = state_ops.assign_sub(
        var, coefficients['lr'] * update, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

  # pylint: disable=arguments-differ
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    """Apply sparse updates to variables."""
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    if not self.amsgrad:  # pylint: disable=no-else-return
      v_sqrt = math_ops.sqrt(v_t)
      update = m_t / (v_sqrt + coefficients['epsilon'])
      if self._do_use_weight_decay(var.name):
        update += coefficients["weight_decay_rate"] * var
      var_update = state_ops.assign_sub(
        var, coefficients['lr'] * update, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat_t = math_ops.maximum(v_hat, v_t)
      with ops.control_dependencies([v_hat_t]):
        v_hat_t = state_ops.assign(
          v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = math_ops.sqrt(v_hat_t)
      update = m_t / (v_hat_sqrt + coefficients['epsilon'])
      if self._do_use_weight_decay(var.name):
        update += coefficients["weight_decay_rate"] * var
      var_update = state_ops.assign_sub(
        var, coefficients['lr'] * update, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

  def get_config(self):
    """Get optimizer config."""
    config = super(AdamWeightDecayOptimizer, self).get_config()
    config.update({
      'learning_rate': self._serialize_hyperparameter('learning_rate'),
      'weight_decay_rate': self.weight_decay_rate,
      'exclude_from_weight_decay': self.exclude_from_weight_decay,
      'decay': self._serialize_hyperparameter('decay'),
      'beta_1': self._serialize_hyperparameter('beta_1'),
      'beta_2': self._serialize_hyperparameter('beta_2'),
      'epsilon': self.epsilon,
      'amsgrad': self.amsgrad,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`.

    Args:
      param_name (str): Name of variable.

    Returns:
      bool: Use weight decay or not.
    """
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for exclude_var_name in self.exclude_from_weight_decay:
        if re.search(exclude_var_name, param_name) is not None:
          return False
    return True

  @staticmethod
  def _get_variable_name(param_name):
    """Get the variable name from the tensor name.

    Args:
      param_name (str): tensor name.

    Returns:
      str: Clean variable name.
    """
    re_m = re.match("^(.*):\\d+$", param_name)
    if re_m is not None:
      param_name = re_m.group(1)
    return param_name


# pylint: disable=abstract-method,protected-access
class MixedPrecisionOptimizerWrapper(optimizer_v2.OptimizerV2):
  """"An optimizer that useful for `mixed precision training`."""

  def __init__(self, optimizer, loss_scale=None, clip_norm=None):
    """An optimizer that applies loss scaling in backprop.

    This class is useful for "mixed precision training" on GPUs (or other
    potential accelerators), an approach to improve compute throughput without
    compromising model quality.

    Args:
      optimizer (tf.train.Optimizer): Base optimizer.
      loss_scale (float or LossScaleManager): Scale loss to prevent underflow.
      clip_norm (float): Optional float, if set the gradient will be clipped.

    """
    super(MixedPrecisionOptimizerWrapper, self).__init__(
      optimizer._name + '-MP')
    self._optimizer = optimizer
    self.clip_norm = clip_norm
    self._loss_scaler = None
    self._fp16_to_fp32 = {}
    if loss_scale is None:
      self._loss_scale = 1.0
    elif isinstance(loss_scale, LossScaleManager):
      self._loss_scaler = loss_scale  # type: LossScaleManager
      self._loss_scale = self._loss_scaler.get_loss_scale()
    else:
      self._loss_scale = loss_scale
    self._track_trackable(self._optimizer, 'base_optimizer')

  def _create_slot_weights(self, var_list):
    """Separate fp32 copy for var, we can not use add_slot."""
    for var in var_list:
      if (var.dtype.base_dtype == tf.float16) and (
          var.name not in self._fp16_to_fp32):
        weight = self.add_weight(
          name="%s/%s" % (var._shared_name, "fp32_copy"),
          shape=var.shape,
          dtype=tf.float32,
          trainable=False,
          initializer=math_ops.cast(var.read_value(), tf.float32))
        self._fp16_to_fp32[var.name] = weight
        backend.track_variable(weight)
        self._weights.append(weight)

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    """Compute gradients of `loss` for the variables in `var_list`.
    See base class @{optimizer_v2.OptimizerV2}.
    Also see https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/mixed_precision/experimental/loss_scale_optimizer.py#L52-L374  # pylint: disable=line-too-long

    Args:
      loss (tf.Tensor): A Tensor containing the value to minimize or a callable
        taking no arguments which returns the value to minimize.
      var_list (list): Optional list or tuple of `tf.Variable` to update to
        minimize `loss`.  Defaults to the list of variables collected in the
        graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      grad_loss (tf.Tensor): Optional. A `Tensor` holding the gradient computed
        for `loss`.

    Returns:
      list: A list of (gradient, variable) pairs. Variable is always present,
        but gradient can be `None`.
    """
    if callable(loss):
      loss_val = loss()
    else:
      loss_val = loss
    scaled_loss = loss_val * math_ops.cast(self._loss_scale,
                                           loss_val.dtype.base_dtype)

    # NOTE: grad in `grads_and_vars_fp16` may be None.
    grads_and_vars_fp16 = self._optimizer.compute_gradients(
      scaled_loss, var_list=var_list, grad_loss=grad_loss)

    scaled_grads = self._down_scale(
      [grad_var[0] for grad_var in grads_and_vars_fp16], self._loss_scale)
    scaled_grads_and_vars = list(zip(
      scaled_grads, [grad_var[1] for grad_var in grads_and_vars_fp16]))
    return scaled_grads_and_vars

  # pylint: disable=arguments-differ
  def apply_gradients(self, grads_and_vars, name=None):
    """Apply gradients. See base class @{optimizer_v2.OptimizerV2}.

    Args:
      grads_and_vars (list): List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      name (str): Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.

    Returns:
      tf.Operation: An `Operation` that applies the specified gradients.
        If `global_step` was not None, that operation also increments
        `global_step`.
    """
    grads_and_vars = tuple(grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]
    with backend.name_scope(self._name):
      # Create iteration if necessary.
      with ops.init_scope():
        self._create_all_weights(var_list)

      if not grads_and_vars:
        # Distribution strategy does not support reducing an empty list of
        # gradients
        return control_flow_ops.no_op()

      grads_and_vars_fp32 = []
      for grad, var in grads_and_vars:
        if (grad is not None) and (var.name in self._fp16_to_fp32):
          fp32_var = self._fp16_to_fp32[var.name]
          fp32_grad = math_ops.cast(grad, tf.float32)
          grads_and_vars_fp32.append((fp32_grad, fp32_var))
        else:
          grads_and_vars_fp32.append((grad, var))

      def apply_ops_wrapper():
        grads_vars_to_apply = grads_and_vars_fp32
        if self.clip_norm:
          grads = [grad for (grad, _) in grads_and_vars_fp32]
          grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)
          grads_vars_to_apply = [(clip_grad, var) for clip_grad, (_, var) in
                                 zip(grads, grads_and_vars_fp32)]

        update_op = self._optimizer.apply_gradients(grads_vars_to_apply, name)
        if not context.executing_eagerly():
          # If the current context is graph mode or any of the update ops are
          # symbolic then the step update should be carried out under a graph
          # context. (eager updates execute immediately)
          with ops._get_graph_from_inputs([update_op]).as_default():
            with ops.control_dependencies([update_op]):
              apply_ops = []
              for grad, var in grads_and_vars:
                if (grad is not None) and (var.name in self._fp16_to_fp32):
                  src_var = self._fp16_to_fp32[var.name]
                  apply_ops.append(
                    var.assign(math_ops.saturate_cast(src_var, tf.float16)))
              if apply_ops:
                return control_flow_ops.group(*apply_ops)
        return update_op

      grads = [g for (g, _) in grads_and_vars]
      is_overall_finite = self._check_grads(grads)
      update_vars = control_flow_ops.cond(is_overall_finite, apply_ops_wrapper,
                                          control_flow_ops.no_op)

      if self._loss_scaler:
        return control_flow_ops.group(
          update_vars, self._loss_scaler.update_loss_scale(is_overall_finite))

      return update_vars

  def get_scaled_loss(self, loss):
    """Scales the loss by the loss scale.
    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.
    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for
    an example.
    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
        a tensor or a callable returning a tensor.
    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale()`.
    """
    loss_scale = self._loss_scale
    if callable(loss):
      # return callable loss
      def new_loss():
        loss_val = loss()
        return loss_val * math_ops.cast(loss_scale, loss_val.dtype)

      return new_loss
    # return scalar loss
    return loss * math_ops.cast(loss_scale, loss.dtype)

  def get_gradients(self, loss, params):
    """Get unscaled gradients.

    Args:
      loss (float or callable): Can either be a tensor or a callable returning
        a tensor.
      params ():

    Returns:
      list: A list of tensors, each which will be divided by the loss scale.
        Can have None values, which are ignored.
    """
    # get unscaled gr
    if callable(loss):
      loss_val = loss()
    else:
      loss_val = loss
    scaled_loss = loss_val * math_ops.cast(self._loss_scale,
                                           loss_val.dtype.base_dtype)
    grads = self._optimizer.get_gradients(scaled_loss, params)
    return self._down_scale(grads, self._loss_scale)

  def get_unscaled_gradients(self, grads):
    """Unscales the gradients by the loss scale.
    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to unscale the gradients
    after computing them with `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.
    If this method is called, `get_scaled_loss` should also be called. See
    the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for an
    example.
    Args:
      grads: A list of tensors, each which will be divided by the loss scale.
        Can have None values, which are ignored.
    Returns:
      A new list the same size as `grads`, where every non-None value in `grads`
      is divided by `LossScaleOptimizer.loss_scale()`.
    """
    return self._down_scale(grads, self._loss_scale)

  @staticmethod
  def _check_grads(grads):
    """Check all gradients are finite.

    Args:
      grads (list): list of gradients.

    Returns:
      tf.Tensor: bool type indicates whether all gradients are finite or not.

    """
    is_finite_grad = []
    for grad in grads:
      if grad is None:
        continue
      is_finite_grad.append(tf.math.reduce_all(tf.math.is_finite(grad)))
    is_overall_finite = tf.math.reduce_all(is_finite_grad)
    return is_overall_finite

  @staticmethod
  def _down_scale(grads, loss_scale):
    """Down scale gradients by loss_scale.

    Args:
      grads (list): List of gradients as returned by
        `compute_gradients()`.
      loss_scale (tf.Tensor): Float value of scale.

    Returns:
      list: List of gradients scaled by `loss_scale`.

    """
    # Down scale grads by the loss_scale.
    scaled_grads = []
    inv_loss_scale = tf.math.reciprocal(loss_scale)
    for grad in grads:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          # no significant difference
          grad_values = grad.values * math_ops.cast(inv_loss_scale,
                                                    grad.dtype.base_dtype)
          grad = tf.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
        else:
          grad *= math_ops.cast(inv_loss_scale, grad.dtype.base_dtype)
        scaled_grads.append(grad)
      else:
        scaled_grads.append(grad)
    return scaled_grads

  def _create_all_weights(self, var_list):
    """Creates all weights, including iterations, hyperparameters and slot vars.
    This will add newly created variables to `optimizer.weights`.
    New variables are only created when this method is called the first time, or
    when called with different variables in the var_list.
    Args:
      var_list: list or tuple of `Variable` objects that will be minimized
        using this optimizer.
    """
    _ = self.iterations
    self._create_hypers()
    self._create_slot_weights(var_list)

  @property
  def iterations(self):
    """Return update iterations of this optimizer."""
    return self._optimizer.iterations

  @iterations.setter
  def iterations(self, variable):
    self._optimizer.iterations = variable

  @property
  def learning_rate(self):
    """Return learning_rate of this optimizer."""
    return self._optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, lr):
    self._optimizer.learning_rate = lr

  @property
  def lr(self):
    """Return learning_rate of this optimizer."""
    return self._optimizer.lr

  @property
  def loss_scale(self):
    """The `LossScale` instance associated with this optimizer."""
    return self._loss_scale

  def get_config(self):
    """Get MixedPrecisionOptimizerWrapper config."""
    config = super(MixedPrecisionOptimizerWrapper, self).get_config()
    serialized_optimizer = optimizers.serialize(self._optimizer)
    config.update({'optimizer': serialized_optimizer})
    # TODO: add loss scaler
    return config
