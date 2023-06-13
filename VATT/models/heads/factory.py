"""Factory to build any type of projection head."""

import logging
from typing import Any, Dict, Mapping, Optional, Text, Union

import tensorflow as tf

from configs import head as head_config
from models.heads import bridge


HEAD_MODULES = {
    "bridge": {
        "mlp_fac": bridge.FACHead,
        "mlp_joint": bridge.JointHead,
    },
}


def merge_dicts(d, u):
  for k, v in u.items():
    if isinstance(v, Mapping):
      d[k] = merge_dicts(d.get(k, {}), v)
    else:
      d[k] = v
  return d


class HeadStack(tf.keras.layers.Layer):
  """Constructs a Head layer given the parameters and head type."""

  def __init__(self,
               head_modules,
               params):
    """HeadStack.

    Args:
      head_modules: a dictionary containing all head modules.
      params: Hyperparameters of the model.

    """
    super(HeadStack, self).__init__(name="head_stack")

    self._bridge_heads = []

    for head_param in params['bridge']:
      head_name = params['bridge']['name'].lower()
      module_kwargs = params['bridge']
      self._bridge_heads.append(
          head_modules["bridge"][head_name](**module_kwargs)
          )

  def call(self,
           inputs,
           training = None,
           mask = None):
    """Call the layer.

    Args:
      inputs: input tensors of different modalities.
      training: True for in the training mode.
      mask: any attention or padding mask (for Transformers)

    Returns:
      output_dict: a dict of model outputs, including necessary input features,
    """
    outputs = {
        "bridge": {},
    }

    with tf.name_scope("bridge"):
      for head_layer in self._bridge_heads:
        outs = head_layer(inputs,
                          training)
        outputs["bridge"] = merge_dicts(outputs["bridge"], outs)

    return outputs


def build_model(
    params,
    override_params = None,
    ):
  """Build model by name."""

  if override_params is not None:
    params.override(override_params)

  heads = {"bridge": {}}

  head_stack_names = []

  for head in params['bridge']:
    head_name = params['bridge']['name'].lower()
    if head_name in HEAD_MODULES["bridge"]:
      heads["bridge"][head_name] = HEAD_MODULES["bridge"][head_name]
      head_stack_names.append(head_name)
    else:
      raise ValueError("Unknown module name {!r}".format(head_name))

  head_stack_layer = HeadStack(heads, params)
  head_stack_name = "+".join(head_stack_names)
  logging.info("Head stack %s created successfully.", head_stack_name)

  return head_stack_layer
