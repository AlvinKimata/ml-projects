"""Config definition for different types of losses."""

import dataclasses
from typing import Tuple
from configs import base_config


@dataclasses.dataclass
class BaseLossConfig(base_config.Config):
  """Base configs for any type of losses.

  Attributes:
    name: .
    loss_weight: .
  """
  name: str = ""
  loss_weight: float = 1.0


@dataclasses.dataclass
class SymmetricNCE(BaseLossConfig):
  """Parameters for symmetrical nce / mil-nce loss."""

  name: str = "symmetric_nce"
  temperature: float = 0.07
  vid_txt_weight: float = 1.
  vid_aud_weight: float = 1.
  aud_txt_weight: float = 0.


@dataclasses.dataclass
class AsymmetricNCE(SymmetricNCE):
  """Parameters for asymmetrical nce / mil-nce loss."""

  name: str = "asymmetric_nce"


@dataclasses.dataclass
class LossStack(base_config.Config):
  """Common BatchNorm configs for all models."""

  bridge: Tuple[BaseLossConfig, Ellipsis] = ()
