"""Config definition for different types of heads."""

import dataclasses
from typing import Tuple

from configs import base_config


@dataclasses.dataclass
class BatchNormConfig(base_config.Config):
  """Common BatchNorm configs for all models."""

  momentum: float = 0.9
  epsilon: float = 1e-5
  scale: bool = True
  name: str = "batch_norm"


@dataclasses.dataclass
class BaseHeadConfig(base_config.Config):
  """Base configs for any type of heads."""

  name: str = "projection_head"


@dataclasses.dataclass
class MLPBridgeConfig(BaseHeadConfig):
  """Parameters of MLP-based One-Rest bridge head."""
  name: str = "mlp_bridge_module"
  modality: str = ""
  d_model: int = 512


@dataclasses.dataclass
class FACBridge(BaseHeadConfig):
  """Parameters for the MLP-based FAC-style bridge head."""

  name: str = "mlp_fac"
  bn_config: base_config.Config = BatchNormConfig()
  use_xreplica_bn: bool = True
  vid_to_aud_txt_kwargs: MLPBridgeConfig = MLPBridgeConfig(
      d_model=512,
      modality="video",
      name="video_mlp_module",
      )
  aud_to_vid_txt_kwargs: MLPBridgeConfig = MLPBridgeConfig(
      d_model=512,
      modality="audio",
      name="audio_mlp_module",
      )
  txt_to_vid_aud_kwargs: MLPBridgeConfig = MLPBridgeConfig(
      d_model=256,
      modality="text",
      name="text_mlp_module",
      )


@dataclasses.dataclass
class JointBridge(BaseHeadConfig):
  """Parameters for the MLP-based Joint-style bridge head."""

  name: str = "mlp_joint"
  bn_config: base_config.Config = BatchNormConfig()
  use_xreplica_bn: bool = True
  vid_to_aud_txt_kwargs: MLPBridgeConfig = MLPBridgeConfig(
      d_model=512,
      modality="video",
      name="video_mlp_module",
      )
  aud_to_vid_txt_kwargs: MLPBridgeConfig = MLPBridgeConfig(
      d_model=512,
      modality="audio",
      name="audio_mlp_module",
      )
  txt_to_vid_aud_kwargs: MLPBridgeConfig = MLPBridgeConfig(
      d_model=512,
      modality="text",
      name="text_mlp_module",
      )


@dataclasses.dataclass
class HeadStack(base_config.Config):
  """Stacked head configs."""

  bridge: Tuple[BaseHeadConfig, Ellipsis] = ()
