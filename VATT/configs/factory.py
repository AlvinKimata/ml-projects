"""Factory to provide experiment configs."""

from configs import audio as audio_config
from configs import experiment
from configs import multimodal as vatt_config
from configs import text as text_config
from configs import unified as unified_config
from configs import video as video_config


VID_MODELS = {
    'I3D': video_config.I3D,
    'VIT_BASE': video_config.ViTBase,
    'VIT_MEDIUM': video_config.ViTMedium,
    'VIT_LARGE': video_config.ViTLarge,
    'VIT_XLARGE': video_config.ViTXLarge,
}

AUD_MODELS = {
    'RESNET2D_50': audio_config.Resnet2D50,
    'RESNET2D_101': audio_config.Resnet2D101,
    'WAT_BASE': audio_config.WaTBase,
    'WAT_MEDIUM': audio_config.WaTMedium,
    'WAT_LARGE': audio_config.WaTLarge,
    'SPT_BASE': audio_config.SpTBase,
    'SPT_MEDIUM': audio_config.SpTMedium,
    'SPT_LARGE': audio_config.SpTLarge,
}

TXT_MODELS = {
    'LINEAR': text_config.LinearModel,
    'T5_SMALL': text_config.T5Small,
    'T5_BASE': text_config.T5Base,
    'BERT_SMALL': text_config.BertSmall,
    'BERT_BASE': text_config.BertBase,
}

UNIFIED_MODELS = {
    'UT_BASE': unified_config.UniTBase,
    'UT_MEDIUM': unified_config.UniTMedium,
    'UT_LARGE': unified_config.UniTLarge,
}

MULTIMODAL_MODELS = {
    # Backbone:CNN, Head:MLP
    'MMV_FAC': vatt_config.MMVFACModel,
    # Backbone:Tx, Head:MLP
    'TX_FAC': vatt_config.TxFACModel,
    # Backbone:Tx, Head:MLP/Attn
    'UT_FAC': vatt_config.UnifiedTxFACModel,
}

EXPERIMENT_CONFIGS = {
    'PRETRAIN': experiment.Pretrain,
    'FINETUNE': experiment.Finetune,
}


def build_experiment_configs(task,
                             model_arch):
  """Get default configs by architecture and data set name."""
  task = task.upper()
  model_arch = model_arch.upper()

  if task not in EXPERIMENT_CONFIGS:
    raise ValueError('{!r} not found in {!r}'.format(task,
                                                     EXPERIMENT_CONFIGS.keys()))

  config = EXPERIMENT_CONFIGS[task](
      model_config=build_model_configs(model_arch)
      )

  return config


def build_model_configs(model_arch):
  """Get default configs by architecture and data set name."""
  model_arch = model_arch.upper()

  if model_arch in VID_MODELS:
    config = VID_MODELS[model_arch]()

  elif model_arch in AUD_MODELS:
    config = AUD_MODELS[model_arch]()

  elif model_arch in TXT_MODELS:
    config = TXT_MODELS[model_arch]()

  elif model_arch in UNIFIED_MODELS:
    config = UNIFIED_MODELS[model_arch]()

  elif model_arch in MULTIMODAL_MODELS:
    config = MULTIMODAL_MODELS[model_arch]()

  else:
    raise ValueError('Unknown module name {!r}'.format(model_arch))

  return config
