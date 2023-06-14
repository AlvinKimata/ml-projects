"""Defines internal data_loader of YouTube datasets."""

import functools
from typing import Any, Dict, Optional, Text

from data.datasets import toy_dataset, fakeavceleb_dataset


# DMVR-based factories
DS_TO_FACTORY = {
    #########################################
    #### put your dataset factories here ####
    #########################################
    'TOY_DS': toy_dataset.ToyFactory,
    'FAKEAVCELEB_DATASET': fakeavceleb_dataset.ToyFactory
}


def get_ds_factory(dataset_name = 'fakeavceleb_dataset',
                   override_args = None):
  """Gets dataset source and name and returns its factory class."""

  dataset_name = dataset_name.upper()

  ds = DS_TO_FACTORY[dataset_name]

  if override_args:
    return functools.partial(ds, **override_args)
  else:
    return ds
