"""Main script to run the project from dataloading to model creation and training."""
import pprint
import yaml
from absl import app
from absl import flags
from absl import logging
from experiments import finetune
from models import model_factory
from configs import factory as config_factory


with open('config.yaml', 'r') as f:
    configurations = yaml.load(f, Loader=yaml.FullLoader)

uvatt = model_factory.build_model(params = configurations)


FLAGS = flags.FLAGS

def get_params():
    """Constructs the configuration of the experiment"""
    params = configurations
    # task = 'finetune'
    # model_arch = 'UT_BASE'
    # params = config_factory.build_experiment_configs(
    #     task = task,
    #     model_arch = model_arch
    # )
    return params

def main(argv):
    del argv

    params = get_params()
    # logging.info(f"Model parameters are: {pprint.pformat(params)}")


    executor = finetune.get_executor(params = params)
    print(f"Executor is: {executor}")
   
    # print(f"Params mode set to: {params.mode}")
    return executor.run(mode = params.mode)

if __name__ == '__main__':
    app.run(main)