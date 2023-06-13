"""Main script to run the project from dataloading to model creation and training."""
import pprint
import yaml
from absl import app
from absl import flags
from absl import logging
# from experiments import finetune
from models import model_factory
# from configs import factory as config_factory


with open('config.yaml', 'r') as f:
    configurations = yaml.load(f, Loader=yaml.FullLoader)

    # print(f'Model configurations are: {configurations}')

uvatt = model_factory.build_model(params = configurations)


# print(f"Uvatt model architecture: {uvatt.summary()}")
for layer in uvatt.layers:
    print(layer)

# def get_params():
#     """Constructs the configuration of the experiment"""
#     params = configurations
#     # params = config_factory.build_experiment_configs(
#     #     task = task,
#     #     model_arch = model_arch
#     # )
#     return params

# def main(argv):
#     del argv

#     params = get_params()
#     logging.info(f"Model parameters are: {pprint.pformat(params.as_dict())}")


#     # executor = finetune.get_executor(params = params)
   
#     print(f"Params mode set to: {params.mode}")
#     # return executor.run(mode = params.mode)

# if __name__ == '__main__':
#     app.run(main)