import sys
sys.path.append('../../SR_Mobile_Quantization')
import argparse
import os
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])
#   except RuntimeError as e:
#     print(e)
from solvers.networks.base7 import MyHyperModel
import keras_tuner as kt
import pdb
def restore(args):
  hypermodel = MyHyperModel()
  tuner = kt.RandomSearch(
      hypermodel,
      objective="val_loss",
      max_trials=100,
      executions_per_trial=1,
      directory="./",
      project_name=args.directory,
  )
  import pdb
  pdb.set_trace()
  qmodel = tuner.get_best_models(1)[0]

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Restore the Tuner Session')
   parser.add_argument('-d', '--directory', required=True)
   args = parser.parse_args()
   restore(args)

