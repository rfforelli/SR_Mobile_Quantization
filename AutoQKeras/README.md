# AutoQKeras
AutoQKeras allows the automatic quantization and rebalancing of deep neural networks by treating quantization and rebalancing of an existing deep neural network as a hyperparameter search in Keras-Tuner using random search, hyperband or gaussian processes. More on these algorithms is defined [here](https://keras.io/api/keras_tuner/tuners/) 

The key addition in AutoQKeras is that optimization considers the quantization of the model so both the accuracy and model model size.

## The three ingredients for AutoQKeras are defined below
1. Model: Keras Tuner does not require a pretrained model though you could begin with predefined weights. This model is defined in [../solvers/networks/base7.py/MyHyperModel](../solvers/networks/base7.py)
2. Algorithm: We use RandomSearch and more on this [here](https://keras.io/api/keras_tuner/tuners/) 
3. Quantization Search Space: the search space that KerasTuner is going to look through is defined in the [quantization_config](./quantization_config.yaml) of the HyperModel. These are the quantizers to attempt on our hypermodel.

## Optimization Process
The tuner is defined in [autoq.py](autoq.py) and is trained by running one of several make commands in the [Makefile](./Makefile). This will do 3 things:
1. Test 100 model configurations each for 20 epochs and score them based on loss and model size and save the best model at completion.
2. Save the results to tensorboard
3. train the best quantized model to 200 total epochs. 20 during tuning and 180 during this further training.

### Run Makefile
The makefile allows you to choose the search algorithm and whether or not to start from a predefined float32 Keras model. 
`make <bayesian | random | hyperband>`: Uses the tuner to search for the model using one of the algorithms and train models from scratch.
`make pretrained_<bayesian | random | hyperband>`: Uses the tuner to search for the model using one of the algorithms and tunes the model from the keras weights.

## Visualizing results
### Tensorboard
TensorBoard is a visualization tool for machine learning experiments. 
Results from this training are automatically saved to a tensorboard directory. 
To visualize the results run: 
`tensorboard --logdir <project_directory>/tb_logs`
