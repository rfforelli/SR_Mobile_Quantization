# KerasTuner
KerasTuner is simple, hyperparameter optimization framework that accelerates the hyperparameter search. Simply edit the configuration and search space and allow KerasTuner to handle the hard work. KerasTuner comes with Bayesian Optimization, Hyperband, and Random Search algorithms built-in which allow you to customize how the Tuner searches through the search space. More on these algorithms is defined [here](https://keras.io/api/keras_tuner/tuners/) 

## The three ingredients for Keras-Tuner are defined below
1. Model: Keras Tuner does not require a pretrained model though you could begin with predefined weights. This model is defined in [../solvers/networks/base7.py/MyHyperModel](../solvers/networks/base7.py)
2. Algorithm: We use RandomSearch and more on this [here](https://keras.io/api/keras_tuner/tuners/) 
3. Search Space: the search space that KerasTuner is going to look through is defined in the [build function](../solvers/networks/base7.py) of the HyperModel. These are the quantizers to attempt on our hypermodel.

## Running the Tuner
The tuner is defined in [tuner.py](tuner.py) and is trained by running `make keras_tuner`. This will do 3 things:
1. Run the tuner to find the model with the best weights and save that model
2. Save the results to tensorboard
3. train the best quantized model to 200 total epochs. 20 during tuning and 180 during this further training.

## Visualizing results
### Tensorboard
TensorBoard is a visualization tool for machine learning experiments. 
Results from this training are automatically saved to a tensorboard directory. 
To visualize the results run: 
`tensorboard --logdir results/tb_logs`
