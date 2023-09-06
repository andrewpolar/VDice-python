"""
Title: Probabilistic Bayesian Neural Networks
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/15
Last modified: 2021/01/15
Description: Building probabilistic Bayesian neural network models with TensorFlow Probability.

Code modified by Andrew Polar: 2023/07/29, the dataset is replaced by mathematically generated, 
the inputs are generated in get_train_and_test_splits, the outputs are generated in get_output.

The modeled system is two quantities of dice and probabilistic switch for choosing either of them.
Number of dice is chosen as np.random.randint(min_number_of_dice, max_number_of_dice), then the 
output is an outcome, which is a sum of a random roll. The probabilistic switch is generated as 
np.random.randint(min_dice, max_dice). 

Since data is programmatically generated, it is possible to compare predicted distributions of
the targets to actual, which are used for assessment of the accuracy of suggested BNN approach.
The accuracy metric is relative distance for two vectors of nested medians.
"""

from re import T
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp
import math
import matplotlib.pyplot as plt

FEATURE_NAMES = ["x0", "x1", "x2"]
dataset_size = 2100
train_size = 2000
hidden_units = [8, 8]
learning_rate = 0.003
batch_size = 256
num_epochs = 600
model_sample_size = 4096
monte_carlo_sample_size = 4096
min_dice = 1
max_dice = 11 #must be greater by 1 than expected maximum

#### Data generation block 

def get_output(z0, z1, z2):
    rnd = np.random.randint(min_dice, max_dice)
    if (rnd <= z0):
        s = 0
        for j in range(1, z1):
            s += np.random.randint(min_dice, max_dice)
        return s
    else:
        s = 0
        for j in range(1, z2):
            s += np.random.randint(min_dice, max_dice)
        return s

def get_Sample(z0, z1, z2, N):
    sample = np.empty(N, dtype=float)
    for j in range(N):
        sample[j] = get_output(z0, z1, z2)
    return sample

def get_MonteCarlo(examples):
    x0 = examples['x0'].numpy()
    x1 = examples['x1'].numpy()
    x2 = examples['x2'].numpy()
    list = []
    for i in range(x0.size):
        sample = get_Sample(x0[i], x1[i], x2[i], monte_carlo_sample_size)
        list.append(sample)
    return list

def get_train_and_test_splits(dataset_size, train_size, batch_size=1):
    x0 = [] 
    x1 = []
    x2 = []

    for j in range(dataset_size):
        x0.append(np.random.randint(min_dice, max_dice))
        x1.append(np.random.randint(min_dice, max_dice))
        x2.append(np.random.randint(min_dice, max_dice))
 
    y = np.empty(dataset_size, dtype=float)

    for idx in range(dataset_size):
        y[idx] = get_output(x0[idx], x1[idx], x2[idx])

    dataset = tf.data.Dataset.from_tensor_slices(({'x0':x0, 'x1':x1, 'x2':x2}, y))

    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)
    return train_dataset, test_dataset    

#### End data generation block

#### Data modeling

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def create_probablistic_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            #activation="sigmoid"
            activation="relu"
        )(features)
    distribution_params = layers.Dense(units=6)(features)     
     
    #next commented out operator was an original code
    #outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    #it is replaced by next nested callables for support of multimodal posteriors
    #units in previous operator should match number of estimated parameters,
    #for original operator it is 2.
    outputs = tfp.layers.DistributionLambda(lambda t:        
        tfp.distributions.Mixture(
            cat = tfp.distributions.Categorical(logits=t[:, :2]),
            components = 
            [
                tfp.distributions.Normal(loc=t[:, 2], scale=tf.math.softplus(t[:, 3])),
                tfp.distributions.Normal(loc=t[:, 4], scale=tf.math.softplus(t[:, 5]))
            ]
        )
        )(distribution_params)
  
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def run_experiment(model, loss, train_dataset, test_dataset):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

#### End data modeling

#### Functions for accuracy estimation

def relativeDistance(X, Y):
    if (len(X) != len(Y)):
        return -1
    dist = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for i in range(len(X)):
        dist += (X[i] - Y[i]) * (X[i] - Y[i])
        norm1 += X[i] * X[i]
        norm2 += Y[i] * Y[i]
    dist = math.sqrt(dist)
    norm1 = math.sqrt(norm1)
    norm2 = math.sqrt(norm2)
    norm = (norm1 + norm2) / 2.0
    return dist/norm

def medianSplit(x, depth, medians):
    x.sort()
    medians.sort()
    if (0 == depth): return
    size = int(len(x))
    median = 0
    if (0 == size % 2):
        median = (x[int(size/2 -1)] + x[int(size/2)]) / 2.0
    else:
        median = x[int(size/2)]
    medians.append(median)

    left = []
    right = []
    for i in range(size):
        if (i < int(size/2)): left.append(x[i])
        if (0 != size % 2):
            if (i > int(size/2)): right.append(x[i])
        else: 
            if (i >= int(size/2)): right.append(x[i])

    medianSplit(left, depth-1, medians)
    medianSplit(right, depth-1, medians)
    return

def BuildHistogram(data, bins, title):
    histogram = np.histogram(data, bins)
    plt.subplots(figsize=(5, 2.4))
    print(f'Histogram values')
    print(histogram)
    yy = list(histogram[0])
    xx = list(histogram[1])
    x2 = []
    for i in range(bins): x2.append((xx[i] + xx[i + 1])/2.0)
    plt.bar(x2, yy, width = (x2[bins-1] - x2[0])/24)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("frequencies")
    plt.show()
    return

#########################################################
# code execution
#########################################################

validation_size = dataset_size - train_size
train_dataset, test_dataset = get_train_and_test_splits(dataset_size, train_size, batch_size)
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(validation_size))[0]
prob_bnn_model = create_probablistic_bnn_model(train_size)
run_experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset)
prediction_distribution = prob_bnn_model(examples)
arr_monte_carlo = get_MonteCarlo(examples)

########################################################################
# at this point prediction is ready, further code is accuracy estimation
########################################################################

samples = prediction_distribution.sample(model_sample_size)

bins = 16
mean_median_dist = 0.0
for idx in range(validation_size):
    mediansX = []
    mediansY = []
    medianSplit(arr_monte_carlo[idx], 5, mediansX)
    medianSplit(samples[:, idx].numpy(), 5, mediansY)
    mediansX.sort()
    mediansY.sort()
    mean_median_dist += relativeDistance(mediansX, mediansY)
    if (0 == idx % 17):
        BuildHistogram(arr_monte_carlo[idx], bins, "Monte Carlo")
        BuildHistogram(samples[:, idx], bins, "Model")

mean_median_dist /= validation_size

print(f"The accuracy metric - average relative median distance = {mean_median_dist}")

# BNN accuracy test 09/03/2023
# (average median distances)     0.3323  0.3233  0.4102  0.3786  0.3981  0.3066  0.4392  0.3224
# the good numbers should be below 0.1
  
