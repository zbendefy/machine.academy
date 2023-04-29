#pragma once
#include <vector>
#include <optional>

#include "common.h"

namespace macademy {
struct TrainingData
{
    std::vector<float> m_input; ///< The input values for the network. The number of elements in this vector must match the number of neurons in the networks input layer of the trained network!
    std::vector<float>
        m_desired_output; ///< The desired output values to the given input. The number of elements in this vector must match the number of neurons in the last (outout) layer of the trained network!
};

struct TrainingSuite
{
    std::vector<TrainingData> m_training_data;

    /// <summary>
    /// Size of the minibatch.
    /// An epoch will take around (numberOfTrainingDatas / miniBatchSize) gradient descent steps
    ///
    /// If not specified, Stochastic gradient descent will not be used meaning
    /// that an epoch will use all training datas to make a single gradient descent step.
    /// </summary>
    std::optional<uint64_t> m_mini_batch_size;

    /// <summary>
    /// The learning rate
    /// After a gradient is calculated on a minibatch, the gradient descent will take a step along the
    /// gradient's direction. This direction is multiplied by the leraningRate.
    ///
    /// Larger learningRate values make the network learn faster, but too high values
    /// will cause the learning to overshoot small vallies
    /// </summary>
    float m_learning_rate = 0.01f;

    /// <summary>
    /// How many epochs to train for
    /// </summary>
    uint32_t m_epochs = 1;

    /// <summary>
    /// If true, the training examples will be shuffled before starting each epoch.
    /// If enabled, the network encounters a larger number of input combinations during a minibatch
    /// If disabled, the learning process will be more deterministic and reproducable
    /// </summary>
    bool m_shuffle_training_data = true;

    /// <summary>
    /// The cost function to use on the network's output
    /// </summary>
    CostFunction m_cost_function = CostFunction::CrossEntropy;

    /// <summary>
    /// Regularization techniques help the network learn numerically smaller weights and biases
    /// Smaller weights and biases can force the network to generalize about the data instead of
    /// learning about the test data peculiarities
    /// </summary>
    Regularization m_regularization = Regularization::L2;

    /// <summary>
    /// The lamdba to use if L1 or L2 regularization is enabled
    /// Larger values force the network more to prefer smaller weights and biases.
    /// </summary>
    float m_regularization_lambda = 0.01f;
};
} // namespace macademy