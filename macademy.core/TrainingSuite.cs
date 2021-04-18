using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Macademy
{

        /// <summary>
        /// Contains configurable aspects of the training procedure
        /// </summary>
        public class TrainingConfig
        {
            public static readonly int DontSubdivideBatches = -1;

            public enum TrainingMode { Backpropagation, Evolution }
            public enum Regularization { None, L1, L2 }

            public TrainingMode trainingMode = TrainingMode.Backpropagation;

            /// <summary>
            /// Size of the minibatch. 
            /// An epoch will take around (numberOfTrainingDatas / miniBatchSize) gradient descent steps
            /// 
            /// If "DontSubdivideBatches" is used, Stochastic gradient descent will not be used meaning 
            /// that an epoch will use all training datas to make a single gradient descent step.
            /// </summary>
            public int miniBatchSize = 1000;

            /// <summary>
            /// The learning rate
            /// After a gradient is calculated on a minibatch, the gradient descent will take a step along the 
            /// gradient's direction. This direction is multiplied by the leraningRate.
            /// 
            /// Larger learningRate values make the network learn faster, but too high values 
            /// will cause the learning to overshoot small vallies
            /// </summary>
            public float learningRate = 0.01f;

            /// <summary>
            /// How many epochs to train for
            /// </summary>
            public int epochs = 1;

            /// <summary>
            /// If true, the training examples will be shuffled before starting each epoch.
            /// If enabled, the network encounters a larger number of input combinations during a minibatch
            /// If disabled, the learning process will be more deterministic and reproducable
            /// </summary>
            public bool shuffleTrainingData = true;

            /// <summary>
            /// The cost function to use on the network's output
            /// </summary>
            public IErrorFunction costFunction = new CrossEntropyErrorFunction();

            /// <summary>
            /// Regularization techniques help the network learn numerically smaller weights and biases
            /// Smaller weights and biases can force the network to generalize about the data instead of
            /// learning about the test data peculiarities
            /// </summary>
            public Regularization regularization = Regularization.L2;

            /// <summary>
            /// The lamdba to use if L1 or L2 regularization is enabled
            /// Larger values force the network more to prefer smaller weights and biases.
            /// </summary>
            public float regularizationLambda = 0.01f;

            public int evolutionPopulationSize = 50;
            public float evolutionSurvivalRate = 0.2f;
            public float evolutionMutationRate = 0.1f;

            internal bool UseMinibatches() { return miniBatchSize != DontSubdivideBatches; }
        };

    /// <summary>
    /// Contains all data and parameter required for training a network
    /// </summary>
    public class TrainingSuite
    {
        public class TrainingData
        {
            public float[] input;
            public float[] desiredOutput;

            /// <summary>
            /// Constructs a training example for the training process
            /// </summary>
            /// <param name="input">The input values for the network. The number of elements in this vector must match the number of neurons in the networks input layer of the trained network!</param>
            /// <param name="desiredOutput">The desired output values to the given input. The number of elements in this vector must match the number of neurons in the last (outout) layer of the trained network!</param>
            public TrainingData(float[] input, float[] desiredOutput)
            {
                this.input = input;
                this.desiredOutput = desiredOutput;
            }
        };

        /// <summary>
        /// Configurable parameters of the training process
        /// </summary>
        public TrainingConfig config = new TrainingConfig();

        public List<TrainingData> trainingData;

        public TrainingSuite(List<TrainingData> trainingDatas)
        {
            this.trainingData = trainingDatas;
        }
    }
}
