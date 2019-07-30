# machine.academy

Neural network training library written in C# as a learning project.

### [Handwritten digit recognizer](http://htmlpreview.github.io/?https://github.com/zbendefy/machine.academy/blob/master/WebApps/NumberRecognize/index.html)

## Features:
 * GPU acceleration (using OpenCL)
 * Cost functions: 
    * Mean-Squared
    * Cross-entropy
 * Activation functions:
    * Sigmoid
 * Optional L1 and L2 regularization
 * Xavier weight initialization
 * Stochastic gradient descent with shuffled mini-batches

## Sample code:

### Initializing a random network:
```
var layerConfig = new int[]{ 784, 32, 32, 10 };
var network = Network.CreateNetworkInitRandom( layerConfig, new SigmoidActivation() );
```

### Initializing from JSON:
```
string jsonData = "{ ... }";
var network = Network.CreateNetworkFromJSON( jsonData ); 
```


### Exporting to JSON:
```
string json = network.ExportToJSON();
```

### Calculating an output to a given input:
```
float[] input = { ...};
float[] results = network.Compute( input, new Calculator() );
```

### Training
```
List<TrainingSuite.TrainingData> trainingData = new List<TrainingSuite.TrainingData>();

//Set up training examples
for (int i = 0; i < 10000; ++i)
{
    float[] trainingInput = ...;
    float[] desiredOutput = ...;
    trainingData.Add( new TrainingSuite.TrainingData( trainingInput, desiredOutput ) );
}

TrainingSuite trainingSuite = new TrainingSuite( trainingData );

//Set up training configuration
trainingSuite.config.epochs = 100;
trainingSuite.config.shuffleTrainingData = true;
trainingSuite.config.miniBatchSize = 50;
trainingSuite.config.learningRate = 0.005f;
trainingSuite.config.costFunction = new CrossEntropyErrorFunction();
trainingSuite.config.regularization = TrainingSuite.TrainingConfig.Regularization.L2;
trainingSuite.config.regularizationLambda = 0.5f;

var trainingPromise = network.Train( trainingSuite, new Calculator() );

trainingPromise.Await();
```

## Dependencies
All dependencies are set up in nuget.
* OpenCL.Net
* Newtonsoft.Json

### Future plans:
 * .NET Core port
 * Optimize memory layout (to reduce number of buffer copies)
 * New features: Dropout, Softmax layers
 * More activation functions
 * Convolutional layer
 * LSTM networks

## Projects in the repo

### Macademy

The neural network library

### NumberRecognizer

A sample app that can load the MNIST digit dataset and trains a network.

### Sandbox

A test app with no specific purpose but to try our some features using a ui

## Resources

These are some nice resources for learning about Neural Networks:
 * [Michael A. Nielsen, Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
 * [3Blue1Brown's But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
