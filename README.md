# machine.academy

Neural network training library written in C# (.net Core 3.1) as a learning project.

## Demos:

### [Handwritten digit recognizer web app](https://zbendefy.github.io/machine.academy/WebApps/NumberRecognize/index.html)
### [Car racing with Evolutional algorithm](https://zbendefy.github.io/machine.academy/WebApps/Evo/index.html)

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
float[] input = { ... };
ComputeDevice computeDevice = ComputeDeviceFactory.GetFirstOpenCLDevice(); //Get an OpenCL device
float[] results = network.Compute( input, computeDevice );
```

### Training
```
ComputeDevice computeDevice = ComputeDeviceFactory.GetFirstOpenCLDevice(); //Get an OpenCL device
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

var trainingPromise = network.Train( trainingSuite, computeDevice );

trainingPromise.Await();
```
## Projects in the repo

### Macademy

The neural network library

### OpenCl.DotNetCore.Interop

Part of the following library: https://github.com/lecode-official/opencl-dotnet. Used for OpenCL access from .net Core

### Testing

Tests for the library

### TestConsole

A console command line for testing

### NumberRecognizer (uses WinForms)

A sample app that can load the MNIST digit dataset and trains a network.

### Sandbox (uses WinForms)

A test app for development with no specific purpose but to try our some features using a ui.

### docs/WebApps/NumberRecognizer

An HTML web application that uses an already trained network to recognize a drawn number.

### docs/WebApps/Evo

An HTML web application that uses neural networks and an evolutional algorithm for training racecars to learn driving on various racetracks.

## Dependencies
The following dependencies are set up in nuget:
* Newtonsoft.Json

This project also includes parts of `opencl-dotnet`: https://github.com/lecode-official/opencl-dotnet

## Future plans:
 * Optimize memory layout (to reduce number of buffer copies)
 * New features: Dropout, Softmax layers
 * More activation functions
 * Convolutional layer
 * LSTM networks

## Resources

These are some nice resources for learning about Neural Networks:
 * [Michael A. Nielsen, Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
 * [3Blue1Brown's But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
