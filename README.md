# machine.academy

Neural network training library written in C# as a learning project.

### Features:
 * GPU acceleration (using OpenCL)
 * Cost functions: Mean-Squared, Cross-entropy
 * Optional L1 and L2 regularization
 * Optimized weight and bias initialization
 * Stochastic gradient descent with shuffled mini-batches


### TODO:
 * Code refactor and cleanup
 * New features: Dropout, Softmax layers
 * Convolutional layer
 * LSTM networks


## Projects in the repo
### Macademy

A Class library that provides Neural network training.

#### Sample code:

Initializing a random network
```
var layerConfig = new int[]{ 784, 32, 32, 10 };
var network = Network.CreateNetworkInitRandom(layerConfig, new SigmoidActivation(), new DefaultWeightInitializer());
```

Initializing from JSON:
```
string jsonData = "{ ... }";
var network = Network.CreateNetworkFromJSON( jsonData ); 
```


Exporting to JSON
```
string json = network.GetNetworkAsJSON();
```


Calculating an output
```
float[] input = { ...};
float[] results = network.Compute( new MathLib(), input );
```

Training
```
//todo: write example
```

## Resources

These are some nice resources for learning about Neural Networks:
 * [Michael A. Nielsen, Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
 * [3Blue1Brown's But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
