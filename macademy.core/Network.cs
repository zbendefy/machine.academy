﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading;
using Newtonsoft.Json;

namespace Macademy
{
    /// <summary>
    /// A neural network consisting of layers.
    /// </summary>
    [Serializable]
    public class Network : ISerializable
    {
        public class TrainingPromise
        {
            private static readonly int maxProgress = 100;

            private readonly object lockObj = new object();
            private int progress = 0;
            private int epochsDone = 0;

            private bool stopAtNextEpoch = false;

            internal void SetProgress(float _progress, int _epochsDone)
            {
                lock (lockObj)
                {
                    epochsDone = _epochsDone;
                    progress = (int)(_progress * 100.0f);
                }
            }

            /// <summary>
            /// Returns if the training is ready or not
            /// Can be called from any thread
            /// </summary>
            /// <returns>true if the training finished, false otherwise</returns>
            public bool IsReady()
            {
                lock (lockObj)
                {
                    return progress >= maxProgress;
                }
            }

            /// <summary>
            /// Returns a percentage value (0.0 to 1.0), indicating the total training progress
            /// </summary>
            /// <returns>returns a float between 0.0 and 1.0 indicating the trainig progress</returns>
            public float GetTotalProgress()
            {
                lock (lockObj)
                {
                    return (float)progress / (float)maxProgress;
                }
            }

            /// <summary>
            /// Returns how many epochs have been finished
            /// </summary>
            /// <returns>returns how many epochs have been finished</returns>
            public int GetEpochsDone() { return epochsDone; } //language guarantees atomic access

            /// <summary>
            /// If called, the training will stop when the next epoch finishes
            /// </summary>
            public void StopAtNextEpoch() { stopAtNextEpoch = true; }

            internal bool IsStopAtNextEpoch() { return stopAtNextEpoch; }

            /// <summary>
            /// Blocks the calling thread, and only returns once the training has been finished
            /// </summary>
            public void Await()
            {
                //TODO use C# async instead of thread.sleep
                while (true)
                {
                    lock (lockObj)
                    {
                        if (IsReady())
                            return;
                    }
                    Thread.Sleep(50);
                }
            }
        }

        internal string name;
        internal string description;
        internal List<Layer> layers;

        internal TrainingPromise trainingPromise = null;
        private Thread trainingThread = null;
        private readonly object lockObj_gradientAccess = new object();

        private Network(Network o)
        {
            this.name = new string(o.name);
            this.description = new string(o.description);
            layers = new List<Layer>();
            foreach(var layer in o.layers)
            {
                layers.Add(new Layer(layer));
            }
        }

        Network(List<Layer> layers)
        {
            this.layers = layers;
        }

        Network(SerializationInfo info, StreamingContext context)
        {
            name = (string)info.GetValue("name", typeof(string));
            description = (string)info.GetValue("description", typeof(string));
            layers = (List<Layer>)info.GetValue("layers", typeof(List<Layer>));
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("name", name, typeof(string));
            info.AddValue("description", description, typeof(string));
            info.AddValue("layers", layers, typeof(List<Layer>));
        }

        /// <summary>
        /// Attaches a name to the Network, that will be visible when exporting to a JSON file
        /// </summary>
        /// <param name="_name">The name tag to be attached</param>
        public void AttachName(string _name) { name = _name;  }

        /// <summary>
        /// Attaches a description to the Network, that will be visible when exporting to a JSON file
        /// </summary>
        /// <param name="_desc">The description string to be attached</param>
        public void AttachDescription(string _desc) { description = _desc;  }

        /// <summary>
        /// Returns the layer configuration of the network in this order:
        /// [input-layer neuron count][1st hidden layer neuron count][2nd hidden layer neuron count]...[nth hidden layer neuron count][output layer neuron count]
        /// </summary>
        /// <returns>The layer config containing the neuron counts of all layers including the input layer</returns>
        public int[] GetLayerConfig()
        {
            List<int> ret = new List<int>();
            ret.Add(layers.First().GetWeightsPerNeuron());
            foreach (var item in layers)
            {
                ret.Add(item.GetNeuronCount());
            }
            return ret.ToArray();
        }

        /// <summary>
        /// Returns the element count of the networks expected input
        /// </summary>
        /// <returns>The element count of the networks input</returns>
        public int GetInputSize() 
        {
            return layers.First().GetWeightsPerNeuron();
        }

        /// <summary>
        /// Returns the element count of the networks output
        /// </summary>
        /// <returns>The element count of the networks output</returns>
        public int GetOutputSize()
        {
            return layers.Last().GetNeuronCount();
        }

        public Network Copy()
        {
            return new Network(this);
        }

        private void TrainWithBackpropagation(TrainingSuite trainingSuite, int trainingDataBegin, int trainingDataEnd, ComputeDevice calculator)
        {
            //Calculate the accumulated gradient. Accumulated means, that the gradient has to be divided by the number of samples in the minibatch.
            List<List<NeuronData>> accumulatedGradient = null;
            accumulatedGradient = calculator.CalculateAccumulatedGradientForMinibatch(this, trainingSuite, trainingDataBegin, trainingDataEnd);
            float sizeDivisor = (float)(trainingDataEnd - trainingDataBegin) / (float)trainingSuite.trainingData.Count;

            //Calculate regularization terms based on the training configuration
            float regularizationTerm1 = 1.0f;
            float regularizationTerm2Base = 0.0f;
            if (trainingSuite.config.regularization == TrainingConfig.Regularization.L2)
            {
                regularizationTerm1 = 1.0f - trainingSuite.config.learningRate * (trainingSuite.config.regularizationLambda / (float)trainingSuite.trainingData.Count);
            }
            else if (trainingSuite.config.regularization == TrainingConfig.Regularization.L1)
            {
                regularizationTerm2Base = -((trainingSuite.config.learningRate * (trainingSuite.config.regularizationLambda / (float)trainingSuite.trainingData.Count)));
            }

            bool applyRegularizationTerm2 = trainingSuite.config.regularization == TrainingConfig.Regularization.L1;

            //Apply accumulated gradient to network (Gradient descent)
            float sizeDivisorAndLearningRate = sizeDivisor * trainingSuite.config.learningRate;
            for (int i = 0; i < layers.Count; ++i)
            {
                var layer = layers[i];
                var weightsPerNeuron = layer.GetWeightsPerNeuron();
                var layerNeuronCount = layer.GetNeuronCount();
                var weightMx = layer.weightMx;
                var biases = layer.biases;

                for (int j = 0; j < layerNeuronCount; ++j)
                {
                    var layerGradientWeights = accumulatedGradient[i][j].weights;
                    biases[j] -= accumulatedGradient[i][j].bias * sizeDivisorAndLearningRate;
                    for (int w = 0; w < weightsPerNeuron; ++w)
                    {
                        weightMx[j, w] = regularizationTerm1 * weightMx[j, w] - layerGradientWeights[w] * sizeDivisorAndLearningRate;
                        if (applyRegularizationTerm2)
                            weightMx[j, w] -= regularizationTerm2Base * Utils.Sign(weightMx[j, w]);
                    }
                }
            }
        }


        private void TrainWithEvolution(TrainingSuite trainingSuite, int trainingDataBegin, int trainingDataEnd, Network[] population, ComputeDevice calculator)
        {
            Dictionary<Network, float> error_acc = new Dictionary<Network, float>();
            for(int i = 0; i < population.Length; ++i)
            {
                if (i != 0) //Keep the best performer
                {
                    population[i].ApplyRandomNudge(trainingSuite.config.evolutionMutationRate);
                }
                error_acc.Add(population[i], 0);
            }


            for(int i = trainingDataBegin; i < trainingDataEnd; ++i)
            {
                foreach(var n in population)
                {
                    float[] result = n.Compute(trainingSuite.trainingData[i].input, calculator);
                    float error = trainingSuite.config.costFunction.CalculateSummedError(result, trainingSuite.trainingData[i].desiredOutput);
                    error_acc[n] += error;
                }
            }

            Array.Sort(population, (a,b)=>{return error_acc[a].CompareTo(error_acc[b]);});

            int population_cutoff = Math.Max(1, (int)((float)population.Length * trainingSuite.config.evolutionSurvivalRate));

            for(int i = population_cutoff; i < population.Length; ++i)
            {
                population[i] = population[i % population_cutoff].Copy();
            }
        }

        /// <summary>
        /// Trains the network using the given training suite and calculator
        /// The functions returns immediately with a promise object that can be used to monitor progress.
        /// Note: Using the network during training is not permitted.
        /// </summary>
        /// <param name="trainingSuite">The training suite to be used</param>
        /// <param name="calculator">The calculator (containing a compute device) to be used for calculations</param>
        /// <returns>A promise that can be used to check the </returns>
        public TrainingPromise Train(TrainingSuite trainingSuite, ComputeDevice calculator)
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");

            trainingPromise = new TrainingPromise();

            if (trainingSuite.config.epochs < 1)
            {
                trainingPromise.SetProgress(1, 0);
                return trainingPromise;
            }

            trainingThread = new Thread(() => {
                Network[] evolution_population = null;
                if(trainingSuite.config.trainingMode == TrainingConfig.TrainingMode.Evolution)
                {
                    evolution_population = new Network[trainingSuite.config.evolutionPopulationSize];
                    for(int i = 0; i < evolution_population.Length; ++i)
                    {
                        evolution_population[i] = new Network(this);
                    }
                }

                for (int currentEpoch = 0; currentEpoch < trainingSuite.config.epochs; currentEpoch++)
                {
                    if (trainingPromise.IsStopAtNextEpoch())
                        break;

                    if (trainingSuite.config.shuffleTrainingData )
                    {
                        Utils.ShuffleList(ref trainingSuite.trainingData);
                    }

                    int trainingDataBegin = 0;
                    int trainingDataEnd = trainingSuite.config.UseMinibatches() ? Math.Min( trainingSuite.config.miniBatchSize, trainingSuite.trainingData.Count) : trainingSuite.trainingData.Count;

                    while (true)
                    {
                        switch(trainingSuite.config.trainingMode)
                        {
                            case TrainingConfig.TrainingMode.Backpropagation:
                            TrainWithBackpropagation(trainingSuite, trainingDataBegin, trainingDataEnd, calculator);
                            break;
                            case TrainingConfig.TrainingMode.Evolution:
                            TrainWithEvolution(trainingSuite, trainingDataBegin, trainingDataEnd, evolution_population, calculator);
                            break;
                            default:
                            //error
                            break;
                        }

                        //Set up the next minibatch, or quit the loop if we're done.
                        if (trainingSuite.config.UseMinibatches())
                        {
                            if (trainingDataEnd >= trainingSuite.trainingData.Count)
                                break;

                            trainingPromise.SetProgress(((float)trainingDataEnd + ((float)currentEpoch * (float)trainingSuite.trainingData.Count)) / ((float)trainingSuite.trainingData.Count * (float)trainingSuite.config.epochs), currentEpoch + 1);

                            trainingDataBegin = trainingDataEnd;
                            trainingDataEnd = Math.Min(trainingDataEnd + trainingSuite.config.miniBatchSize, trainingSuite.trainingData.Count);
                        }
                        else
                        {
                            break;
                        }
                    }
                }

                calculator.FlushWorkingCache(); //Release any cache that the mathLib has built up.

                if(trainingSuite.config.trainingMode == TrainingConfig.TrainingMode.Evolution)
                {
                    this.layers = evolution_population[0].layers; //move the best performing layer without copying
                    evolution_population = null;
                }

                trainingPromise.SetProgress(1, trainingPromise.GetEpochsDone()); //Report that the training is finished
                trainingPromise = null;
            });

            trainingThread.Start();


            return trainingPromise;
        }

        public void ApplyRandomNudge(float nudge)
        {
            Random r = new Random();
            for (int i = 0; i < layers.Count; ++i)
            {
                var weights = layers[i].weightMx;
                var biases = layers[i].biases;

                int len0 = weights.GetLength(0);
                int len1 = weights.GetLength(1);
                for(int j = 0; j < len1; ++j)
                {
                    for(int k = 0; k < len0; ++k)
                    {
                        weights[k, j] += (((float)r.NextDouble()) * 2.0f - 1.0f) * nudge;
                    }    
                }

                for(int j = 0; j < biases.Length; ++j)
                {
                    biases[j] += (((float)r.NextDouble()) * 2.0f - 1.0f) * nudge;
                }
            }
        }

        /// <summary>
        /// Calculates an output of the network to a given input
        /// </summary>
        /// <param name="input">The input array. Must have the same number of elements as the input layer of the network</param>
        /// <param name="calculator">The calculator (containing a compute device) to use for calculations</param>
        /// <returns></returns>
        public float[] Compute(float[] input, ComputeDevice calculator )
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");
            if ( input == null || input.Length != layers.First().GetWeightsPerNeuron() )
                throw new Exception("Invalid input argument!");
            return calculator.EvaluateNetwork(input, this);
        }

        /// <summary>
        /// Create a network with specific weights and biases
        /// </summary>
        /// <param name="inputLayers">A structure containing the weights and biases for each network</param>
        /// <param name="activationFunction">The activation function used by the network</param>
        /// <returns></returns>
        public static Network CreateNetwork(List< Tuple< IActivationFunction, List< Tuple< List<float>, float> > > > inputLayers)
        {
            List<Layer> layers = new List<Layer>();
            foreach (var layerData in inputLayers)
            {
                int neuronCountInLayer = layerData.Item2.Count;
                int weightsPerNeuron = layerData.Item2[0].Item1.Count;

                if ( layers.Count > 0)
                {
                    if (weightsPerNeuron != layers.Last().biases.Length)
                        throw new Exception("Invalid layer config! Layer #" + layers.Count + " doesnt have the number of biases required by the previous layer!");
                    if (weightsPerNeuron != layers.Last().weightMx.GetLength(0))
                        throw new Exception("Invalid layer config! Layer #" + layers.Count + " doesnt have the number of weights required by the previous layer!");
                }

                float[,] weightMx = new float[neuronCountInLayer, weightsPerNeuron];
                float[] biases = new float[neuronCountInLayer];

                for (int i = 0; i < neuronCountInLayer; ++i)
                {
                    for (int j = 0; j < weightsPerNeuron; ++j)
                    {
                        weightMx[i, j] = layerData.Item2[i].Item1[j];
                    }
                    biases[i] = layerData.Item2[i].Item2;
                }

                layers.Add( new Layer(weightMx, biases, layerData.Item1) ); 
            }

            return new Network(layers);
        }

        /// <summary>
        /// Get a JSON representation of the network
        /// </summary>
        /// <returns>a JSON string containing the networks properties</returns>
        public string ExportToJSON()
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");
            string output = JsonConvert.SerializeObject(this);
            return output;
        }

        /// <summary>
        /// Create a network from a previously exported JSON
        /// </summary>
        /// <param name="jsonData">The JSON data to create the network from</param>
        /// <returns></returns>
        public static Network CreateNetworkFromJSON(string jsonData)
        {
            return JsonConvert.DeserializeObject<Network>(jsonData);
        }

        /// <summary>
        /// Create a network with random weights and biases
        /// </summary>
        /// <param name="layerConfig">The layer configuration containing the number of neurons in each layer in this order: [input layer][1st hidden layer][2nd hidden layer]...[nth hidden layer][output layer]</param>
        /// <param name="activationFunction">The activation function to use</param>
        /// <param name="weightInitializer">The weight and bias initializer to use</param>
        /// <returns></returns>
        public static Network CreateNetworkInitRandom(int[] layerConfig, IActivationFunction activationFunction, IWeightInitializer weightInitializer = null)
        {
            var initList = new List<Tuple<IActivationFunction, int>>();
            for(int i = 1; i < layerConfig.Length; ++i)
            {
                initList.Add(new Tuple<IActivationFunction, int>(activationFunction, layerConfig[i]));
            }
            return CreateNetworkInitRandom(layerConfig[0], initList, weightInitializer);
        }

        /// <summary>
        /// Create a network with random weights and biases
        /// </summary>
        /// <param name="input_layer">The neurons in the input layer</param>
        /// <param name="hidden_and_output_layers">The layer configuration containing the activation function, and the number of neurons in each layer in this order: [1st hidden layer][2nd hidden layer]...[nth hidden layer][output layer]</param>
        /// <param name="weightInitializer">The weight and bias initializer to use</param>
        /// <returns></returns>
        public static Network CreateNetworkInitRandom(int input_layer, List<Tuple<IActivationFunction, int>> hidden_and_output_layers, IWeightInitializer weightInitializer = null)
        {
            if (weightInitializer == null)
                weightInitializer = new DefaultWeightInitializer();

            var inputLayers = new List<Tuple<IActivationFunction, List<Tuple<List<float>, float>>>>();

            for(int layId = 0; layId < hidden_and_output_layers.Count; ++layId)
            {
                int prevLayerSize = layId == 0 ? input_layer : hidden_and_output_layers[layId - 1].Item2;
                int layerSize = hidden_and_output_layers[layId].Item2;
                IActivationFunction activationFunction = hidden_and_output_layers[layId].Item1;
                List<Tuple<List<float>, float>> neuronList = new List<Tuple<List<float>, float>>();
                for (int i = 0; i < layerSize; i++)
                {
                    List<float> weights = new List<float>();
                    for (int j = 0; j < prevLayerSize; ++j)
                    {
                        weights.Add(weightInitializer.GetRandomWeight(prevLayerSize));
                    }
                    neuronList.Add(new Tuple<List<float>, float>(weights, weightInitializer.GetRandomBias()));
                }
                inputLayers.Add(new Tuple<IActivationFunction, List<Tuple<List<float>, float>>>(activationFunction, neuronList));
            }

            return CreateNetwork(inputLayers);
        }

        public List<List<NeuronData>> __GetInternalConfiguration()
        {
            var ret = new List<List<NeuronData>>();
            foreach (var layer in layers)
            {
                var layerStruct = new List<NeuronData>();
                for (int i = 0; i < layer.GetNeuronCount(); ++i)
                {
                    var weights = new float[layer.GetWeightsPerNeuron()];
                    for (int j = 0; j < weights.Length; j++)
                    {
                        weights[j] = layer.weightMx[i,j];
                    }
                    layerStruct.Add(new NeuronData(weights, layer.biases[i]));
                }

                ret.Add(layerStruct);
            }
            return ret;
        }
    }
}
