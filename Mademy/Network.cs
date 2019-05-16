using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading;
using Newtonsoft.Json;

namespace Mademy
{
    [Serializable]
    public class Network : ISerializable
    {
        public class TrainingPromise
        {
            private static readonly int maxProgress = 100;

            private object lockObj = new object();
            private int progress = 0;
            private int epochsDone = 0;

            private bool stopAtNextEpoch = false;

            public bool IsReady()
            {
                lock (lockObj)
                {
                    return progress >= maxProgress;
                }
            }

            public float GetTotalProgress()
            {
                lock (lockObj)
                {
                    return (float)progress / (float)maxProgress;
                }
            }

            internal void SetProgress(float _progress, int _epochsDone)
            {
                lock (lockObj)
                {
                    epochsDone = _epochsDone;
                    progress = (int)(_progress * 100.0f);
                }
            }
            public int GetEpochsDone() { return epochsDone; } //language guarantees atomic access

            public void StopAtNextEpoch() { stopAtNextEpoch = true; }

            internal bool IsStopAtNextEpoch() { return stopAtNextEpoch; }
        }

        internal string name;
        internal string description;
        internal List<Layer> layers;
        internal string activationFunctionName = "";

        internal IActivationFunction activationFunction;
        internal TrainingPromise trainingPromise = null;
        private Thread trainingThread = null;
        private Object lockObj_gradientAccess = new object();

        Network(List<Layer> layers, IActivationFunction activationFunction)
        {
            this.layers = layers;
            this.activationFunction = activationFunction;
            this.activationFunctionName = activationFunction.GetSerializedName();
        }

        Network(SerializationInfo info, StreamingContext context)
        {
            name = (string)info.GetValue("name", typeof(string));
            description = (string)info.GetValue("description", typeof(string));
            activationFunctionName = (string)info.GetValue("activationFunctionName", typeof(string));
            layers = (List<Layer>)info.GetValue("layers", typeof(List<Layer>));

            Type t = Type.GetType("Mademy."+activationFunctionName);
            activationFunction = (IActivationFunction)Activator.CreateInstance(t);
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("name", name, typeof(string));
            info.AddValue("description", description, typeof(string));
            info.AddValue("activationFunctionName", activationFunctionName, typeof(string));
            info.AddValue("layers", layers, typeof(List<Layer>));
        }

        public void AttachName(string _name) { name = _name;  }

        public void AttachDescription(string _desc) { description = _desc;  }

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

        public TrainingPromise Train(MathLib mathLib, TrainingSuite trainingSuite)
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
                for (int currentEpoch = 0; currentEpoch < trainingSuite.config.epochs; currentEpoch++)
                {
                    if (trainingPromise.IsStopAtNextEpoch())
                        break;

                    if (trainingSuite.config.shuffleTrainingData )
                    {
                        Utils.ShuffleList(ref trainingSuite.trainingData);
                    }

                    int trainingDataBegin = 0;
                    int trainingDataEnd = trainingSuite.config.UseMinibatches() ? trainingSuite.config.miniBatchSize : trainingSuite.trainingData.Count;

                    while (true)
                    {
                        //Calculate the accumulated gradient. Accumulated means, that the gradient has to be divided by the number of samples in the minibatch.
                        List<List<NeuronData>> accumulatedGradient = null;
                        accumulatedGradient = mathLib.CalculateAccumulatedGradientForMinibatch(this, trainingSuite, trainingDataBegin, trainingDataEnd);
                        float sizeDivisor = (float)(trainingDataEnd - trainingDataBegin) / (float)trainingSuite.trainingData.Count;

                        //Calculate regularization terms based on the training configuration
                        float regularizationTerm1 = 1.0f;
                        float regularizationTerm2Base = 0.0f;
                        if (trainingSuite.config.regularization == TrainingSuite.TrainingConfig.Regularization.L2)
                        {
                            regularizationTerm1 = 1.0f - trainingSuite.config.learningRate * (trainingSuite.config.regularizationLambda / (float)trainingSuite.trainingData.Count);
                        }
                        else if (trainingSuite.config.regularization == TrainingSuite.TrainingConfig.Regularization.L1)
                        {
                            regularizationTerm2Base = -((trainingSuite.config.learningRate * (trainingSuite.config.regularizationLambda / (float)trainingSuite.trainingData.Count)));
                        }

                        bool applyRegularizationTerm2 = trainingSuite.config.regularization == TrainingSuite.TrainingConfig.Regularization.L1;

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

                mathLib.FlushWorkingCache(); //Release any cache that the mathLib has built up.

                trainingPromise.SetProgress(1, trainingPromise.GetEpochsDone()); //Report that the training is finished
                trainingPromise = null;
            });

            trainingThread.Start();


            return trainingPromise;
        }

        internal float[] Compute(MathLib mathLib, float[] input, ref List<float[]> activations, ref List<float[]> zValues, bool flushMathlibWorkingCache)
        {
            var current = input;
            bool applySigmoid = zValues == null;
            PasstroughActivation passtroughActivation = applySigmoid ? null : new PasstroughActivation();

            foreach(var layer in layers)
            {
                current = layer.Compute(mathLib, current, applySigmoid ? activationFunction : passtroughActivation);
                if (zValues != null)
                {
                    zValues.Add((float[])current.Clone());
                    for (int i = 0; i < current.Length; i++)
                        current[i] = activationFunction.Calculate(current[i]);
                }
                if (activations != null)
                    activations.Add(current);
            }

            if (flushMathlibWorkingCache)
                mathLib.FlushWorkingCache();

            return current;
        }

        public float[] Compute(MathLib mathLib, float[] input)
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");
            List<float[]> doesntNeedActivations = null;
            List<float[]> doesntNeedZ = null;
            return Compute(mathLib, input, ref doesntNeedActivations, ref doesntNeedZ, true);
        }

        public static Network CreateNetwork(List< List< Tuple< List<float>, float> > > inputLayers, IActivationFunction activationFunction)
        {
            List<Layer> layers = new List<Layer>();
            foreach (var layerData in inputLayers)
            {
                int neuronCountInLayer = layerData.Count;
                int weightsPerNeuron = layerData[0].Item1.Count;

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
                        weightMx[i, j] = layerData[i].Item1[j];
                    }
                    biases[i] = layerData[i].Item2;
                }

                layers.Add( new Layer(weightMx, biases) ); 
            }

            return new Network(layers, activationFunction);
        }

        public string GetNetworkAsJSON()
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");
            string output = JsonConvert.SerializeObject(this);
            return output;
        }

        public static Network CreateNetworkFromJSON(string jsonData)
        {
            return JsonConvert.DeserializeObject<Network>(jsonData);
        }

        public static Network CreateNetworkInitRandom(int[] layerConfig, IActivationFunction activationFunction, IWeightInitializer weightInitializer = null)
        {
            if (weightInitializer == null)
                weightInitializer = new DefaultWeightInitializer();

            List<List<Tuple<List<float>, float>>> inputLayers = new List<List<Tuple<List<float>, float>>>();

            for(int layId = 1; layId < layerConfig.Length; ++layId)
            {
                int prevLayerSize = layerConfig[layId - 1];
                int layerSize = layerConfig[layId];
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
                inputLayers.Add(neuronList);
            }

            return CreateNetwork(inputLayers, activationFunction);
        }
    }
}
