using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using CLMath;
using MathNet.Numerics.LinearAlgebra;
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

            public bool IsReady()
            {
                lock (lockObj)
                {
                    return progress >= maxProgress;
                }
            }

            public float GetProgress()
            {
                lock (lockObj)
                {
                    return (float)progress / (float)maxProgress;
                }
            }

            internal void SetProgress(float _progress)
            {
                lock (lockObj)
                {
                    progress = (int)(_progress * 100.0f);
                }
            }
        }

        internal string name;
        internal string description;
        internal List<Layer> layers;

        internal TrainingPromise trainingPromise = null;
        private Thread trainingThread = null;
        private Object lockObj_gradientAccess = new object();

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

        void AttachName(string _name) { name = _name;  }
        void AttachDescription(string _desc) { description = _desc;  }
        
        public TrainingPromise Train(MathLib mathLib, TrainingSuite trainingSuite)
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");

            trainingPromise = new TrainingPromise();

            if (trainingSuite.config.epochs < 1)
            {
                trainingPromise.SetProgress(1);
                return trainingPromise;
            }

            trainingThread = new Thread(() => {
                for (int currentEpoch = 0; currentEpoch < trainingSuite.config.epochs; currentEpoch++)
                {

                    int trainingDataBegin = 0;
                    int trainingDataEnd = trainingSuite.config.UseMinibatches() ? trainingSuite.config.miniBatchSize : trainingSuite.trainingData.Count;

                    while (true)
                    {
                        List<List<NeuronData>> gradient = null;
                        if (trainingSuite.config.numThreads <= 1)
                            gradient = CalculateGradientSingleThread(mathLib, trainingSuite, trainingDataBegin, trainingDataEnd);
                        else
                            gradient = CalculateGradientMultiThreaded(mathLib, trainingSuite, trainingDataBegin, trainingDataEnd);

                        //Apply gradient to network
                        for (int i = 0; i < layers.Count; ++i)
                        {
                            var layer = layers[i];
                            for (int j = 0; j < layer.GetNeuronCount(); ++j)
                            {
                                layer.biases[j] -= gradient[i][j].bias * trainingSuite.config.learningRate;
                                for (int w = 0; w < layer.GetWeightsPerNeuron(); ++w)
                                {
                                    layer.weightMx[j, w] -= gradient[i][j].weights[w] * trainingSuite.config.learningRate;
                                }
                            }
                        }

                        if (trainingSuite.config.UseMinibatches())
                        {
                            if (trainingDataEnd >= trainingSuite.trainingData.Count)
                                break;

                            trainingPromise.SetProgress(((float)trainingDataEnd + ((float)currentEpoch * (float)trainingSuite.trainingData.Count)) / ((float)trainingSuite.trainingData.Count * (float)trainingSuite.config.epochs));

                            trainingDataBegin = trainingDataEnd;
                            trainingDataEnd = Math.Min(trainingDataEnd + trainingSuite.config.miniBatchSize, trainingSuite.trainingData.Count);
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                trainingPromise.SetProgress(1);
                trainingPromise = null;
            });

            trainingThread.Start();


            return trainingPromise;
        }

        private void CalculateOutputLayerGradient(MathLib mathLib, ref List<NeuronData> gradientData, ref List<float> gamma_k_vector, List<float[]> activations, float[] trainingInput, List<float[]> zValues, float[] desiredOutput)
        {
            var prevActivations = activations.Count <= 1 ? trainingInput : activations[activations.Count - 2];
            int lastLayerWeightCount = layers.Last().GetWeightsPerNeuron();
            int lastLayerNeuronCount = layers.Last().GetNeuronCount();
            for (int i = 0; i < lastLayerNeuronCount; i++)
            {
                float outputValue = activations.Last()[i];
                float gamma_k = (outputValue - desiredOutput[i]) * MathLib.SigmoidPrime(zValues.Last()[i]);

                var gradientDataItem = gradientData[i];
                //Assert(gradientData[i].weights.Length == prevActivations.Length);
                for (int j = 0; j < lastLayerWeightCount; j++)
                {
                    gradientDataItem.weights[j] += gamma_k * (prevActivations[j]);
                }
                gradientDataItem.bias += gamma_k;
                gamma_k_vector.Add(gamma_k);
            }
        }

        private void CalculateHiddenLayerGradient(MathLib mathLib, int L, ref List<NeuronData> gradientData, ref List<float> gamma_k_vector, float[] prevLayerActivations, List<float[]> zValues)
        {
            List<float> newGammak = new List<float>();
            int layerWeightCount = layers[L].GetWeightsPerNeuron();
            int layerNeuronCount = layers[L].GetNeuronCount();

            for (int i = 0; i < layerNeuronCount; i++)
            {
                float gamma_j = 0;
                //Assert(gamma_k_vector.Count == layers[L + 1].weightMx.GetLength(0));
                for (int k = 0; k < gamma_k_vector.Count; k++)
                {
                    gamma_j += gamma_k_vector[k] * layers[L+1].weightMx[k, i];
                }
                gamma_j *= MathLib.SigmoidPrime(zValues[L][i]);
                newGammak.Add(gamma_j);

                //Assert(gradientData[i].weights.Length == prevLayerActivations.Length);
                var gradientDataItem = gradientData[i];
                for (int j = 0; j < layerWeightCount; j++)
                {
                    gradientDataItem.weights[j] += gamma_j * (prevLayerActivations[j]);
                }
                gradientDataItem.bias += gamma_j;
            }

            gamma_k_vector = newGammak;
        }

        private void CalculateGradientThread(MathLib mathLib, TrainingSuite suite, ref List<List<NeuronData>> gradientVector, int threadId, int trainingDataBegin, int trainingDataEnd)
        {
            var myMathLib = mathLib.Clone();
            var intermediateResult = Utils.CreateGradientVector(this);
            int threadCount = suite.config.GetThreadCount();
            for (int i = trainingDataBegin + threadId; i < trainingDataEnd; i+= threadCount)
            {
                CalculateGradientForSingleTrainingExample(myMathLib, ref intermediateResult, suite.trainingData[i].input, suite.trainingData[i].desiredOutput);
            }

            lock (lockObj_gradientAccess)
            {
                for (int i = 0; i < gradientVector.Count; i++)
                {
                    for (int j = 0; j < gradientVector[i].Count; j++)
                    {
                        for (int k = 0; k < gradientVector[i][j].weights.Length; k++)
                        {
                            gradientVector[i][j].weights[k] += intermediateResult[i][j].weights[k];
                        }
                        gradientVector[i][j].bias += intermediateResult[i][j].bias;
                    }
                }
            }
        }

        private void CalculateGradientForSingleTrainingExample(MathLib mathLib, ref List<List<NeuronData>> intermediateResults, float[] trainingInput, float[] trainingDesiredOutput)
        {
            List<float[]> activations = new List<float[]>();
            List<float[]> zValues = new List<float[]>();
            Compute(mathLib, trainingInput, ref activations, ref zValues);

            var lastLayerGradient = intermediateResults.Last();
            List<float> delta_k_holder = new List<float>();
            CalculateOutputLayerGradient(mathLib, ref lastLayerGradient, ref delta_k_holder, activations, trainingInput, zValues, trainingDesiredOutput);

            for (int i = layers.Count - 2; i >= 0; --i)
            {
                var layerGradient = intermediateResults[i];
                CalculateHiddenLayerGradient(mathLib, i, ref layerGradient, ref delta_k_holder, i == 0 ? trainingInput : activations[i - 1], zValues);
            }
        }

        private void AdjustGradientVectorWithToAverage(ref List<List<NeuronData>> gradientVector, float multiplier)
        {
            for (int i = 0; i < gradientVector.Count; i++)
            {
                for (int j = 0; j < gradientVector[i].Count; j++)
                {
                    for (int k = 0; k < gradientVector[i][j].weights.Length; k++)
                    {
                        gradientVector[i][j].weights[k] *= multiplier;
                    }
                    gradientVector[i][j].bias *= multiplier;
                }
            }
        }

        private List<List<NeuronData>> CalculateGradientSingleThread(MathLib mathLib, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            //Backpropagation
            var ret = Utils.CreateGradientVector(this);

            for (int i = trainingDataBegin; i < trainingDataEnd; i++)
            {
                CalculateGradientForSingleTrainingExample(mathLib, ref ret, suite.trainingData[i].input, suite.trainingData[i].desiredOutput);
            }

            float sizeDivisor = 1.0f / (float)(trainingDataEnd - trainingDataBegin);
            AdjustGradientVectorWithToAverage(ref ret, sizeDivisor);

            return ret;
        }

        private List<List<NeuronData>> CalculateGradientMultiThreaded(MathLib mathLib, TrainingSuite suite, int trainingDataBegin, int trainingDataEnd)
        {
            //Backpropagation
            var ret = Utils.CreateGradientVector(this);

            Thread[] workerThreads = new Thread[suite.config.GetThreadCount()];

            for (int i = 0; i < workerThreads.Length; i++)
            {
                int idx = i;
                workerThreads[i] = new Thread(() => {
                    CalculateGradientThread(mathLib, suite, ref ret, idx, trainingDataBegin, trainingDataEnd);
                });
                workerThreads[i].Start();
            }

            for (int i = 0; i < workerThreads.Length; i++)
            {
                workerThreads[i].Join();
            }

            float sizeDivisor = 1.0f / (float)(trainingDataEnd - trainingDataBegin);
            AdjustGradientVectorWithToAverage(ref ret, sizeDivisor);

            return ret;
        }

        private float[] Compute(MathLib mathLib, float[] input, ref List<float[]> activations, ref List<float[]> zValues)
        {
            var current = input;
            foreach(var layer in layers)
            {
                bool applySigmoid = zValues == null;
                current = layer.Compute(mathLib, current, applySigmoid);
                if (zValues != null)
                {
                    zValues.Add((float[])current.Clone());
                    for (int i = 0; i < current.Length; i++)
                        current[i] = MathLib.Sigmoid(current[i]);
                }
                if (activations != null)
                    activations.Add(current);
            }
            return current;
        }

        public float[] Compute(MathLib mathLib, float[] input)
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");
            List<float[]> doesntNeedActivations = null;
            List<float[]> doesntNeedZ = null;
            return Compute(mathLib, input, ref doesntNeedActivations, ref doesntNeedZ);
        }

        public static Network CreateNetwork(List< List< Tuple< List<float>, float> > > inputLayers)
        {
            List<Layer> layers = new List<Layer>();
            foreach (var layerData in inputLayers)
            {
                int neuronCountInLayer = layerData.Count;
                int weightsPerNeuron = layerData[0].Item1.Count;

                float[,] weightMx = new float[neuronCountInLayer, weightsPerNeuron];
                float[] biases = new float[neuronCountInLayer];

                for (int i = 0; i < neuronCountInLayer; i++)
                {
                    for (int j = 0; j < weightsPerNeuron; j++)
                    {
                        weightMx[i, j] = layerData[i].Item1[j];
                    }
                    biases[i] = layerData[i].Item2;
                }

                layers.Add( new Layer(weightMx, biases) ); 
            }

            return new Network(layers);
        }

        public string GetTrainingDataJSON()
        {
            if (trainingPromise != null)
                throw new Exception("Cannot perform operation while training is in progress!");
            string output = JsonConvert.SerializeObject(this);
            return output;
        }

        public static Network LoadTrainingDataFromJSON(string jsonData)
        {
            return JsonConvert.DeserializeObject<Network>(jsonData);
        }

        public static Network CreateNetworkInitRandom(List<int> layerConfig)
        {
            List<List<Tuple<List<float>, float>>> inputLayers = new List<List<Tuple<List<float>, float>>>();

            for(int layId = 1; layId < layerConfig.Count; ++layId)
            {
                int prevLayerSize = layerConfig[layId - 1];
                int layerSize = layerConfig[layId];
                List<Tuple<List<float>, float>> neuronList = new List<Tuple<List<float>, float>>();
                for (int i = 0; i < layerSize; i++)
                {
                    List<float> weights = new List<float>();
                    for (int j = 0; j < prevLayerSize; ++j)
                    {
                        weights.Add(Utils.GetRandomWeight(prevLayerSize));
                    }
                    neuronList.Add(new Tuple<List<float>, float>(weights, Utils.GetRandomBias()));
                }
                inputLayers.Add(neuronList);
            }

            return CreateNetwork(inputLayers);
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("name", name, typeof(string));
            info.AddValue("description", description, typeof(string));
            info.AddValue("layers", layers, typeof(List<Layer>));
        }
    }
}
