using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using CLMath;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;

namespace Mademy
{
    [Serializable]
    public class Network : ISerializable
    {
        public struct TrainingConfig
        {
            public static readonly int DontSubdivideBatches = -1;

            public int miniBatchSize;
            public float learningRate;

            public static TrainingConfig CreateTrainingConfig()
            {
                var ret = new TrainingConfig();
                ret.miniBatchSize = 1000;
                ret.learningRate = 0.001f;
                return ret;
            }

            public bool UseMinibatches() { return miniBatchSize != DontSubdivideBatches; }
        };

        string name;
        string description;
        List<Layer> layers;


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
        
        public void Train(MathLib mathLib, List<Tuple<float[], float[]>> trainingData, TrainingConfig config)
        {
            int trainingDataBegin = 0;
            int trainingDataEnd = config.UseMinibatches() ? config.miniBatchSize : trainingData.Count;

            while(true)
            {
                var gradient = CalculateGradient(mathLib, trainingData, trainingDataBegin, trainingDataEnd);

                //Apply gradient to network
                for (int i = 0; i < layers.Count; ++i)
                {
                    var layer = layers[i];
                    for (int j = 0; j < layer.GetNeuronCount(); ++j)
                    {
                        layer.biases[j] -= gradient[i][j].bias * config.learningRate;
                        for (int w = 0; w < layer.GetWeightsPerNeuron(); ++w)
                        {
                            layer.weightMx[j, w] -= gradient[i][j].weights[w] * config.learningRate;
                        }
                    }
                }

                if (config.UseMinibatches())
                {
                    if (trainingDataEnd >= trainingData.Count)
                        break;

                    trainingDataBegin = trainingDataEnd;
                    trainingDataEnd = Math.Min( trainingDataEnd + config.miniBatchSize, trainingData.Count);
                }
                else
                {
                    break;
                }
            }
        }

        private void Assert(bool v)
        {
            if (!v)
                throw new Exception("baaad");
        }

        private void CalculateOutputLayerGradient(MathLib mathLib, ref List<NeuronData> gradientData, ref List<float> gamma_k_vector, List<float[]> activations, List<float[]> zValues, float[] desiredOutput)
        {
            var prevActivations = activations[activations.Count - 2];
            for (int i = 0; i < layers.Last().GetNeuronCount(); i++)
            {
                float outputValue = activations.Last()[i];
                float gamma_k = (outputValue - desiredOutput[i]) * MathLib.SigmoidPrime(zValues.Last()[i]);

                Assert(gradientData[i].weights.Length == prevActivations.Length);
                for (int j = 0; j < layers.Last().GetWeightsPerNeuron(); j++)
                {
                    gradientData[i].weights[j] = gamma_k * (prevActivations[j]);
                }
                gradientData[i].bias = gamma_k;
                gamma_k_vector.Add(gamma_k);
            }
        }

        private void CalculateHiddenLayerGradient(MathLib mathLib, int L, ref List<NeuronData> gradientData, ref List<float> gamma_k_vector, float[] prevLayerActivations, List<float[]> zValues)
        {
            List<float> newGammak = new List<float>();
            for (int i = 0; i < layers[L].GetNeuronCount(); i++)
            {
                float gamma_j = 0;
                Assert(gamma_k_vector.Count == layers[L + 1].weightMx.GetLength(0));
                for (int k = 0; k < gamma_k_vector.Count; k++)
                {
                    gamma_j += gamma_k_vector[k] * layers[L+1].weightMx[k, i];
                }
                gamma_j *= MathLib.SigmoidPrime(zValues[L][i]);
                newGammak.Add(gamma_j);

                Assert(gradientData[i].weights.Length == prevLayerActivations.Length);
                for (int j = 0; j < layers[L].GetWeightsPerNeuron(); j++)
                {
                    gradientData[i].weights[j] = gamma_j * (prevLayerActivations[j]);
                }
                gradientData[i].bias = gamma_j;
            }

            gamma_k_vector = newGammak;
        }

        private List<List<NeuronData>> CalculateGradient(MathLib mathLib, List<Tuple<float[], float[]>> trainingData, int trainingDataBegin, int trainingDataEnd)
        {
            //Backpropagation
            var ret = new List<List<NeuronData>>();
            {
                for (int i = 0; i < layers.Count; i++)
                {
                    var nlist = new List<NeuronData>();
                    for (int j = 0; j < layers[i].GetNeuronCount(); j++)
                    {
                        var wlist = new float[layers[i].GetWeightsPerNeuron()];
                        nlist.Add(new NeuronData(wlist, 0));
                    }
                    ret.Add(nlist);
                }
            }

            var intermediate = new List<List<NeuronData>>();
            {
                for (int i = 0; i < layers.Count; i++)
                {
                    var nlist = new List<NeuronData>();
                    for (int j = 0; j < layers[i].GetNeuronCount(); j++)
                    {
                        var wlist = new float[layers[i].GetWeightsPerNeuron()];
                        nlist.Add(new NeuronData(wlist, 0));
                    }
                    intermediate.Add(nlist);
                }
            }

            float sizeDivisor = 1.0f / (float)(trainingDataEnd - trainingDataBegin);
            for (int t = trainingDataBegin; t < trainingDataEnd; ++t)
            {
                List<float[]> activations = new List<float[]>();
                List<float[]> zValues = new List<float[]>();
                Compute(mathLib, trainingData[t].Item1, ref activations, ref zValues);

                var lastLayerGradient = intermediate.Last();
                List<float> delta_k_holder = new List<float>();
                CalculateOutputLayerGradient(mathLib, ref lastLayerGradient, ref delta_k_holder, activations, zValues, trainingData[t].Item2);

                for (int i = layers.Count-2; i >= 0; --i)
                {
                    var layerGradient = intermediate[i];
                    CalculateHiddenLayerGradient(mathLib, i, ref layerGradient, ref delta_k_holder, i == 0 ? trainingData[t].Item1 : activations[i - 1], zValues);
                }


                for (int i = 0; i < intermediate.Count; i++)
                {
                    for (int j = 0; j < intermediate[i].Count; j++)
                    {
                        for (int k = 0; k < intermediate[i][j].weights.Length; k++)
                        {
                            ret[i][j].weights[k] += intermediate[i][j].weights[k] * sizeDivisor;
                        }
                        ret[i][j].bias += intermediate[i][j].bias * sizeDivisor;
                    }
                }

            }
            return ret;
        }

        /*
        private float CalculateZL(float[,] weightMx, int neuronId, float bias, float[] prevActivations)
        {
            float zL = 0;
            for (int i = 0; i < weightMx.GetLength(1); i++)
            {
                zL += weightMx[neuronId, i] * prevActivations[i];
            }
            zL += bias;
            return zL;
        }*/

        /*

        private float CalculateDC_DA(List<float> layerActivation, List<float> expectedResults)
        {
            float ret = 0;
            for (int i = 0; i < layerActivation.Count; i++)
            {
                ret += layerActivation[i] - expectedResults[i];
            }
            return 2.0f * ret;
        }



        private void CalculateGradientOutputLayer(ref List<List<NeuronData>> results, List<List<float>> activations, ref List<float> gammaValues , List<List<float>> zValues, Tuple<List<float>, List<float>> trainingData)
        {
            Layer layer = layers.Last();

            for (int i = 0; i < layer.neurons.Count; i++)
                gammaValues.Add((activations.Last()[i] - trainingData.Item2[i]) * Utils.FastSigmoidD(activations.Last()[i]));

            for (int i = 0; i < layer.neurons.Count; i++)
            {
                for (int j = 0; j < layer.neurons[i].weights.Count; j++)
                {
                    results.Last()[i].weights[j] = gammaValues[i] + 
                }
            }


        }


        private void CalculateNudge(ref List<List<NeuronData>> results, List<List<float>> activations, List<List<float>> zValues, Tuple<List<float>, List<float>> trainingData)
        {






            var layer = layers.Last();
            int L = layers.Count - 1;

            float dC_dal = CalculateDC_DA(activations.Last(), trainingData.Item2);//not sure if function is correct

            for (int j = 0; j < layer.neurons.Count; j++)
            {
                var neuron = layer.neurons[j];

                float dal_dzl = Utils.FastSigmoidD(CalculateZL(neuron, activations[L - 1]));

                for (int k = 0; k < neuron.weights.Count; k++)
                {
                    float dzl_dw = activations[L - 1][k]; //checked in wolfram alpha: d/dw (w*a+u*t+b) = a
                    float sensitivityToWeight = dzl_dw * dal_dzl * dC_dal;
                    results[L][j].weights[k] += sensitivityToWeight;
                }

                float sensitivityToBias = dal_dzl * dC_dal;
                results[L][j].bias += sensitivityToBias;

            }
        }

        */

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
                    for (int j = 0; j < prevLayerSize; j++)
                    {
                        weights.Add(Utils.GetRandomValue());
                    }
                    neuronList.Add(new Tuple<List<float>, float>(weights, Utils.GetRandomValue()));
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
