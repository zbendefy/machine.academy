using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
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

            public TrainingConfig CreateTrainingConfig()
            {
                var ret = new TrainingConfig();
                ret.miniBatchSize = DontSubdivideBatches;
                learningRate = 0.001f;
                return ret;
            }

            public bool UseMinibatches() { return miniBatchSize != DontSubdivideBatches; }
        };

        string name;
        string description;
        List<Layer> layers;

        List<float[]> gammaValues;

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

        public void Train(List<Tuple<List<float>, List<float>>> trainingData, TrainingConfig config)
        {
            int trainingDataBegin = 0;
            int trainingDataEnd = config.UseMinibatches() ? config.miniBatchSize : trainingData.Count;

            while(true)
            {
                var gradient = CalculateGradient(trainingData, trainingDataBegin, trainingDataEnd);

                //Apply gradient to network
                for (int i = 0; i < layers.Count; ++i)
                {
                    var layer = layers[i];
                    for (int j = 0; j < layer.neurons.Count; ++j)
                    {
                        layer.neurons[j].bias -= gradient[i][j].bias * config.learningRate;
                        for (int w = 0; w < layer.neurons[j].weights.Count; ++w)
                        {
                            layer.neurons[j].weights[w] -= gradient[i][j].weights[w] * config.learningRate;
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

        private float CalculateZL(Neuron neuron, List<float> prevActivations)
        {
            float zL = 0;
            for (int i = 0; i < neuron.weights.Count; i++)
            {
                zL += neuron.weights[i] * prevActivations[i];
            }
            zL += neuron.bias;
            return zL;
        }

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

                /*
                float sensitivityToPrevActivation = 0;
                for (int i = 0; i < activations[L - 1].Count; i++)
                {
                    sensitivityToPrevActivation += weights[i] * dal_dzl * dC_dal;
                }
                dC_dal = sensitivityToPrevActivation;
                */
            }
        }

        private List<List<NeuronData>> CalculateGradient(List<Tuple<List<float>, List<float>>> trainingData, int trainingDataBegin, int trainingDataEnd)
        {
            //Backpropagation
            var ret = new List<List<NeuronData>>();
            {
                for (int i = 0; i < layers.Count; i++)
                {
                    var nlist = new List<NeuronData>();
                    for (int j = 0; j < layers[i].neurons.Count; j++)
                    {
                        var wlist = new List<float>();
                        for (int k = 0; k < layers[i].neurons[j].weights.Count; k++)
                        {
                            wlist.Add(0);
                        }
                        nlist.Add(new NeuronData(wlist, 0));
                    }
                    ret.Add(nlist);
                }
            }

            for (int t = trainingDataBegin; t < trainingDataEnd; ++t)
            {
                List<List<float>> activations = new List<List<float>>();
                List<List<float>> zValues = new List<List<float>>();
                Compute(trainingData[t].Item1, ref activations, ref zValues);
                CalculateNudge(ref ret, activations, zValues, trainingData[t]);
            }
            return ret;
        }

        private List<float> Compute(List<float> input, ref List<List<float>> activations, ref List<List<float>> zValues)
        {
            var current = input;
            foreach(var layer in layers)
            {
                List<float> zvalueList = null;
                if (zValues != null)
                {
                    zvalueList = new List<float>();
                    zValues.Add(zvalueList);
                }

                current = layer.Compute(current, ref zvalueList);
                if (activations != null)
                    activations.Add(current);
            }
            return current;
        }

        public List<float> Compute(List<float> input)
        {
            List<List<float>> doesntNeedActivations = null;
            List<List<float>> doesntNeedZValues = null;
            return Compute(input, ref doesntNeedActivations, ref doesntNeedZValues);
        }

        public static Network CreateNetwork(List< List< Tuple< List<float>, float> > > inputLayers)
        {
            List<Layer> layers = new List<Layer>();
            foreach (var layerData in inputLayers)
            {
                List<Neuron> neurons = new List<Neuron>();
                foreach (var neuronData in layerData)
                {
                    neurons.Add(new Neuron(neuronData.Item1, neuronData.Item2));
                }
                layers.Add( new Layer(neurons) ); 
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
