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

            public TrainingConfig CreateTrainingConfig()
            {
                var ret = new TrainingConfig();
                ret.miniBatchSize = DontSubdivideBatches;
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
                        layer.neurons[j].bias += gradient[i][j].Item2;
                        for (int w = 0; w < layer.neurons[j].weights.Count; ++w)
                        {
                            layer.neurons[j].weights[w] += gradient[i][j].Item1[w];
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

        private List<List<Tuple<List<float>, float>>> CalculateGradient(List<Tuple<List<float>, List<float>>> trainingData, int trainingDataBegin, int trainingDataEnd)
        {
            float error = 0;
            foreach (var item in trainingData)
            {
                var output = Compute(item.Item1);
                //error += Utils.CalculateError(output, item.Item2);
            }
            return new List<List<Tuple<List<float>, float>>>();
        }

        public List<float> Compute(List<float> input)
        {
            var current = input;
            foreach(var layer in layers){
                current = layer.Compute(current);
            }
            return current;
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
