using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Mademy
{
    public class Network
    {
        List<Layer> layers;

        Network(List<Layer> layers)
        {
            this.layers = layers;
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
    }
}
