using System;
using System.Collections.Generic;

namespace Macademy
{
    internal static class Utils
    {
        private static Random rnd = new Random();

        public static void ShuffleList<T>(ref List<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rnd.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static float Sign(float x)
        {
            if (x < 0)
                return -1;
            else if (x > 0)
                return 1;
            return 0;
        }

        public static List<List<NeuronData>> CreateGradientVector(Network network)
        {
            var ret = new List<List<NeuronData>>();
            {
                for (int i = 0; i < network.layers.Count; i++)
                {
                    var nlist = new List<NeuronData>();
                    for (int j = 0; j < network.layers[i].GetNeuronCount(); j++)
                    {
                        var wlist = new float[network.layers[i].GetWeightsPerNeuron()];
                        nlist.Add(new NeuronData(wlist, 0));
                    }
                    ret.Add(nlist);
                }
            }
            return ret;
        }

        public static float GetGaussianRandom(float mean, float stdDev)
        {
            float u1 = 1.0f - (float)rnd.NextDouble(); //uniform(0,1] random doubles
            float u2 = 1.0f - (float)rnd.NextDouble();
            float randStdNormal = (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            float randNormal = mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
            return randNormal;
        }
    }

    public class NeuronData
    {
        public float[] weights;
        public float bias;

        public NeuronData(float[] weights, float bias)
        {
            this.weights = weights;
            this.bias = bias;
        }
    }
}
