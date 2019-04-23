using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    static class Utils
    {
        private static Random rnd = new Random();

        public static float FastSigmoid(float x)
        {
            return x / (1.0f + Math.Abs(x));
        }

        public static float FastSigmoidD(float x)
        {
            float div = (1.0f + Math.Abs(x));
            return 1 / (div*div);
        }

        public static float ReLU(float x)
        {
            return Math.Max(0, x);
        }

        public static float Sigmoid(float x)
        {
            return x / (1.0f + (float)Math.Pow(Math.E, -x));
        }

        public static float CalculateError(float output, float expectedOutput)
        {
            var diff = output - expectedOutput;
            return diff * diff;
        }

        public static float GetRandomWeight(int numberOfInputWeights)
        {
            return GetGaussianRandom(0, (float)Math.Sqrt( (double)numberOfInputWeights)); 
        }

        public static float GetRandomBias()
        {
            return GetGaussianRandom(0, 1);
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
