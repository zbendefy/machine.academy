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
        private static float minVal = -100;
        private static float maxVal = 100;

        public static float FastSigmoid(float x)
        {
            return x / (1.0f + Math.Abs(x));
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

        public static float GetRandomValue()
        {
            float ret = (float)rnd.NextDouble();
            return (ret * (maxVal - minVal)) + minVal;
        }
    }
}
