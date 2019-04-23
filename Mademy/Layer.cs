using CLMath;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    [Serializable]
    class Layer : ISerializable
    {
        public float[,] weightMx;
        public float[] biases;

        public Layer(float[,] weightMx, float[] biases)
        {
            this.weightMx = weightMx;
            this.biases = biases;

            if (weightMx.GetLength(0) != biases.GetLength(0))
                throw new Exception("Invalid layer!");
        }

        public int GetNeuronCount()
        {
            return biases.Length;
        }

        public int GetWeightsPerNeuron() { return weightMx.GetLength(1); }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("weightMx", weightMx);
            info.AddValue("biases", biases);
        }

        Layer(SerializationInfo info, StreamingContext context)
        {
            weightMx = (float[,])info.GetValue("weightMx", typeof(float[,]));
            biases = (float[])info.GetValue("biases", typeof(float[]));
        }

        public float[] Compute(MathLib mathLib, float[] input, bool applySigmoid = true)
        {
            if (applySigmoid)
                return mathLib.CalculateLayer(weightMx, biases, input, MathLib.SigmoidFunction.Sigmoid);
            else
                return mathLib.CalculateLayer(weightMx, biases, input, MathLib.SigmoidFunction.Passtrough);
        }

    }
}
