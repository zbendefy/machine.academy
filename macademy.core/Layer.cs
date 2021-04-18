using System;
using System.Runtime.Serialization;

namespace Macademy
{
    [Serializable]
    internal class Layer : ISerializable
    {
        public float[,] weightMx;
        public float[] biases;
        public IActivationFunction activationFunction;

        internal Layer(Layer o)
        {
            this.activationFunction = o.activationFunction;
            this.weightMx = (float[,])o.weightMx.Clone();
            this.biases = (float[])o.biases.Clone();
        }

        public Layer(float[,] weightMx, float[] biases, IActivationFunction activationFunction)
        {
            this.weightMx = weightMx;
            this.biases = biases;
            this.activationFunction = activationFunction;

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
            info.AddValue("activationFunction", activationFunction.GetSerializedName(), typeof(string));
            info.AddValue("weightMx", weightMx);
            info.AddValue("biases", biases);
        }

        Layer(SerializationInfo info, StreamingContext context)
        {
            weightMx = (float[,])info.GetValue("weightMx", typeof(float[,]));
            biases = (float[])info.GetValue("biases", typeof(float[]));

            try
            {
                var activationFunctionName = (string)info.GetValue("activationFunction", typeof(string));
                activationFunction = Utils.GetActivationFunctionFromString(activationFunctionName);
            }
            catch (System.Exception)
            {
                activationFunction = new SigmoidActivation(); //compatibility with old formats
            }
        }
    }
}
