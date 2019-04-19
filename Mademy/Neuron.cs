using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    [Serializable]
    class Neuron : ISerializable
    {
        public List<float> weights;
        public float bias;

        public Neuron(List<float> weights, float bias)
        {
            this.weights = weights;
            this.bias = bias;
        }

        Neuron(SerializationInfo info, StreamingContext context)
        {
            weights = (List<float>)info.GetValue("weights", typeof(List<float>));
            bias = (float)info.GetValue("bias", typeof(float));
        }

        public float Compute(List<float> input)
        {
            if (input.Count != weights.Count)
                throw new ArgumentException("Error! Invalid input for neutron!");

            float result = 0;
            for (int i = 0; i < input.Count; ++i)
            {
                result += weights[i] * input[i];
            }

            return result + bias;
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("weights", weights, typeof(List<float>));
            info.AddValue("bias", bias, typeof(float));
        }
    }
}
