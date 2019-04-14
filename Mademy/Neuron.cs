using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    class Neuron
    {
        List<float> weights;
        float bias;

        public Neuron(List<float> weights, float bias)
        {
            this.weights = weights;
            this.bias = bias;
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

            return Utils.Sigmoid( result + bias );
        }
    }
}
