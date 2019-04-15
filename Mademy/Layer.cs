using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    class Layer
    {
        public List<Neuron> neurons;

        public Layer(List<Neuron> neurons)
        {
            this.neurons = neurons;
        }

        public List<float> Compute(List<float> input)
        {
            var ret = new List<float>();
            foreach(var n in neurons)
            {
                ret.Add(n.Compute(input));
            }
            return ret;
        }
    }
}
