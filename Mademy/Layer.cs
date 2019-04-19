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
        public List<Neuron> neurons;

        public Layer(List<Neuron> neurons)
        {
            this.neurons = neurons;
        }

        Layer(SerializationInfo info, StreamingContext context)
        {
            neurons = (List<Neuron>)info.GetValue("neurons", typeof(List<Neuron>));
        }

        public List<float> Compute(List<float> input)
        {
            List<float> zValues = null;
            return Compute(input, ref zValues);
        }

        public List<float> Compute(List<float> input, ref List<float> zValues)
        {
            var ret = new List<float>();
            foreach(var n in neurons)
            {
                ret.Add( Utils.FastSigmoid( n.Compute(input)));
                if (zValues != null)
                {
                    zValues.Add(n.Compute(input));
                }
            }
            return ret;
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("neurons", neurons, typeof(List<Neuron>));
        }
    }
}
