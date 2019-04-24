using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    interface IActivationFunction
    {
        float Calculate(float x);
        float CalculatePrime(float x);
        int GetOpenCLFunctionId();
    }

    class SigmoidActivationFunction : IActivationFunction
    {
        public float Calculate(float x)
        {
            return x / (1.0f + (float)Math.Pow(Math.E, -x));
        }

        public float CalculatePrime(float x)
        {
            throw new NotImplementedException();
        }

        public int GetOpenCLFunctionId()
        {
            throw new NotImplementedException();
        }
    }
}
