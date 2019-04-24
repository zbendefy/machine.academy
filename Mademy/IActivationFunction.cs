using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    public abstract class IActivationFunction
    {
        public abstract float Calculate(float x);
        public abstract float CalculatePrime(float x);
        public abstract int GetOpenCLFunctionId();
        public string GetSerializedName() { return GetType().Name; }
    }

    public class PasstroughActivation : IActivationFunction
    {
        public override float Calculate(float x)
        {
            return x;
        }

        public override float CalculatePrime(float x)
        {
            return 0;
        }

        public override int GetOpenCLFunctionId()
        {
            return 0;
        }
    }

    public class SigmoidActivation : IActivationFunction
    {
        public override float Calculate(float x)
        {
            return x / (1.0f + (float)Math.Pow(Math.E, -x));
        }

        public override float CalculatePrime(float x)
        {
            float sigm = Calculate(x);
            return sigm * (1.0f - sigm);
        }

        public override int GetOpenCLFunctionId()
        {
            return 1;
        }
    }
}
