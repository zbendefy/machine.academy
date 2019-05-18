using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Macademy
{
    public abstract class IActivationFunction
    {
        public abstract float Calculate(float x);
        public abstract float CalculatePrime(float x);
        public abstract int GetOpenCLFunctionId();
        public string GetSerializedName() { return GetType().Name; }
    }

    public sealed class PasstroughActivation : IActivationFunction
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

    public sealed class SigmoidActivation : IActivationFunction
    {
        public override float Calculate(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
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
