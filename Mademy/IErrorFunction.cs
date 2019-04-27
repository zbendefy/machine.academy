using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    public abstract class IErrorFunction
    {
        public abstract float CalculateError(float output, float desiredOutput);
        public abstract float CalculateDelta(float z, float a, float desiredOutput, IActivationFunction activationFunction);
        public abstract int GetOpenCLFunctionID();
    }

    public class MeanSquaredError : IErrorFunction
    {
        public override float CalculateDelta(float z, float a, float desiredOutput, IActivationFunction activationFunction)
        {
            return (a - desiredOutput) * activationFunction.CalculatePrime(z);
        }

        public override float CalculateError(float output, float desiredOutput)
        {
            float v = output - desiredOutput;
            return 0.5f * v*v;
        }

        public override int GetOpenCLFunctionID()
        {
            return 0;
        }
    }

    public class CrossEntropy : IErrorFunction
    {
        public override float CalculateDelta(float z, float a, float desiredOutput, IActivationFunction activationFunction)
        {
            return a - desiredOutput;
        }

        public override float CalculateError(float output, float desiredOutput)
        {
            return -desiredOutput * (float)Math.Log(output) - (1.0f-desiredOutput)*(float)Math.Log(1-output);
        }

        public override int GetOpenCLFunctionID()
        {
            return 1;
        }
    }
}
