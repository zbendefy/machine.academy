using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Macademy
{
    /// <summary>
    /// An interface for error functions
    /// </summary>
    public abstract class IErrorFunction
    {
        public float CalculateSummedError(float[] output, float[] desiredOutput)
        {
            float err = 0;
            for(int i = 0; i < output.Length; ++i)
            {
                err += CalculateError(output[i], desiredOutput[i]);
            }
            return err;
        }

        public abstract float CalculateError(float output, float desiredOutput);
        public abstract float CalculateDelta(float z, float a, float desiredOutput, IActivationFunction activationFunction);
        public abstract int GetOpenCLFunctionID();
    }

    /// <summary>
    /// Mean squared error function
    /// </summary>
    public sealed class MeanSquaredErrorFunction : IErrorFunction
    {
        public override float CalculateDelta(float z, float a, float desiredOutput, IActivationFunction activationFunction)
        {
            return (a - desiredOutput) * activationFunction.CalculatePrime(z);
        }

        public override float CalculateError(float output, float desiredOutput)
        {
            float v = output - desiredOutput;
            return 0.5f * v * v;
        }

        public override int GetOpenCLFunctionID()
        {
            return 0;
        }
    }

    /// <summary>
    /// Cross entropy error function.
    /// Avoids slow learning on steep gradients of the sigmoid function
    /// </summary>
    public sealed class CrossEntropyErrorFunction : IErrorFunction
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
