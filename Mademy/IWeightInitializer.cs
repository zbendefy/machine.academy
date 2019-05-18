using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Macademy
{
    /// <summary>
    /// Interface for initializing weights and biases
    /// </summary>
    public interface IWeightInitializer
    {
        float GetRandomWeight(int numberOfInputWeights);
        float GetRandomBias();
    }

    /// <summary>
    /// Initializes weights using Gaussian random (std.dev: 1.0/sqrt(inputWeightCount), mean: 0) and biases with Gaussian random (std.dev: 1, mean: 0)
    /// </summary>
    public sealed class DefaultWeightInitializer : IWeightInitializer
    {
        public float GetRandomWeight(int numberOfInputWeights)
        {
            return Utils.GetGaussianRandom(0, 1.0f / ((float)Math.Sqrt(numberOfInputWeights)));
        }

        public float GetRandomBias()
        {
            return Utils.GetGaussianRandom(0, 1);
        }
    }
}
