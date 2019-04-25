using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mademy
{
    public interface IWeightInitializer
    {

        float GetRandomWeight(int numberOfInputWeights);
        float GetRandomBias();
    }

    public class DefaultWeightInitializer : IWeightInitializer
    {

        public float GetRandomWeight(int numberOfInputWeights)
        {
            return Utils.GetGaussianRandom(0, (float)Math.Sqrt(numberOfInputWeights));
        }

        public float GetRandomBias()
        {
            return Utils.GetGaussianRandom(0, 1);
        }
    }
}
