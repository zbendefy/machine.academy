#include "i_weight_initializer.h"
#include "common.h"

namespace macademy
{
    class DefaultWeightInitializer : IWeightInitializer
    {
        public: 
        virtual float GetRandomWeight(uint32_t input_weight_count) const override
        {
            //xavier weight initialization
            return GenerateGaussianRandom<float>(0, 1.0f / std::sqrtf(input_weight_count));
        }

        virtual float GetRandomBias() 
        {
            return GenerateGaussianRandom<float>(0, 1);
        }
    };
}