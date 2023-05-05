#include "i_weight_initializer.h"
#include "common.h"

namespace macademy {
class DefaultWeightInitializer : public IWeightInitializer
{
    mutable std::default_random_engine m_generator;

  public:
    explicit DefaultWeightInitializer(unsigned seed = 5489U) : m_generator(seed) {}

    virtual float GetRandomWeight(uint32_t input_weight_count) const override
    {
        // xavier weight initialization
        return GenerateGaussianRandom<float>(m_generator, 0, 1.0f / std::sqrtf(input_weight_count));
    }

    virtual float GetRandomBias() const override { return GenerateGaussianRandom<float>(m_generator, 0, 1); }
};
} // namespace macademy