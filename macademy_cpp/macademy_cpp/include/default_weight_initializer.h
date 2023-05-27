#include "i_weight_initializer.h"
#include "common.h"

namespace macademy {

// Also known as 'Glorot'. Recommended weight initializer for sigmoid and tanh. It is not recommended for ReLU.
class XavierWeightInitializer : public IWeightInitializer
{
    mutable std::default_random_engine m_generator;

  public:
    explicit XavierWeightInitializer(unsigned seed = 5489U) : m_generator(seed) {}

    virtual float GetRandomWeight(uint32_t input_weight_count) const override
    {
        // xavier weight initialization
        return GenerateGaussianRandom<float>(m_generator, 0, 1.0f / sqrtf(input_weight_count));
    }

    virtual float GetRandomBias() const override { return GenerateGaussianRandom<float>(m_generator, 0, 1); }
};

//  Also known as 'Kaiming'. Recommended weight initializer for ReLU.
class HeWeightInitializer : public IWeightInitializer
{
    mutable std::default_random_engine m_generator;

  public:
    explicit HeWeightInitializer(unsigned seed = 5489U) : m_generator(seed) {}

    virtual float GetRandomWeight(uint32_t input_weight_count) const override
    {
        // xavier weight initialization
        return GenerateGaussianRandom<float>(m_generator, 0, sqrtf(2.0f / float(input_weight_count)));
    }

    virtual float GetRandomBias() const override { return GenerateGaussianRandom<float>(m_generator, 0, 1); }
};
} // namespace macademy