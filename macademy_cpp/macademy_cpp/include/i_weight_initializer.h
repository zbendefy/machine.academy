#pragma once

namespace macademy {
class IWeightInitializer
{
  public:
    virtual ~IWeightInitializer() {}

    virtual float GetRandomWeight(uint32_t input_weight_count) const = 0;
    virtual float GetRandomBias() const = 0;
};
} // namespace macademy